import os
from typing import Optional

import torch

from videollama3.videollama3_trainer import VideoLLaMA3Trainer


class VideoLLaMA3DiffusionTrainer(VideoLLaMA3Trainer):
    def __init__(
        self,
        diffusion_head=None,
        vae=None,
        diffusion_loss_weight: float = 1.0,
        *args,
        **kwargs,
    ):
        self.diffusion_head = diffusion_head
        self.vae = vae
        self.diffusion_loss_weight = diffusion_loss_weight
        self._last_loss_logs = {}
        super().__init__(*args, **kwargs)
        if self.diffusion_head is not None:
            self.diffusion_head.to(self.model.device)
            if hasattr(self, "accelerator"):
                self.diffusion_head = self.accelerator.prepare_model(self.diffusion_head)
        if self.vae is not None:
            self.vae.to(self.model.device)

    def _get_diffusion_head_module(self):
        if self.diffusion_head is None:
            return None
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self.diffusion_head)
        return getattr(self.diffusion_head, "module", self.diffusion_head)

    def _encode_diffusion_target(self, diffusion_images: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise ValueError("Diffusion supervision requires a VAE.")
        images = diffusion_images.to(device=self.model.device, dtype=torch.float32)
        self.vae = self.vae.to(self.model.device)
        target_grid = int(getattr(self.model.config, "diffusion_target_spatial", 21))
        unshuffle_factor = int(getattr(self.model.config, "diffusion_pixel_unshuffle", 2))

        def _encode_frames(frames: torch.Tensor) -> torch.Tensor:
            posterior = self.vae.encode(frames).latent_dist
            latent = (posterior.sample() - self.vae.shift_factor) * self.vae.scaling_factor
            if unshuffle_factor > 1:
                latent = torch.nn.functional.pixel_unshuffle(latent, downscale_factor=unshuffle_factor)
            if latent.shape[-2:] != (target_grid, target_grid):
                raise ValueError(
                    f"Unexpected diffusion latent grid {tuple(latent.shape[-2:])}; expected {(target_grid, target_grid)}"
                )
            k, c, h, w = latent.shape
            return latent.permute(0, 2, 3, 1).reshape(k * h * w, c).contiguous()

        # Run VAE in float32 regardless of the outer autocast context.
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
            return _encode_frames(images)

    def _extract_diffusion_conditions(self, model, outputs) -> list[list[torch.Tensor]]:
        if outputs.hidden_states is None:
            raise RuntimeError("output_hidden_states=True is required for diffusion supervision.")
        hidden_states = outputs.hidden_states[-1]
        owner = None
        candidates = [model, self.model]
        seen = set()
        while candidates:
            candidate = candidates.pop(0)
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            if hasattr(candidate, "_diffusion_condition_mask"):
                owner = candidate
                break
            module = getattr(candidate, "module", None)
            if module is not None:
                candidates.append(module)
            get_base_model = getattr(candidate, "get_base_model", None)
            if callable(get_base_model):
                candidates.append(get_base_model())
        if owner is None:
            owner = model
        condition_mask = getattr(owner, "_diffusion_condition_mask", None)
        condition_segment_ids = getattr(owner, "_diffusion_condition_segment_ids", None)
        condition_token_indices = getattr(owner, "_diffusion_condition_token_indices", None)
        full_segment_lengths = getattr(owner, "_diffusion_full_segment_lengths", None)
        if condition_mask is None or condition_segment_ids is None or condition_token_indices is None:
            raise RuntimeError("Video token condition mask was not produced by the multimodal forward.")
        condition_mask = condition_mask.to(hidden_states.device)
        condition_segment_ids = condition_segment_ids.to(hidden_states.device)
        condition_token_indices = condition_token_indices.to(hidden_states.device)

        conditions_by_sample = []
        for sample_idx, (sample_hidden, sample_mask, sample_segment_ids, sample_token_indices) in enumerate(
            zip(hidden_states, condition_mask, condition_segment_ids, condition_token_indices)
        ):
            cur_hidden = sample_hidden[sample_mask]
            cur_segment_ids = sample_segment_ids[sample_mask]
            cur_token_indices = sample_token_indices[sample_mask]
            sample_conditions = []
            if cur_segment_ids.numel() > 0:
                for segment_id in range(int(cur_segment_ids.max().item()) + 1):
                    segment_mask = cur_segment_ids == segment_id
                    segment_tokens = cur_hidden[segment_mask]
                    if segment_tokens.numel() > 0:
                        sample_conditions.append(
                            {
                                "tokens": segment_tokens,
                                "indices": cur_token_indices[segment_mask],
                            }
                        )
            if full_segment_lengths is not None and sample_idx < len(full_segment_lengths):
                sample_lengths = full_segment_lengths[sample_idx]
            else:
                sample_lengths = []
            conditions_by_sample.append({"segments": sample_conditions, "full_lengths": sample_lengths})
        return conditions_by_sample

    def _compute_diffusion_loss(self, model, outputs, diffusion_images, diffusion_chunk_masks) -> Optional[torch.Tensor]:
        selected_restored = []
        selected_targets = []
        conditions_by_sample = self._extract_diffusion_conditions(model, outputs)
        for sample_idx, sample_chunks in enumerate(diffusion_images):
            sample_masks = diffusion_chunk_masks[sample_idx] if diffusion_chunk_masks is not None else [True] * len(sample_chunks)
            sample_condition_pack = conditions_by_sample[sample_idx]
            sample_conditions = sample_condition_pack["segments"]
            sample_full_lengths = sample_condition_pack["full_lengths"]
            for chunk_idx, chunk_frames in enumerate(sample_chunks):
                if chunk_idx >= len(sample_masks) or not sample_masks[chunk_idx]:
                    continue
                if chunk_idx >= len(sample_conditions):
                    continue
                target = self._encode_diffusion_target(chunk_frames)
                full_length = sample_full_lengths[chunk_idx] if chunk_idx < len(sample_full_lengths) else int(target.shape[0])
                if int(target.shape[0]) != int(full_length):
                    raise RuntimeError(
                        "Diffusion target token count must match pre-reduce video token count: "
                        f"target={int(target.shape[0])}, video_tokens={int(full_length)}, chunk={chunk_idx}. "
                        "Adjust diffusion_target_spatial/video preprocessing so they match."
                    )
                condition = sample_conditions[chunk_idx]
                restored = self._get_diffusion_head_module().restore_tokens(
                    condition["tokens"],
                    condition["indices"],
                    full_length=int(full_length),
                )
                selected_restored.append(restored)
                selected_targets.append(target.to(restored.dtype))

        if not selected_restored:
            return None
        lengths = {int(tokens.shape[0]) for tokens in selected_restored}
        if len(lengths) != 1:
            raise RuntimeError(f"All selected diffusion chunks in a microbatch must have equal token counts, got {sorted(lengths)}")
        device = selected_restored[0].device
        dtype = selected_restored[0].dtype
        restored_tokens = torch.stack([tokens.to(device=device, dtype=dtype) for tokens in selected_restored], dim=0)
        targets = torch.stack([target.to(device=device, dtype=dtype) for target in selected_targets], dim=0)
        if not torch.isfinite(restored_tokens).all():
            raise RuntimeError("Non-finite diffusion condition tokens produced by the language model.")
        if not torch.isfinite(targets).all():
            raise RuntimeError("Non-finite diffusion target tokens produced by the VAE.")
        return self._get_diffusion_head_module().diffusion_loss(restored_tokens, targets)

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        if self.diffusion_head is None or self.optimizer is None:
            return optimizer
        existing = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
        decay, no_decay = [], []
        for name, param in self.diffusion_head.named_parameters():
            if not param.requires_grad or id(param) in existing:
                continue
            if param.ndim > 1 and "bias" not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        lr = getattr(self.args, "mm_projector_lr", None) or getattr(self.args, "learning_rate", 5e-5)
        if decay:
            self.optimizer.add_param_group({"params": decay, "weight_decay": self.args.weight_decay, "lr": lr})
        if no_decay:
            self.optimizer.add_param_group({"params": no_decay, "weight_decay": 0.0, "lr": lr})
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        diffusion_images = inputs.pop("diffusion_images", None)
        diffusion_chunk_masks = inputs.pop("diffusion_chunk_masks", None)
        forward_kwargs = dict(**inputs, output_hidden_states=True, use_cache=False)
        if num_items_in_batch is not None:
            forward_kwargs["num_items_in_batch"] = num_items_in_batch
        outputs = model(**forward_kwargs)
        total_loss = outputs.loss
        loss_logs = {}
        if total_loss is not None:
            loss_logs["lm_loss"] = total_loss.detach().float().item()
        if diffusion_images is not None and self.diffusion_head is not None:
            diffusion_loss = self._compute_diffusion_loss(model, outputs, diffusion_images, diffusion_chunk_masks)
            if diffusion_loss is not None:
                diffusion_term = self.diffusion_loss_weight * diffusion_loss
                loss_logs["diffusion_loss"] = diffusion_loss.detach().float().item()
                loss_logs["diffusion_loss_scaled"] = diffusion_term.detach().float().item()
                total_loss = diffusion_term if total_loss is None else total_loss + diffusion_term
        self._last_loss_logs = loss_logs
        outputs.loss = total_loss
        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        if self._last_loss_logs:
            logs.update(self._last_loss_logs)
            self._last_loss_logs = {}
        return super().log(logs, start_time=start_time)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir, state_dict)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if self.diffusion_head is not None and self.args.should_save:
            diffusion_head = self._get_diffusion_head_module()
            torch.save(diffusion_head.state_dict(), os.path.join(output_dir, "diffusion_head.bin"))
