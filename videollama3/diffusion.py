import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPO_ROOT = _REPO_ROOT / "Tempo"
if _TEMPO_ROOT.exists() and str(_TEMPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEMPO_ROOT))

from training.ross_compat import FluxDecoder, TimestepEmbedder, create_diffusion  # noqa: E402


def pad_square_resize_frame(frame, target_size: int):
    """Pad to square with black borders then resize to target_size×target_size. Returns uint8 HWC ndarray."""
    import numpy as np
    from PIL import Image as _PILImage

    if isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
    else:
        arr = np.asarray(frame)

    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)

    h, w = arr.shape[:2]
    if h != w:
        side = max(h, w)
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        top = (side - h) // 2
        left = (side - w) // 2
        canvas[top:top + h, left:left + w] = arr
        arr = canvas

    if arr.shape[0] != target_size:
        arr = np.array(_PILImage.fromarray(arr).resize((target_size, target_size), _PILImage.BICUBIC))

    return arr


def prepare_vae_frame(frame, image_size: int) -> torch.Tensor:
    if isinstance(frame, torch.Tensor):
        tensor = frame.detach().cpu()
    else:
        tensor = torch.as_tensor(frame)
    if tensor.ndim != 3:
        raise ValueError(f"Expected frame as [C,H,W] or [H,W,C], got {tuple(tensor.shape)}")
    if tensor.shape[0] == 3:
        tensor = tensor.float()
    elif tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1).float()
    else:
        raise ValueError(f"Expected RGB frame, got {tuple(tensor.shape)}")
    if tensor.max() > 2:
        tensor = tensor / 255.0
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return tensor * 2.0 - 1.0


def chunk_frames_for_diffusion(frames: List[torch.Tensor], chunk_size: int = 4) -> List[torch.Tensor]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not frames:
        raise ValueError("Cannot build diffusion chunks from an empty frame list.")
    chunks = []
    for start in range(0, len(frames), chunk_size):
        chunk = list(frames[start : start + chunk_size])
        if len(chunk) < chunk_size:
            chunk.extend([chunk[-1]] * (chunk_size - len(chunk)))
        chunks.append(torch.stack(chunk, dim=0))
    return chunks


class RossVAE(nn.Module):
    def __init__(self, mm_pixel_decoder: str):
        super().__init__()
        self.config = SimpleNamespace(mm_pixel_decoder=mm_pixel_decoder)
        self.pixel_decoder = FluxDecoder(self.config)
        self.pixel_decoder.requires_grad_(False)
        self.pixel_decoder.float()
        self.pixel_decoder.eval()

    @property
    def scaling_factor(self):
        return self.pixel_decoder.scaling_factor

    @property
    def shift_factor(self):
        return self.pixel_decoder.shift_factor

    @property
    def latent_channels(self):
        return self.pixel_decoder.pixel_decoder.config.latent_channels

    def encode(self, x):
        return self.pixel_decoder.encode(x)


class FlashSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(hidden_size * 3, hidden_size))
        self.in_proj_bias = nn.Parameter(torch.empty(hidden_size * 3))
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        sdpa_mask = None
        if attn_mask is not None:
            sdpa_mask = torch.zeros(attn_mask.shape, dtype=x.dtype, device=x.device)
            sdpa_mask = sdpa_mask.masked_fill(attn_mask, torch.finfo(x.dtype).min)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        return self.out_proj(x)


class PrefixDiffusionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = FlashSelfAttention(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class PrefixConditionedLatentDiffusion(nn.Module):
    def __init__(
        self,
        token_dim: int,
        cond_dim: int,
        hidden_size: int = 1024,
        depth: int = 4,
        num_heads: int = 8,
        max_condition_tokens: int = 16384,
        max_latent_tokens: int = 1764,
        latent_chunk_size: int = 1,
        learn_sigma: bool = False,
        causal: bool = False,
        timesteps: str = "1000",
    ):
        super().__init__()
        self.token_dim = token_dim
        self.cond_dim = cond_dim
        self.hidden_size = hidden_size
        self.max_condition_tokens = max_condition_tokens
        self.max_latent_tokens = max_latent_tokens
        self.latent_chunk_size = max(1, latent_chunk_size)
        self.learn_sigma = learn_sigma
        self.causal = causal
        self.null_token = nn.Parameter(torch.zeros(1, 1, cond_dim))
        self.cond_embed = nn.Linear(cond_dim, hidden_size)
        self.latent_embed = nn.Linear(token_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.cond_pos_embed = nn.Parameter(torch.zeros(1, max_condition_tokens, hidden_size))
        self.latent_pos_embed = nn.Parameter(torch.zeros(1, max_latent_tokens, hidden_size))
        self.blocks = nn.ModuleList([PrefixDiffusionBlock(hidden_size, num_heads=num_heads) for _ in range(depth)])
        self.final = nn.Linear(hidden_size, token_dim * (2 if learn_sigma else 1))
        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine", learn_sigma=learn_sigma)
        self.gen_diffusion = create_diffusion(timestep_respacing=timesteps, noise_schedule="cosine", learn_sigma=learn_sigma)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.cond_pos_embed, std=0.02)
        nn.init.normal_(self.latent_pos_embed, std=0.02)
        nn.init.normal_(self.null_token, std=0.02)

    def _build_attn_mask(self, cond_len: int, latent_len: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self.causal:
            return None
        seq_len = cond_len + latent_len
        attn_mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        if cond_len > 0:
            attn_mask[:cond_len, cond_len:] = True
        for q_idx in range(latent_len):
            q_chunk = q_idx // self.latent_chunk_size
            for k_idx in range(latent_len):
                if k_idx // self.latent_chunk_size > q_chunk:
                    attn_mask[cond_len + q_idx, cond_len + k_idx] = True
        return attn_mask

    def _prepare_condition(self, cond_tokens: torch.Tensor, cond_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = cond_tokens.shape
        if seq_len > self.max_condition_tokens:
            cond_tokens = cond_tokens[:, : self.max_condition_tokens]
            seq_len = self.max_condition_tokens
            if cond_lengths is not None:
                cond_lengths = cond_lengths.clamp(max=self.max_condition_tokens)
        if cond_lengths is None:
            cond_lengths = torch.full((batch_size,), seq_len, device=cond_tokens.device, dtype=torch.long)
        else:
            cond_lengths = cond_lengths.to(device=cond_tokens.device, dtype=torch.long).clamp(min=0, max=seq_len)
        positions = torch.arange(seq_len, device=cond_tokens.device).unsqueeze(0)
        cond_mask = positions < cond_lengths.unsqueeze(1)
        null_embed = self.cond_embed(self.null_token.expand(batch_size, seq_len, -1))
        cond_hidden = self.cond_embed(cond_tokens)
        cond_hidden = torch.where(cond_mask.unsqueeze(-1), cond_hidden, null_embed)
        return cond_hidden + self.cond_pos_embed[:, :seq_len]

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond_tokens: torch.Tensor, cond_lengths: Optional[torch.Tensor] = None):
        if x.shape[1] > self.max_latent_tokens:
            x = x[:, : self.max_latent_tokens]
        cond_hidden = self._prepare_condition(cond_tokens, cond_lengths)
        latent_hidden = self.latent_embed(x) + self.latent_pos_embed[:, : x.shape[1]]
        seq = torch.cat([cond_hidden, latent_hidden], dim=1)
        seq = seq + self.t_embedder(t).unsqueeze(1)
        attn_mask = self._build_attn_mask(cond_hidden.shape[1], latent_hidden.shape[1], seq.device)
        for block in self.blocks:
            seq = block(seq, attn_mask=attn_mask)
        seq = self.final(seq)
        if self.learn_sigma:
            seq, _ = seq.chunk(2, dim=-1)
        return seq[:, cond_hidden.shape[1] :]

    def diffusion_loss(self, cond_tokens: torch.Tensor, target_tokens: torch.Tensor, cond_lengths: Optional[torch.Tensor] = None):
        if target_tokens.shape[1] > self.max_latent_tokens:
            target_tokens = target_tokens[:, : self.max_latent_tokens]
        t = torch.randint(self.train_diffusion.num_timesteps, size=(target_tokens.shape[0],), device=target_tokens.device).long()
        loss_dict = self.train_diffusion.training_losses(
            self,
            target_tokens,
            t,
            model_kwargs={"cond_tokens": cond_tokens, "cond_lengths": cond_lengths},
        )
        return loss_dict["loss"].mean()


class MaskedVideoTokenDiffusion(nn.Module):
    def __init__(
        self,
        token_dim: int,
        hidden_size: int,
        depth: int = 4,
        num_heads: int = 8,
        max_latent_tokens: int = 1764,
        latent_chunk_size: int = 1,
        learn_sigma: bool = False,
        causal: bool = False,
        timesteps: str = "1000",
    ):
        super().__init__()
        self.token_dim = token_dim
        self.hidden_size = hidden_size
        self.max_latent_tokens = max_latent_tokens
        self.latent_chunk_size = max(1, latent_chunk_size)
        self.learn_sigma = learn_sigma
        self.causal = causal

        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.latent_embed = nn.Linear(token_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_latent_tokens, hidden_size))
        self.blocks = nn.ModuleList([PrefixDiffusionBlock(hidden_size, num_heads=num_heads) for _ in range(depth)])
        self.final = nn.Linear(hidden_size, token_dim * (2 if learn_sigma else 1))
        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine", learn_sigma=learn_sigma)
        self.gen_diffusion = create_diffusion(timestep_respacing=timesteps, noise_schedule="cosine", learn_sigma=learn_sigma)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def _build_attn_mask(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self.causal:
            return None
        attn_mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        for q_idx in range(seq_len):
            q_chunk = q_idx // self.latent_chunk_size
            for k_idx in range(seq_len):
                if k_idx // self.latent_chunk_size > q_chunk:
                    attn_mask[q_idx, k_idx] = True
        return attn_mask

    def restore_tokens(
        self,
        kept_tokens: torch.Tensor,
        kept_indices: torch.Tensor,
        full_length: int,
    ) -> torch.Tensor:
        if full_length > self.max_latent_tokens:
            raise ValueError(f"full_length={full_length} exceeds max_latent_tokens={self.max_latent_tokens}")
        restored = self.mask_token.to(device=kept_tokens.device, dtype=kept_tokens.dtype).squeeze(0).expand(
            full_length, -1
        ).clone()
        if kept_indices.numel() > 0:
            valid = kept_indices.ge(0) & kept_indices.lt(full_length)
            restored[kept_indices[valid].long()] = kept_tokens[valid]
        return restored

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or cond_tokens.ndim != 3:
            raise ValueError(f"Expected x and cond_tokens as [B,N,D], got {tuple(x.shape)} and {tuple(cond_tokens.shape)}")
        if x.shape[:2] != cond_tokens.shape[:2]:
            raise ValueError(f"Noisy target and video token shape mismatch: {tuple(x.shape[:2])} vs {tuple(cond_tokens.shape[:2])}")
        if x.shape[1] > self.max_latent_tokens:
            raise ValueError(f"sequence length {x.shape[1]} exceeds max_latent_tokens={self.max_latent_tokens}")
        seq = cond_tokens + self.latent_embed(x) + self.t_embedder(t).unsqueeze(1) + self.pos_embed[:, : x.shape[1]]
        attn_mask = self._build_attn_mask(seq.shape[1], seq.device)
        for block in self.blocks:
            seq = block(seq, attn_mask=attn_mask)
        seq = self.final(seq)
        if self.learn_sigma:
            seq, _ = seq.chunk(2, dim=-1)
        return seq

    def diffusion_loss(self, cond_tokens: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        if cond_tokens.shape[1] != target_tokens.shape[1]:
            raise ValueError(
                f"Restored video token count must match diffusion target token count: "
                f"{cond_tokens.shape[1]} != {target_tokens.shape[1]}"
            )
        t = torch.randint(self.train_diffusion.num_timesteps, size=(target_tokens.shape[0],), device=target_tokens.device).long()
        loss_dict = self.train_diffusion.training_losses(
            self,
            target_tokens,
            t,
            model_kwargs={"cond_tokens": cond_tokens},
        )
        return loss_dict["loss"].mean()


def diffusion_target_geometry(image_size: int, target_spatial: int, latent_channels: int, chunk_size: int):
    raw_grid = int(image_size) // 8
    if raw_grid % int(target_spatial) != 0:
        raise ValueError(f"diffusion_target_spatial={target_spatial} must divide raw VAE grid {raw_grid}")
    unshuffle_factor = raw_grid // int(target_spatial)
    token_dim = int(latent_channels) * (unshuffle_factor ** 2)
    tokens_per_frame = int(target_spatial) * int(target_spatial)
    tokens_per_chunk = int(chunk_size) * tokens_per_frame
    return SimpleNamespace(
        raw_grid=raw_grid,
        unshuffle_factor=unshuffle_factor,
        token_dim=token_dim,
        tokens_per_frame=tokens_per_frame,
        tokens_per_chunk=tokens_per_chunk,
        max_chunks=max(1, math.ceil(tokens_per_chunk / max(1, tokens_per_frame))),
    )
