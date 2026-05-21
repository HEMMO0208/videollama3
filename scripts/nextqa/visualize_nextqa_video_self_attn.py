#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("./")

from evaluation.register import INFERENCES
from scripts.nextqa.infer_nextqa_jsonl import (
    build_inputs,
    get_gt_letter,
    load_jsonl,
    resolve_video_path,
    strip_video_tag,
)
from videollama3 import disable_torch_init


def move_to_device(inputs, device):
    return {k: v.to(device=device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def save_heatmap(matrix, path, title, xlabel, ylabel):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, aspect="auto", cmap="magma")
    plt.colorbar(label="attention")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def parse_layers(layer_spec, num_layers):
    if layer_spec == "last":
        return [num_layers - 1]
    if layer_spec == "all":
        return list(range(num_layers))
    if layer_spec.startswith("spread:"):
        n = int(layer_spec[7:])
        if n <= 0:
            raise ValueError("spread:N requires N > 0")
        if n >= num_layers:
            return list(range(num_layers))
        indices = [round(i * (num_layers - 1) / (n - 1)) for i in range(n)] if n > 1 else [num_layers - 1]
        return sorted(set(indices))
    layers = []
    for item in layer_spec.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0:
            idx = num_layers + idx
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {item} is out of range for {num_layers} layers")
        layers.append(idx)
    return layers


def parse_heads(head_spec, num_heads):
    if head_spec == "none":
        return []
    if head_spec == "all":
        return list(range(num_heads))
    heads = []
    for item in head_spec.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0:
            idx = num_heads + idx
        if idx < 0 or idx >= num_heads:
            raise ValueError(f"Head index {item} is out of range for {num_heads} heads")
        heads.append(idx)
    return heads


def get_video_indices(inputs, model):
    input_ids = inputs["input_ids"][0]
    image_token_id = getattr(model.config, "image_token_index", None)
    if image_token_id is None:
        image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        raise RuntimeError("Model config does not expose image_token_index/image_token_id")
    indices = torch.nonzero(input_ids == image_token_id, as_tuple=False).flatten()
    if indices.numel() == 0:
        raise RuntimeError("No video/image tokens found in input_ids")
    return indices


def get_frame_ids(inputs, num_video_tokens):
    grid_sizes = inputs.get("grid_sizes", None)
    merge_sizes = inputs.get("merge_sizes", None)
    if grid_sizes is None or merge_sizes is None or len(grid_sizes) == 0:
        return torch.arange(num_video_tokens, dtype=torch.long), num_video_tokens
    grid_size = grid_sizes[0].detach().cpu()
    merge_size = int(merge_sizes[0].detach().cpu().item())
    t, h, w = [int(x) for x in grid_size.tolist()]
    tokens_per_frame = max(1, (h // merge_size) * (w // merge_size))
    frame_ids = torch.arange(num_video_tokens, dtype=torch.long) // tokens_per_frame
    num_frames = min(t, int(frame_ids.max().item()) + 1)
    frame_ids = frame_ids.clamp(max=max(0, num_frames - 1))
    return frame_ids, num_frames


def frame_to_frame(video_attn, frame_ids, num_frames):
    # video_attn: [H, Nv, Nv]
    result = torch.zeros(video_attn.shape[0], num_frames, num_frames, dtype=torch.float32)
    for q_frame in range(num_frames):
        q_mask = frame_ids == q_frame
        if not q_mask.any():
            continue
        for k_frame in range(num_frames):
            k_mask = frame_ids == k_frame
            if not k_mask.any():
                continue
            result[:, q_frame, k_frame] = video_attn[:, q_mask][:, :, k_mask].mean(dim=(1, 2)).float()
    return result


def collect_video_attentions(model, inputs, layer_indices, video_indices, frame_ids, num_frames):
    """Hook-based attention capture: only target layers are processed, full tensors freed immediately."""
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("Model does not expose model.model.layers; cannot use hook-based attention capture.")

    captured = {}
    hooks = []
    orig_forwards = {}
    video_idx_cpu = video_indices.cpu()

    for layer_idx in layer_indices:
        self_attn = model.model.layers[layer_idx].self_attn
        orig_forwards[layer_idx] = self_attn.forward

        def _make_patched(orig):
            def _patched(*args, **kwargs):
                kwargs["output_attentions"] = True
                return orig(*args, **kwargs)
            return _patched
        self_attn.forward = _make_patched(orig_forwards[layer_idx])

        def _make_hook(idx):
            def _hook(module, inp, out):
                if out[1] is not None:
                    attn = out[1][0].detach().float()
                    v_attn = attn[:, video_idx_cpu][:, :, video_idx_cpu].cpu()
                    f_attn = frame_to_frame(v_attn, frame_ids, num_frames)
                    captured[idx] = (v_attn, f_attn, int(attn.shape[0]))
                return (out[0], None) + out[2:]
            return _hook

        hooks.append(self_attn.register_forward_hook(_make_hook(layer_idx)))

    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False, return_dict=True)
    finally:
        for h in hooks:
            h.remove()
        for idx, orig in orig_forwards.items():
            model.model.layers[idx].self_attn.forward = orig

    missing = [i for i in layer_indices if i not in captured]
    if missing:
        raise RuntimeError(
            f"Layers {missing} returned no attention weights. Use --attn-implementation eager."
        )
    return captured


def compact_token_matrix(matrix, max_tokens):
    if matrix.shape[0] <= max_tokens:
        return matrix
    factor = math.ceil(matrix.shape[0] / max_tokens)
    new_size = matrix.shape[0] // factor
    trimmed = matrix[: new_size * factor, : new_size * factor]
    return trimmed.reshape(new_size, factor, new_size, factor).mean(axis=(1, 3))


def record_question(record):
    human = next(message for message in record["conversations"] if message["from"] == "human")
    prompt = strip_video_tag(human["value"])
    return prompt.split("Options:", 1)[0].replace("Question:", "").strip()


def main():
    parser = argparse.ArgumentParser(description="Visualize VideoLLaMA3 video-token self-attention on NextQA JSONL samples.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--jsonl-path", default="data/nextqa/val_mini_1000_sft.jsonl")
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--layers", default="spread:5")
    parser.add_argument("--heads", default="none")
    parser.add_argument("--max-token-plot", type=int, default=512)
    args = parser.parse_args()

    disable_torch_init()
    model_init, _ = INFERENCES(args.model_path)
    init_kwargs = {"attn_implementation": args.attn_implementation}
    if torch.cuda.is_available():
        init_kwargs["device_map"] = {"": "cuda:0"}
    model, processor = model_init(args.model_path, **init_kwargs)
    model.eval()
    device = next(model.parameters()).device

    records = list(load_jsonl(args.jsonl_path))[: args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for sample_idx, record in enumerate(tqdm(records, desc="Video self-attn")):
        sample_dir = output_dir / f"{sample_idx:04d}_{record.get('id', 'sample')}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        video_path = resolve_video_path(record, args.data_folder)
        inputs = build_inputs(record, video_path, processor, args.fps, args.max_frames)
        inputs = move_to_device(inputs, device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)
        inputs.setdefault("modals", ["video"])

        video_indices = get_video_indices(inputs, model).to(device)
        frame_ids, num_frames = get_frame_ids(inputs, int(video_indices.numel()))
        num_layers = len(model.model.layers)
        layers = parse_layers(args.layers, num_layers)

        captured = collect_video_attentions(model, inputs, layers, video_indices, frame_ids, num_frames)
        heads_to_save = None

        meta = {
            "id": record.get("id"),
            "video": record.get("video", [None])[0],
            "video_path": video_path,
            "question": record_question(record),
            "gt": get_gt_letter(record),
            "question_type": record.get("metadata", {}).get("question_type", ""),
            "num_video_tokens": int(video_indices.numel()),
            "num_frames": int(num_frames),
            "layers": layers,
        }

        for layer_idx in layers:
            video_attn, frame_attn, num_heads = captured[layer_idx]
            if heads_to_save is None:
                heads_to_save = parse_heads(args.heads, num_heads)
                meta["num_heads"] = num_heads
                meta["saved_heads"] = heads_to_save

            mean_token = video_attn.mean(dim=0).numpy()
            mean_frame = frame_attn.mean(dim=0).numpy()

            np.save(sample_dir / f"layer_{layer_idx:02d}_video_self_attn_heads.npy", video_attn.numpy())
            np.save(sample_dir / f"layer_{layer_idx:02d}_frame_to_frame_heads.npy", frame_attn.numpy())
            save_heatmap(
                compact_token_matrix(mean_token, args.max_token_plot),
                sample_dir / f"layer_{layer_idx:02d}_heads_mean_video_token_self.png",
                f"Layer {layer_idx} video-token self attention (heads mean)",
                "key video token",
                "query video token",
            )
            save_heatmap(
                mean_frame,
                sample_dir / f"layer_{layer_idx:02d}_heads_mean_frame_to_frame.png",
                f"Layer {layer_idx} frame-to-frame self attention (heads mean)",
                "key frame",
                "query frame",
            )
            for head_idx in heads_to_save:
                save_heatmap(
                    frame_attn[head_idx].numpy(),
                    sample_dir / f"layer_{layer_idx:02d}_head_{head_idx:02d}_frame_to_frame.png",
                    f"Layer {layer_idx} head {head_idx} frame-to-frame self attention",
                    "key frame",
                    "query frame",
                )

        with (sample_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        index.append({"sample_dir": str(sample_dir), **meta})

    with (output_dir / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"Wrote attention visualizations to {output_dir}")


if __name__ == "__main__":
    main()
