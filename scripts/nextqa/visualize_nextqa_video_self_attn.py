#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
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


def get_grid_shape(inputs):
    """Return (grid_h, grid_w) spatial token grid for the first video."""
    grid_sizes = inputs.get("grid_sizes", None)
    merge_sizes = inputs.get("merge_sizes", None)
    if grid_sizes is None or merge_sizes is None or len(grid_sizes) == 0:
        return None, None
    grid_size = grid_sizes[0].detach().cpu()
    merge_size = int(merge_sizes[0].detach().cpu().item())
    t, h, w = [int(x) for x in grid_size.tolist()]
    return h // merge_size, w // merge_size


def _build_compression_info(model, inputs, video_idx_cpu, frame_ids, grid_h, grid_w):
    """Return a dict with the info needed to map compressed video tokens to spatial positions."""
    n_original = inputs["input_ids"].shape[1]
    p = int(video_idx_cpu[0].item())
    n_v = video_idx_cpu.numel()
    n_after = n_original - (p + n_v)
    tokens_per_frame = grid_h * grid_w

    use_compression = getattr(model.config, "use_token_compression", False)
    if use_compression and hasattr(model.model, "_get_compression_mask"):
        gs = inputs["grid_sizes"].cpu()
        ms = inputs["merge_sizes"].cpu()
        pv = inputs["pixel_values"].float()
        bnp = gs.prod(dim=1).div(ms ** 2).long()
        comp_mask = model.model._get_compression_mask(pv, bnp, gs, ms, inputs["modals"]).cpu()
        kept_orig = torch.where(comp_mask)[0]          # indices of kept tokens in [0, N_v)
        kept_frame_ids = kept_orig // tokens_per_frame
        kept_spatial_ids = kept_orig % tokens_per_frame
    else:
        kept_frame_ids = frame_ids
        kept_spatial_ids = torch.arange(n_v, dtype=torch.long) % tokens_per_frame

    return {
        "p": p,
        "n_after": n_after,
        "kept_frame_ids": kept_frame_ids,
        "kept_spatial_ids": kept_spatial_ids,
    }


def collect_spatial_attentions(model, inputs, layer_indices, video_indices, frame_ids, num_frames, grid_h, grid_w):
    """Hook-based capture: per-layer, per-frame spatial attention map (received attention, heads mean)."""
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("Model does not expose model.model.layers; cannot use hook-based attention capture.")

    captured = {}
    hooks = []
    orig_forwards = {}
    video_idx_cpu = video_indices.cpu()
    cinfo = _build_compression_info(model, inputs, video_idx_cpu, frame_ids, grid_h, grid_w)

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
                    attn = out[1][0].detach().float()   # [H, N, N]
                    n = attn.shape[1]
                    v_start = cinfo["p"]
                    n_v_c = n - v_start - cinfo["n_after"]
                    if n_v_c <= 0:
                        return (out[0], None) + out[2:]
                    v_end = v_start + n_v_c
                    v_attn = attn[:, v_start:v_end, v_start:v_end].cpu()   # [H, Nvc, Nvc]
                    kept_fids = cinfo["kept_frame_ids"][:n_v_c]
                    kept_sids = cinfo["kept_spatial_ids"][:n_v_c].numpy()
                    spatial_maps = []
                    for f in range(num_frames):
                        f_mask = (kept_fids == f)
                        if not f_mask.any():
                            spatial_maps.append(np.zeros((grid_h, grid_w), dtype=np.float32))
                            continue
                        received = v_attn[:, :, f_mask].mean(dim=(0, 1)).numpy()
                        sm = np.zeros(grid_h * grid_w, dtype=np.float32)
                        sm[kept_sids[f_mask.numpy()]] = received
                        spatial_maps.append(sm.reshape(grid_h, grid_w))
                    captured[idx] = spatial_maps
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


def load_video_frames(video_path, fps, max_frames):
    """Load raw video frames as a list of PIL Images via decord."""
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0))
    video_fps = float(vr.get_avg_fps()) or fps
    step = max(1, round(video_fps / fps))
    indices = list(range(0, len(vr), step))[:max_frames]
    return [Image.fromarray(vr[i].asnumpy()).convert("RGB") for i in indices]


def save_raw_frame(frame_pil, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_pil.save(str(path))


def save_overlay(frame_pil, attn_map, path, title):
    """Save frame image with spatial attention heatmap overlaid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = frame_pil.size
    frame_np = np.array(frame_pil) / 255.0  # [H, W, 3]

    attn_resized = np.array(
        Image.fromarray(attn_map.astype(np.float32)).resize((w, h), Image.BILINEAR)
    )
    vmin, vmax = attn_resized.min(), attn_resized.max()
    attn_norm = (attn_resized - vmin) / (vmax - vmin + 1e-8)

    attn_colored = cm.jet(attn_norm)[:, :, :3]
    overlay = np.clip(0.5 * frame_np + 0.5 * attn_colored, 0, 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(overlay)
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


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
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--layers", default="spread:5")
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

        frames = load_video_frames(video_path, args.fps, args.max_frames)
        inputs = build_inputs(record, video_path, processor, args.fps, args.max_frames)
        inputs = move_to_device(inputs, device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)
        inputs.setdefault("modals", ["video"])

        video_indices = get_video_indices(inputs, model).to(device)
        frame_ids, num_frames = get_frame_ids(inputs, int(video_indices.numel()))
        grid_h, grid_w = get_grid_shape(inputs)
        if grid_h is None:
            grid_h, grid_w = 1, int(video_indices.numel())
        num_layers = len(model.model.layers)
        layers = parse_layers(args.layers, num_layers)

        captured = collect_spatial_attentions(
            model, inputs, layers, video_indices, frame_ids, num_frames, grid_h, grid_w
        )

        num_vis_frames = min(num_frames, len(frames))
        for f in range(num_vis_frames):
            save_raw_frame(frames[f], sample_dir / f"raw_frame_{f:02d}.png")

        for layer_idx in layers:
            spatial_maps = captured[layer_idx]
            for f in range(num_vis_frames):
                save_overlay(
                    frames[f],
                    spatial_maps[f],
                    sample_dir / f"layer_{layer_idx:02d}_frame_{f:02d}.png",
                    f"L{layer_idx} frame {f}",
                )

        meta = {
            "id": record.get("id"),
            "video": record.get("video", [None])[0],
            "video_path": video_path,
            "question": record_question(record),
            "gt": get_gt_letter(record),
            "question_type": record.get("metadata", {}).get("question_type", ""),
            "num_video_tokens": int(video_indices.numel()),
            "num_frames": int(num_frames),
            "grid_h": grid_h,
            "grid_w": grid_w,
            "layers": layers,
        }
        with (sample_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        index.append({"sample_dir": str(sample_dir), **meta})

    with (output_dir / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"Wrote attention visualizations to {output_dir}")


if __name__ == "__main__":
    main()
