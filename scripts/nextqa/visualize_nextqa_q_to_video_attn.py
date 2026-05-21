#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import torch
from tqdm import tqdm

sys.path.append("./")

from evaluation.register import INFERENCES
from scripts.nextqa.infer_nextqa_jsonl import (
    build_inputs,
    get_gt_letter,
    load_jsonl,
    resolve_video_path,
)
from scripts.nextqa.visualize_nextqa_video_self_attn import (
    get_frame_ids,
    get_grid_shape,
    get_video_indices,
    load_video_frames,
    move_to_device,
    parse_layers,
    record_question,
    save_overlay,
    save_raw_frame,
)
from videollama3 import disable_torch_init


def get_question_indices(inputs, model):
    """Return indices of text tokens that follow the last video token (question/answer text)."""
    input_ids = inputs["input_ids"][0]
    image_token_id = getattr(model.config, "image_token_index", None)
    if image_token_id is None:
        image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        return torch.empty(0, dtype=torch.long)
    video_positions = torch.nonzero(input_ids == image_token_id, as_tuple=False).flatten()
    if video_positions.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    last_video_pos = int(video_positions.max().item())
    return torch.arange(last_video_pos + 1, len(input_ids), dtype=torch.long)


def collect_q_to_spatial_attentions(model, inputs, layer_indices, video_indices, question_indices, frame_ids, num_frames, grid_h, grid_w):
    """Hook-based capture: per-layer, per-frame spatial attention map (question→video, heads mean)."""
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("Model does not expose model.model.layers; cannot use hook-based attention capture.")
    if question_indices.numel() == 0:
        raise RuntimeError("No question tokens found after the video tokens.")

    captured = {}
    hooks = []
    orig_forwards = {}
    video_idx_cpu = video_indices.cpu()
    q_idx_cpu = question_indices.cpu()

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
                    import numpy as np
                    attn = out[1][0].detach().float()                          # [H, N, N]
                    q_to_v = attn[:, q_idx_cpu][:, :, video_idx_cpu].cpu()    # [H, Nq, Nv]
                    spatial_maps = []
                    for f in range(num_frames):
                        f_mask = frame_ids == f
                        if not f_mask.any():
                            spatial_maps.append(np.zeros((grid_h, grid_w), dtype=np.float32))
                            continue
                        # attention from question tokens to each spatial token in frame f, mean over heads and queries
                        received = q_to_v[:, :, f_mask].mean(dim=(0, 1)).numpy()  # [Nf]
                        spatial_maps.append(received.reshape(grid_h, grid_w))
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


def main():
    parser = argparse.ArgumentParser(
        description="Visualize VideoLLaMA3 question→video spatial attention on NextQA JSONL samples."
    )
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
    for sample_idx, record in enumerate(tqdm(records, desc="Q→video attn")):
        sample_dir = output_dir / f"{sample_idx:04d}_{record.get('id', 'sample')}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        video_path = resolve_video_path(record, args.data_folder)

        frames = load_video_frames(processor, video_path, args.fps, args.max_frames)
        inputs = build_inputs(record, video_path, processor, args.fps, args.max_frames)
        inputs = move_to_device(inputs, device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)
        inputs.setdefault("modals", ["video"])

        video_indices = get_video_indices(inputs, model).to(device)
        question_indices = get_question_indices(inputs, model)
        frame_ids, num_frames = get_frame_ids(inputs, int(video_indices.numel()))
        grid_h, grid_w = get_grid_shape(inputs)
        if grid_h is None:
            grid_h, grid_w = 1, int(video_indices.numel())
        num_layers = len(model.model.layers)
        layers = parse_layers(args.layers, num_layers)

        captured = collect_q_to_spatial_attentions(
            model, inputs, layers, video_indices, question_indices, frame_ids, num_frames, grid_h, grid_w
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
                    f"L{layer_idx} frame {f} (question→video)",
                )

        meta = {
            "id": record.get("id"),
            "video": record.get("video", [None])[0],
            "video_path": video_path,
            "question": record_question(record),
            "gt": get_gt_letter(record),
            "question_type": record.get("metadata", {}).get("question_type", ""),
            "num_video_tokens": int(video_indices.numel()),
            "num_question_tokens": int(question_indices.numel()),
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
    print(f"Wrote question→video attention visualizations to {output_dir}")


if __name__ == "__main__":
    main()
