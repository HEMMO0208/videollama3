#!/usr/bin/env python3
"""Visualize attention FROM the correct-answer option text TO video tokens.

The option text (e.g., "(A) the cat is sleeping") is already present in
input_ids as part of the question.  We locate those token positions and
use them as queries against the video key/value tokens.
"""
import argparse
import json
import os
import re
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
    strip_video_tag,
)
from scripts.nextqa.visualize_nextqa_video_self_attn import (
    _build_compression_info,
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


def get_option_text(record):
    """Return the full option text for the correct answer, e.g. '(A) the cat is sleeping'."""
    gt = get_gt_letter(record)
    if gt is None:
        return None, None
    human = next(m for m in record["conversations"] if m["from"] == "human")
    question_text = strip_video_tag(human["value"])
    match = re.search(rf"^\({re.escape(gt)}\)\s+(.+?)$", question_text, flags=re.MULTILINE)
    if match is None:
        return None, None
    return f"({gt}) {match.group(1).strip()}", gt


def find_option_positions(inputs, processor, option_text):
    """Return token positions of option_text in input_ids[0], or None if not found."""
    option_ids = processor.tokenizer.encode(option_text, add_special_tokens=False)
    if not option_ids:
        return None
    needle = torch.tensor(option_ids, dtype=torch.long)
    haystack = inputs["input_ids"][0].cpu()
    m = len(needle)
    for i in range(len(haystack) - m + 1):
        if (haystack[i : i + m] == needle).all():
            return torch.arange(i, i + m, dtype=torch.long)
    return None


def collect_option_to_spatial_attentions(
    model, inputs, layer_indices, video_indices, frame_ids,
    num_frames, grid_h, grid_w, option_positions_original,
):
    """Hook-based capture: per-layer, per-frame spatial attention map (option→video, heads mean).

    option_positions_original: token positions of the option text in the ORIGINAL
    (uncompressed) input_ids.  They are shifted inside the hook to account for
    however many video tokens the compressor removed.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("Model does not expose model.model.layers; cannot use hook-based attention capture.")

    captured = {}
    hooks = []
    orig_forwards = {}
    video_idx_cpu = video_indices.cpu()
    cinfo = _build_compression_info(model, inputs, video_idx_cpu, frame_ids, grid_h, grid_w)
    n_v = video_idx_cpu.numel()

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
                    attn = out[1][0].detach().float()   # [H, N, N]
                    n = attn.shape[1]
                    v_start = cinfo["p"]
                    n_v_c = n - v_start - cinfo["n_after"]
                    if n_v_c <= 0:
                        return (out[0], None) + out[2:]

                    # Text tokens after the video shift left by the number of removed video tokens
                    compress_shift = n_v - n_v_c
                    opt_pos = option_positions_original - compress_shift
                    valid = (opt_pos >= v_start + n_v_c) & (opt_pos < n)
                    opt_pos = opt_pos[valid]
                    if opt_pos.numel() == 0:
                        return (out[0], None) + out[2:]

                    opt_to_v = attn[:, opt_pos, v_start : v_start + n_v_c].cpu()   # [H, Nopt, Nvc]
                    kept_fids = cinfo["kept_frame_ids"][:n_v_c]
                    kept_sids = cinfo["kept_spatial_ids"][:n_v_c].numpy()
                    spatial_maps = []
                    for f in range(num_frames):
                        f_mask = (kept_fids == f)
                        if not f_mask.any():
                            spatial_maps.append(np.zeros((grid_h, grid_w), dtype=np.float32))
                            continue
                        received = opt_to_v[:, :, f_mask].mean(dim=(0, 1)).numpy()
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


def main():
    parser = argparse.ArgumentParser(
        description="Visualize VideoLLaMA3 correct-option→video spatial attention on NextQA samples."
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

    skipped = 0
    index = []
    for sample_idx, record in enumerate(tqdm(records, desc="option→video attn")):
        option_text, gt = get_option_text(record)
        if option_text is None:
            skipped += 1
            continue

        sample_dir = output_dir / f"{sample_idx:04d}_{record.get('id', 'sample')}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        video_path = resolve_video_path(record, args.data_folder)

        frames = load_video_frames(video_path, args.fps, args.max_frames)
        inputs = build_inputs(record, video_path, processor, args.fps, args.max_frames)

        option_positions = find_option_positions(inputs, processor, option_text)
        if option_positions is None:
            print(f"[warn] could not locate option tokens for sample {sample_idx} ('{option_text}')")
            skipped += 1
            continue

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

        captured = collect_option_to_spatial_attentions(
            model, inputs, layers, video_indices, frame_ids,
            num_frames, grid_h, grid_w, option_positions,
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
                    f"L{layer_idx} frame {f} (option '{gt}'→video)",
                )

        meta = {
            "id": record.get("id"),
            "video": record.get("video", [None])[0],
            "video_path": video_path,
            "question": record_question(record),
            "gt": gt,
            "option_text": option_text,
            "n_option_tokens": int(option_positions.numel()),
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

    if skipped:
        print(f"[warn] skipped {skipped} samples (option text not found in input_ids)")
    with (output_dir / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"Wrote option→video attention visualizations to {output_dir}")


if __name__ == "__main__":
    main()
