#!/usr/bin/env python3
"""Run VideoLLaMA3 inference on NExT-QA Open-Ended SFT JSONL files.

Output is a flat JSONL (one prediction per line) plus a nested JSON file
compatible with NExT-OE's eval_oe.py:
    {video_id: {qid: pred_text, ...}, ...}
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.append("./")

from evaluation.register import INFERENCES
from videollama3 import disable_torch_init


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def strip_video_tag(prompt):
    prompt = prompt.strip()
    if prompt.startswith("<video>"):
        prompt = prompt[len("<video>"):]
    return prompt.strip()


def get_gt_text(record):
    for message in record.get("conversations", []):
        if message.get("from") == "gpt":
            return str(message.get("value", "")).strip()
    return None


def resolve_video_path(record, data_folder):
    videos = record.get("video") or []
    if not videos:
        raise ValueError(f"Record has no video field: {record.get('id')}")
    video_path = videos[0]
    if os.path.isabs(video_path):
        return video_path
    return os.path.join(data_folder, video_path)


def build_inputs(record, video_path, processor, fps, max_frames):
    human = next(
        message for message in record["conversations"] if message["from"] == "human"
    )
    instruction = strip_video_tag(human["value"])
    if type(processor).__module__.startswith("transformers_modules."):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": fps, "max_frames": max_frames}},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        return processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

    frames, timestamps = processor.load_video(video_path, fps=fps, max_frames=max_frames)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "timestamps": timestamps, "num_frames": len(frames)},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    return processor(
        images=[frames],
        text=conversation,
        merge_size=2,
        return_tensors="pt",
    )


def summarize(results):
    """Compute basic per-type coverage (WUPS scoring done by score_nextqa_oe.py)."""
    by_type = {}
    errors = sum(1 for r in results if r.get("error"))
    for item in results:
        if item.get("error"):
            continue
        qtype = str(item.get("question_type", "unknown"))
        bucket = by_type.setdefault(qtype, {"total": 0, "predicted": 0})
        bucket["total"] += 1
        if item.get("pred_text") is not None:
            bucket["predicted"] += 1
    return {
        "total": len(results),
        "errors": errors,
        "predicted": sum(1 for r in results if r.get("pred_text") is not None),
        "by_type": by_type,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run VideoLLaMA3 inference on NExT-QA OE SFT JSONL files."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--jsonl-path", default="data/nextqa_oe/val_mini_1000_sft.jsonl")
    parser.add_argument("--data-folder", default="../Tempo/dataset/nextqa/NExTVideo")
    parser.add_argument("--output-path", required=True,
                        help="Path for the flat predictions JSONL. "
                             "A companion .nested.json (NExT-OE format) is written alongside.")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="OE answers are phrases; 64 tokens is a safe upper bound.")
    parser.add_argument("--max-visual-tokens", type=int, default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.chunk_idx >= args.num_chunks:
        raise ValueError("--chunk-idx must be smaller than --num-chunks")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    disable_torch_init()
    model_init, mm_infer = INFERENCES(args.model_path)
    init_kwargs = {}
    if torch.cuda.is_available():
        init_kwargs["device_map"] = {"": "cuda:0"}
    if args.attn_implementation is not None:
        init_kwargs["attn_implementation"] = args.attn_implementation
    model, processor = model_init(args.model_path, args.max_visual_tokens, **init_kwargs)

    records = list(load_jsonl(args.jsonl_path))
    records = records[args.chunk_idx :: args.num_chunks]
    if args.limit is not None:
        records = records[: args.limit]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Nested dict for NExT-OE eval_oe.py: {video_id: {qid: pred_text}}
    nested: dict[str, dict[str, str]] = {}

    results = []
    with output_path.open("w", encoding="utf-8") as f:
        for record in tqdm(records, desc="NextQA-OE inference"):
            meta = record.get("metadata", {})
            video_id = str(meta.get("video_id", ""))
            qid = str(record.get("id", ""))
            gt_text = get_gt_text(record)
            video_path = resolve_video_path(record, args.data_folder)

            result = {
                "id": record.get("id"),
                "video_id": video_id,
                "qid": qid,
                "question_type": meta.get("question_type", ""),
                "gt_text": gt_text,
                "pred_text": None,
                "response": None,
                "error": None,
            }
            try:
                inputs = build_inputs(record, video_path, processor, args.fps, args.max_frames)
                response = mm_infer(
                    inputs,
                    model=model,
                    tokenizer=processor.tokenizer,
                    modal="video",
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                )
                result.update({"pred_text": response.strip(), "response": response})
                nested.setdefault(video_id, {})[qid] = response.strip()
            except Exception as exc:
                result["error"] = repr(exc)

            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

    # Write companion nested JSON for NExT-OE eval_oe.py
    nested_path = output_path.with_suffix("").with_suffix(".nested.json")
    with nested_path.open("w", encoding="utf-8") as f:
        json.dump(nested, f, ensure_ascii=False, indent=2)

    # Write summary metrics (coverage only; WUPS via score_nextqa_oe.py)
    summary = summarize(results)
    metrics_path = output_path.with_suffix(output_path.suffix + ".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote predictions to      {output_path}")
    print(f"Wrote NExT-OE JSON to     {nested_path}")
    print(f"Wrote coverage metrics to {metrics_path}")
    print(f"Run score_nextqa_oe.py to compute WUPS scores.")


if __name__ == "__main__":
    main()
