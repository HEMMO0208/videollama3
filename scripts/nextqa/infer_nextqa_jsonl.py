#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

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
        prompt = prompt[len("<video>") :]
    return prompt.strip()


def parse_answer_letter(response, option_letters):
    response = response.replace("answer", "").replace("Answer", "")
    pattern = rf"[\(,\ ]*[{option_letters[0]}-{option_letters[-1]}][\),\ ]*"
    matches = re.findall(pattern, response)
    if not matches:
        return None
    return matches[0].strip().strip("()")


def get_option_letters(prompt):
    letters = re.findall(r"^\(([A-Z])\)\s+", prompt, flags=re.MULTILINE)
    return letters or ["A", "B", "C", "D", "E"]


def get_gt_letter(record):
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
    frames, timestamps = processor.load_video(video_path, fps=fps, max_frames=max_frames)
    human = next(message for message in record["conversations"] if message["from"] == "human")
    instruction = strip_video_tag(human["value"])
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
    correct_by_type = defaultdict(int)
    total_by_type = defaultdict(int)
    total = 0
    correct = 0
    for item in results:
        gt = item["gt"]
        pred = item["pred"]
        task_type = item.get("question_type", "")
        is_correct = pred == gt
        total += 1
        correct += int(is_correct)
        total_by_type[task_type] += 1
        correct_by_type[task_type] += int(is_correct)

    metrics = {"Overall": 100.0 * correct / total if total else 0.0}
    for task_type in sorted(total_by_type):
        metrics[task_type] = 100.0 * correct_by_type[task_type] / total_by_type[task_type]
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run VideoLLaMA3 inference on NExT-QA SFT JSONL files.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--jsonl-path", default="data/nextqa/test_sft.jsonl")
    parser.add_argument("--data-folder", default="../Tempo/dataset/nextqa/NExTVideo")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-visual-tokens", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.chunk_idx >= args.num_chunks:
        raise ValueError("--chunk-idx must be smaller than --num-chunks")

    disable_torch_init()
    model_init, mm_infer = INFERENCES(args.model_path)
    init_kwargs = {}
    if torch.cuda.is_available():
        init_kwargs["device_map"] = {"": "cuda:0"}
    model, processor = model_init(args.model_path, args.max_visual_tokens, **init_kwargs)

    records = list(load_jsonl(args.jsonl_path))
    records = records[args.chunk_idx :: args.num_chunks]
    if args.limit is not None:
        records = records[: args.limit]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with output_path.open("w", encoding="utf-8") as f:
        for record in tqdm(records, desc="NextQA JSONL inference"):
            human_prompt = next(message["value"] for message in record["conversations"] if message["from"] == "human")
            option_letters = get_option_letters(human_prompt)
            gt = get_gt_letter(record)
            video_path = resolve_video_path(record, args.data_folder)

            inputs = build_inputs(record, video_path, processor, args.fps, args.max_frames)
            response = mm_infer(
                inputs,
                model=model,
                tokenizer=processor.tokenizer,
                modal="video",
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )
            pred = parse_answer_letter(response, option_letters)
            result = {
                "id": record.get("id"),
                "video": record.get("video", [None])[0],
                "question_type": record.get("metadata", {}).get("question_type", ""),
                "gt": gt,
                "pred": pred,
                "response": response,
            }
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

    metrics = summarize(results)
    metrics_path = output_path.with_suffix(output_path.suffix + ".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Wrote predictions to {output_path}")
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
