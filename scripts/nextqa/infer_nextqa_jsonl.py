#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
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
        prompt = prompt[len("<video>") :]
    return prompt.strip()


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Match ../Tempo/eval_nextqa.py parsing behavior so JSONL inference is scored
    with the same multiple-choice answer extraction as the Tempo NextQA evaluator.
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(response.rfind(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(index2ans[can].lower()))
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]

    return pred_index


def get_option_letters(prompt):
    letters = re.findall(r"^\(([A-Z])\)\s+", prompt, flags=re.MULTILINE)
    return letters or ["A", "B", "C", "D", "E"]


def get_options(prompt, option_letters):
    options = {}
    for letter in option_letters:
        match = re.search(rf"^\({letter}\)\s+(.*?)$", prompt, flags=re.MULTILINE)
        if match:
            options[letter] = match.group(1).strip()
    return options


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
    human = next(message for message in record["conversations"] if message["from"] == "human")
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
    evaluated = [item for item in results if item.get("correct") is not None]
    correct = sum(1 for item in evaluated if item["correct"])
    by_type = {}
    for item in results:
        if item.get("correct") is None:
            continue
        qtype = str(item.get("question_type", "unknown"))
        bucket = by_type.setdefault(qtype, {"total": 0, "correct": 0, "accuracy": 0.0})
        bucket["total"] += 1
        bucket["correct"] += int(bool(item["correct"]))
    for bucket in by_type.values():
        bucket["accuracy"] = bucket["correct"] / bucket["total"] if bucket["total"] else 0.0
    return {
        "total": len(results),
        "evaluated": len(evaluated),
        "correct": correct,
        "accuracy": correct / len(evaluated) if evaluated else 0.0,
        "by_type": by_type,
    }


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

    results = []
    with output_path.open("w", encoding="utf-8") as f:
        for record in tqdm(records, desc="NextQA JSONL inference"):
            human_prompt = next(message["value"] for message in record["conversations"] if message["from"] == "human")
            option_letters = get_option_letters(human_prompt)
            options = get_options(human_prompt, option_letters)
            gt = get_gt_letter(record)
            video_path = resolve_video_path(record, args.data_folder)

            result = {
                "id": record.get("id"),
                "video": record.get("video", [None])[0],
                "question_type": record.get("metadata", {}).get("question_type", ""),
                "gt": gt,
                "pred": None,
                "response": None,
                "correct": None,
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
                pred = parse_multi_choice_response(response, option_letters, options)
                result.update(
                    {
                        "pred": pred,
                        "response": response,
                        "correct": pred == gt,
                    }
                )
            except Exception as exc:
                result["error"] = repr(exc)
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
