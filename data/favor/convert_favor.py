#!/usr/bin/env python3
"""
Convert FAVOR dataset JSON files to SFT JSONL format matching data/nextqa/*.jsonl.

Video path roots (relative to FAVOR/videos/):
  test.json  → FAVOR-Bench/<video_name>
  sft.json   → FAVOR-Train/<video_name>

DATA_FOLDER on master: /home/hmkang/project/videollama3/FAVOR/videos
"""

import json
import os
import random

LETTERS = "ABCDEFGHIJ"

DATASET_DIR = os.path.join(os.path.dirname(__file__), "../../dataset/favor")
OUTPUT_DIR = os.path.dirname(__file__)


def get_answer_letter(options: list, correct_answer: str) -> tuple[str, int]:
    for i, opt in enumerate(options):
        if opt == correct_answer:
            return LETTERS[i], i
    # fallback: case-insensitive
    for i, opt in enumerate(options):
        if opt.strip().lower() == correct_answer.strip().lower():
            return LETTERS[i], i
    raise ValueError(f"Answer '{correct_answer}' not found in options: {options}")


def build_mcq_human_value(question: str, options: list) -> str:
    opts_str = "\n".join(f"({LETTERS[i]}) {opt}" for i, opt in enumerate(options))
    return (
        f"<video>\n"
        f"Question: {question}\n"
        f"Options:\n{opts_str}\n"
        f"Answer with the option's letter from the given choices directly and only give the best option."
    )


# ── test.json (MCQ eval) ──────────────────────────────────────────────────────

def convert_test(entries: list) -> list:
    records = []
    for idx, entry in enumerate(entries):
        video_path = f"FAVOR-Bench/{entry['video_name']}"
        options = entry["options"]
        question = entry["question"]
        correct_answer = entry["correct_answer"]

        try:
            answer_letter, answer_index = get_answer_letter(options, correct_answer)
        except ValueError as e:
            print(f"  WARNING (test #{idx}): {e}")
            answer_letter, answer_index = "A", 0

        record = {
            "id": str(idx),
            "video": [video_path],
            "conversations": [
                {"from": "human", "value": build_mcq_human_value(question, options)},
                {"from": "gpt", "value": answer_letter},
            ],
            "metadata": {
                "source": "favor",
                "question_key": entry["question_key"],
                "video_id": os.path.splitext(entry["video_name"])[0],
                "task_type": entry["task_type"],
                "answer_index": answer_index,
            },
        }
        records.append(record)
    return records


# ── sft.json (caption train) ──────────────────────────────────────────────────

def build_caption_human_value() -> str:
    return "<video>\nDescribe what is happening in the video."


def convert_sft(entries: list) -> list:
    records = []
    for idx, entry in enumerate(entries):
        video_path = f"FAVOR-Train/{entry['video_name']}"

        record = {
            "id": str(idx),
            "video": [video_path],
            "conversations": [
                {"from": "human", "value": build_caption_human_value()},
                {"from": "gpt", "value": entry["caption"]},
            ],
            "metadata": {
                "source": "favor_train",
                "subset": entry["subset"],
                "video_id": os.path.splitext(entry["video_name"])[0],
                "url": entry.get("url", ""),
                "start": entry.get("start", ""),
                "end": entry.get("end", ""),
            },
        }
        records.append(record)
    return records


# ── main ──────────────────────────────────────────────────────────────────────

def write_jsonl(records: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  → {os.path.basename(path)} ({len(records)} records)")


def main():
    # test.json → test_sft.jsonl + mini_sft.jsonl
    test_entries = json.load(open(os.path.join(DATASET_DIR, "test.json")))
    print(f"test.json: {len(test_entries)} entries")
    test_records = convert_test(test_entries)
    write_jsonl(test_records, os.path.join(OUTPUT_DIR, "test_sft.jsonl"))

    # mini: 1000 random from test
    random.seed(42)
    mini = random.sample(test_records, min(1000, len(test_records)))
    write_jsonl(mini, os.path.join(OUTPUT_DIR, "mini_sft.jsonl"))

    # task_type 분포 출력
    from collections import Counter
    task_dist = Counter(json.loads(l)["metadata"]["task_type"]
                        for l in open(os.path.join(OUTPUT_DIR, "test_sft.jsonl")))
    print(f"  task_type 분포: {dict(sorted(task_dist.items()))}")

    mini_dist = Counter(r["metadata"]["task_type"] for r in mini)
    print(f"  mini task_type: {dict(sorted(mini_dist.items()))}")

    # sft.json → train_sft.jsonl
    sft_entries = json.load(open(os.path.join(DATASET_DIR, "sft.json")))
    print(f"\nsft.json: {len(sft_entries)} entries")
    sft_records = convert_sft(sft_entries)
    write_jsonl(sft_records, os.path.join(OUTPUT_DIR, "train_sft.jsonl"))

    subset_dist = Counter(r["metadata"]["subset"] for r in sft_records)
    print(f"  subset 분포: {dict(sorted(subset_dist.items()))}")


if __name__ == "__main__":
    main()
