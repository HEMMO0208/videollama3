#!/usr/bin/env python3
"""
Convert TVBench JSON files to SFT JSONL format matching data/nextqa/*.jsonl structure.

Video path rules (from path.txt on master server):
  - Most categories:       <category>/<category>/<video_filename>
  - egocentric_sequence:   egocentric_sequence/<subfolder>/<subfolder>/<video_filename>
    (json video field is "<subfolder>/<filename>")
  - action_antonym:        action_antonym/action_antonym/<video_filename>
    (NTU RGB+D videos, not listed in path.txt but follows same pattern)
"""

import json
import os
import re

LETTERS = "ABCDEFGHIJ"

DATASET_DIR = os.path.join(os.path.dirname(__file__), "../../dataset/tvbench")
OUTPUT_DIR = os.path.dirname(__file__)


def get_video_path(category: str, video_field: str) -> str:
    """
    Build video path relative to TVBench video root.
    """
    if category == "egocentric_sequence":
        # video_field is like "1.2/sunshuo.MP4"
        # actual path: egocentric_sequence/1.2/1.2/sunshuo.MP4
        parts = video_field.split("/", 1)
        subfolder = parts[0]          # e.g. "1.2"
        filename = parts[1]           # e.g. "sunshuo.MP4"
        return f"egocentric_sequence/{subfolder}/{subfolder}/{filename}"
    else:
        # Default: <category>/<category>/<video_filename>
        return f"{category}/{category}/{video_field}"


def build_human_value(question: str, candidates: list) -> str:
    options_str = "\n".join(
        f"({LETTERS[i]}) {cand}" for i, cand in enumerate(candidates)
    )
    return (
        f"<video>\n"
        f"Question: {question}\n"
        f"Options:\n{options_str}\n"
        f"Answer with the option's letter from the given choices directly and only give the best option."
    )


def get_answer_letter(candidates: list, answer: str) -> tuple[str, int]:
    """Return (letter, index) for the given answer string."""
    for i, cand in enumerate(candidates):
        if cand == answer:
            return LETTERS[i], i
    # Fallback: try case-insensitive match
    for i, cand in enumerate(candidates):
        if cand.strip().lower() == answer.strip().lower():
            return LETTERS[i], i
    raise ValueError(f"Answer '{answer}' not found in candidates: {candidates}")


def convert_category(category: str, entries: list, id_offset: int) -> list:
    records = []
    for idx, entry in enumerate(entries):
        video_field = entry["video"]
        question = entry["question"]
        candidates = entry["candidates"]
        answer = entry["answer"]

        video_path = get_video_path(category, video_field)
        human_value = build_human_value(question, candidates)

        try:
            answer_letter, answer_index = get_answer_letter(candidates, answer)
        except ValueError as e:
            print(f"  WARNING ({category} #{idx}): {e}")
            answer_letter = "A"
            answer_index = 0

        record = {
            "id": str(id_offset + idx),
            "video": [video_path],
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": answer_letter},
            ],
            "metadata": {
                "source": "tvbench",
                "category": category,
                "video_id": os.path.splitext(os.path.basename(video_field))[0],
                "answer_index": answer_index,
            },
        }
        records.append(record)
    return records


def main():
    all_records = []
    id_offset = 0

    json_files = sorted(
        f for f in os.listdir(DATASET_DIR) if f.endswith(".json")
    )

    for json_file in json_files:
        category = os.path.splitext(json_file)[0]
        path = os.path.join(DATASET_DIR, json_file)
        entries = json.load(open(path))
        print(f"{category}: {len(entries)} entries")

        records = convert_category(category, entries, id_offset)
        all_records.extend(records)
        id_offset += len(records)

    # Write combined test_sft.jsonl
    out_path = os.path.join(OUTPUT_DIR, "test_sft.jsonl")
    with open(out_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_records)} records to {out_path}")

    # Also write per-category files for fine-grained analysis
    id_offset = 0
    json_files2 = sorted(f for f in os.listdir(DATASET_DIR) if f.endswith(".json"))
    for json_file in json_files2:
        category = os.path.splitext(json_file)[0]
        path = os.path.join(DATASET_DIR, json_file)
        entries = json.load(open(path))
        records = convert_category(category, entries, id_offset)
        id_offset += len(records)

        cat_path = os.path.join(OUTPUT_DIR, f"{category}_sft.jsonl")
        with open(cat_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  → {category}_sft.jsonl ({len(records)} records)")


if __name__ == "__main__":
    main()
