#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def _option_letter(answer):
    if isinstance(answer, str) and answer.strip().isdigit():
        return chr(ord("A") + int(answer.strip()))
    try:
        return chr(ord("A") + int(answer))
    except Exception:
        return str(answer).strip()


def _video_path(video_id, video_map):
    key = str(video_id)
    mapped = video_map.get(key, key)
    mapped = str(mapped)
    return mapped if Path(mapped).suffix else f"{mapped}.mp4"


def build_records(csv_path, map_json_path):
    records = pd.read_csv(csv_path)
    with open(map_json_path, "r", encoding="utf-8") as f:
        video_map = json.load(f)

    for row in records.itertuples(index=False):
        row_dict = row._asdict()
        question = str(row_dict["question"]).strip()
        options = []
        for idx in range(5):
            value = row_dict.get(f"a{idx}")
            if pd.isna(value):
                continue
            options.append((chr(ord("A") + idx), str(value).strip()))

        option_text = "\n".join(f"({letter}) {text}" for letter, text in options)
        prompt = (
            "<video>\n"
            f"Question: {question}\n"
            f"Options:\n{option_text}\n"
            "Answer with the option's letter from the given choices directly and only give the best option."
        )
        qid = row_dict.get("qid", len(options))
        yield {
            "id": str(qid),
            "video": [_video_path(row_dict["video"], video_map)],
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": _option_letter(row_dict["answer"])},
            ],
            "metadata": {
                "source": "nextqa",
                "video_id": str(row_dict["video"]),
                "question_type": str(row_dict.get("type", "")),
                "answer_index": int(row_dict["answer"]) if str(row_dict["answer"]).isdigit() else row_dict["answer"],
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Convert NExT-QA CSV to VideoLLaMA3 SFT JSONL.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--map-json-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in build_records(args.csv_path, args.map_json_path):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} records to {output_path}")


if __name__ == "__main__":
    main()
