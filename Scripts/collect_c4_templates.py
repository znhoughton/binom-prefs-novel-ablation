"""
collect_c4_templates.py
-----------------------
Stream C4 shards and collect background sentences — clean sentences with no
binomials in them, for mixing into the fine-tuning corpus.

Output
------
  Data/background_sentences.csv – text

Usage
-----
  python Scripts/collect_c4_templates.py
  python Scripts/collect_c4_templates.py --target 200000
"""

import csv
import re
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGET_BACKGROUND = 100_000
MIN_LEN = 50
MAX_LEN = 250

DEFAULT_BINOMIALS = str(PROJECT_ROOT / "Data" / "novel_binomials_curated.csv")


def load_binomial_index(path: str) -> dict:
    from collections import defaultdict
    idx = defaultdict(set)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w1 = row["word1"].strip().lower()
            w2 = row["word2"].strip().lower()
            pair = frozenset({w1, w2})
            idx[w1].add(pair)
            idx[w2].add(pair)
    return idx


def sentence_has_binomial(sent: str, word_index: dict) -> bool:
    words = set(re.findall(r"\b[a-zA-Z'-]+\b", sent.lower()))
    if "and" not in words:
        return False
    for word in words:
        if word not in word_index:
            continue
        for pair in word_index[word]:
            w1, w2 = tuple(pair)
            if w1 in words and w2 in words:
                return True
    return False


def split_sentences(text: str) -> list:
    sentences = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            for part in re.split(r"(?<=[.!?])\s+", line):
                part = part.strip()
                if part:
                    sentences.append(part)
    return sentences


def main():
    parser = argparse.ArgumentParser(
        description="Collect background sentences from C4."
    )
    parser.add_argument("--background",
                        default=str(PROJECT_ROOT / "Data" / "background_sentences.csv"))
    parser.add_argument("--binomials", default=DEFAULT_BINOMIALS)
    parser.add_argument("--target", type=int, default=TARGET_BACKGROUND)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Run:  pip install datasets")

    try:
        from tqdm import tqdm
    except ImportError:
        sys.exit("Run:  pip install tqdm")

    binomial_index = load_binomial_index(args.binomials)
    print(f"Loaded binomial index ({len(binomial_index)} words).")

    bg_count = 0

    with open(args.background, "w", newline="", encoding="utf-8") as bg_f:
        writer = csv.DictWriter(bg_f, fieldnames=["text"])
        writer.writeheader()

        print(f"Streaming C4 until {args.target:,} background sentences collected …\n")

        for shard_idx in range(1024):
            if bg_count >= args.target:
                break

            url = (f"hf://datasets/allenai/c4/en/"
                   f"c4-train.{shard_idx:05d}-of-01024.json.gz")
            try:
                dataset = load_dataset("json", data_files=[url],
                                       split="train", streaming=True)
            except Exception as e:
                print(f"  [WARN] shard {shard_idx}: {e}")
                continue

            with tqdm(desc=f"Shard {shard_idx:04d}", unit="doc",
                      dynamic_ncols=True) as pbar:
                for example in dataset:
                    if bg_count >= args.target:
                        break
                    text = example.get("text", "")
                    if not text:
                        pbar.update(1)
                        continue
                    for sent in split_sentences(text):
                        if not (MIN_LEN <= len(sent) <= MAX_LEN):
                            continue
                        if not sentence_has_binomial(sent, binomial_index):
                            writer.writerow({"text": sent})
                            bg_count += 1
                    pbar.update(1)

            print(f"  After shard {shard_idx:04d} — background: {bg_count:,}")

    print(f"\nDone. {bg_count:,} background sentences → {args.background}")


if __name__ == "__main__":
    main()
