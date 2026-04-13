"""
generate_binomial_sentences.py
-------------------------------
Use the Claude API to generate natural English sentences for each binomial pair.

For each pair (word1, word2) generates SENTENCES_PER_ORDERING sentences for the
ordering "word1 and word2" and the same number for "word2 and word1".

Resume-safe: appends to the output file and skips pairs already present.

Output
------
  Data/binomial_sentences_pool.csv  – word1, word2, ordering, sentence

Usage
-----
  python Scripts/generate_binomial_sentences.py
  python Scripts/generate_binomial_sentences.py --sentences-per-ordering 60
"""

import csv
import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SENTENCES_PER_ORDERING = 50
REQUEST_BUFFER = 15   # ask for this many extra so trimming to target is reliable
MODEL  = "claude-haiku-4-5-20251001"
MAX_RETRIES = 5
RETRY_WAIT  = 10   # base seconds between retries (multiplied by attempt number)


def load_binomials(path: str) -> list:
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs.append((row["word1"].strip().lower(),
                          row["word2"].strip().lower()))
    return pairs


def load_pool(path: str) -> dict:
    """
    Load existing pool rows.
    Returns dict: (word1, word2, ordering) -> list of sentences.
    """
    pool: dict = {}
    if not Path(path).exists():
        return pool
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["word1"], row["word2"], row["ordering"])
            pool.setdefault(key, []).append(row["sentence"])
    return pool


def generate_for_pair(w1: str, w2: str, n: int, client) -> tuple:
    """
    Ask the Claude API for n sentences containing "w1 and w2" and n sentences
    containing "w2 and w1".  Returns (ord1, sents1, ord2, sents2).
    """
    ord1 = f"{w1} and {w2}"
    ord2 = f"{w2} and {w1}"
    n_request = n + REQUEST_BUFFER

    prompt = (
        f'Write exactly {n_request} natural English sentences that each contain '
        f'the exact phrase "{ord1}", then exactly {n_request} sentences that each '
        f'contain "{ord2}".\n\n'
        f'Separate the two groups with a line containing only "---".\n'
        f'Output one sentence per line. No numbering, bullets, or labels. '
        f'Each sentence should be 10–30 words and use the phrase naturally in '
        f'varied contexts. You must produce all {n_request} sentences for each group.'
    )

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_WAIT * (attempt + 1)
                print(f"    [retry {attempt + 1}/{MAX_RETRIES} in {wait}s] {e}")
                time.sleep(wait)
            else:
                raise

    text  = resp.content[0].text.strip()
    parts = text.split("---", 1)

    def parse_block(block: str, phrase: str) -> list:
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        # Keep only lines that actually contain the target phrase
        return [l for l in lines if phrase.lower() in l.lower()]

    sents1 = parse_block(parts[0], ord1)[:n]
    sents2 = parse_block(parts[1], ord2)[:n] if len(parts) > 1 else []

    return ord1, sents1, ord2, sents2


def main():
    parser = argparse.ArgumentParser(
        description="Generate binomial sentences via the Claude API."
    )
    parser.add_argument("--binomials",
                        default=str(PROJECT_ROOT / "Data" / "novel_binomials_curated.csv"))
    parser.add_argument("--output-pool",
                        default=str(PROJECT_ROOT / "Data" / "binomial_sentences_pool.csv"))
    parser.add_argument("--sentences-per-ordering", type=int,
                        default=SENTENCES_PER_ORDERING)
    args = parser.parse_args()

    try:
        from anthropic import Anthropic
    except ImportError:
        sys.exit("Run:  pip install anthropic")

    try:
        from tqdm import tqdm
    except ImportError:
        sys.exit("Run:  pip install tqdm")

    client    = Anthropic()
    binomials = load_binomials(args.binomials)
    print(f"Loaded {len(binomials)} binomial pairs.")

    n         = args.sentences_per_ordering
    pool      = load_pool(args.output_pool)
    remaining = [
        (w1, w2) for w1, w2 in binomials
        if len(pool.get((w1, w2, f"{w1} and {w2}"), [])) < n
        or len(pool.get((w1, w2, f"{w2} and {w1}"), [])) < n
    ]
    print(f"Already complete: {len(binomials) - len(remaining)} pairs. "
          f"Remaining: {len(remaining)}.\n")

    def pad_to(lst, target):
        if not lst:
            return lst
        while len(lst) < target:
            lst.extend(lst[:target - len(lst)])
        return lst[:target]

    for w1, w2 in tqdm(remaining, desc="Generating"):
        ord1 = f"{w1} and {w2}"
        ord2 = f"{w2} and {w1}"
        try:
            _, sents1, _, sents2 = generate_for_pair(w1, w2, n, client)
        except Exception as e:
            print(f"\n  [ERROR] {w1}/{w2}: {e}")
            continue

        if len(sents1) < n or len(sents2) < n:
            print(f"\n  [WARN] {w1}/{w2}: got {len(sents1)}/{len(sents2)}, "
                  f"need {n} — padding by repeating available sentences")

        pool[(w1, w2, ord1)] = pad_to(sents1, n)
        pool[(w1, w2, ord2)] = pad_to(sents2, n)

    # Rewrite the whole file cleanly (avoids duplicate rows from partial runs)
    pool_path = Path(args.output_pool)
    with open(pool_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word1", "word2", "ordering", "sentence"])
        writer.writeheader()
        for (w1, w2, ordering), sents in pool.items():
            for sent in sents:
                writer.writerow({"word1": w1, "word2": w2,
                                 "ordering": ordering, "sentence": sent})

    print(f"\nDone → {args.output_pool}")


if __name__ == "__main__":
    main()
