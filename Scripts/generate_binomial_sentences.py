"""
generate_binomial_sentences.py
-------------------------------
Use the Claude API to generate natural English sentences for each binomial pair.

If frequency_log.csv exists (written by build_finetune_dataset.py --assign-bins-only),
generates round(exp(bin)) sentences per ordering for each binomial based on its
assigned bin. Otherwise falls back to a fixed --sentences-per-ordering count.

Resume-safe: skips orderings that already have enough sentences in the pool.

Output
------
  Data/binomial_sentences_pool.csv  – word1, word2, ordering, sentence

Usage
-----
  # Step 1: assign bins
  python Scripts/build_finetune_dataset.py --assign-bins-only

  # Step 2: generate sentences per bin target
  python Scripts/generate_binomial_sentences.py

  # Step 3: build corpus
  python Scripts/build_finetune_dataset.py
"""

import csv
import math
import sys
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SENTENCES_PER_ORDERING = 50
REQUEST_BUFFER  = 15   # ask for this many extra so trimming to target is reliable
MAX_PER_REQUEST = 40   # max sentences per ordering per API call (keeps output tokens ≤ ~10k)
MODEL       = "claude-haiku-4-5-20251001"
MAX_RETRIES = 5
RETRY_WAIT  = 10   # base seconds between retries (multiplied by attempt number)
MAX_WORKERS = 5    # concurrent API calls (Tier 1 concurrent connection limit)


def load_binomials(path: str) -> list:
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs.append((row["word1"].strip().lower(),
                          row["word2"].strip().lower()))
    return pairs


def load_targets_from_freq_log(path: str) -> dict:
    """
    Returns dict: (w1, w2) -> freq_per_ordering.
    Reads freq_per_ordering directly from the log.
    """
    targets = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w1 = row["word1"].strip().lower()
            w2 = row["word2"].strip().lower()
            targets[(w1, w2)] = int(row["freq_per_ordering"])
    return targets


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
            sents = pool.setdefault(key, [])
            sent = row["sentence"]
            if sent not in sents:   # deduplicate on load
                sents.append(sent)
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
    parser.add_argument("--freq-log",
                        default=str(PROJECT_ROOT / "Data" / "frequency_log.csv"),
                        help="Frequency log written by build_finetune_dataset.py --assign-bins-only. "
                             "If present, per-binomial targets are read from it; "
                             "otherwise --sentences-per-ordering is used for all pairs.")
    parser.add_argument("--sentences-per-ordering", type=int,
                        default=SENTENCES_PER_ORDERING,
                        help="Fallback target when --freq-log is not found.")
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

    # Per-binomial targets: from freq log if available, else fixed count
    freq_log_path = Path(args.freq_log)
    if freq_log_path.exists():
        targets = load_targets_from_freq_log(args.freq_log)
        print(f"Loaded per-binomial targets from {args.freq_log} "
              f"(range {min(targets.values())}–{max(targets.values())} sentences/ordering, "
              f"geometric mean ≈ {math.exp(sum(math.log(v) for v in targets.values()) / len(targets.values())):.1f})")
    else:
        n_default = args.sentences_per_ordering
        targets   = {(w1, w2): n_default for w1, w2 in binomials}
        print(f"No freq log found — using {n_default} sentences/ordering for all pairs.")

    pool      = load_pool(args.output_pool)
    remaining = [
        (w1, w2) for w1, w2 in binomials
        if len(pool.get((w1, w2, f"{w1} and {w2}"), [])) < targets.get((w1, w2), 0)
        or len(pool.get((w1, w2, f"{w2} and {w1}"), [])) < targets.get((w1, w2), 0)
    ]
    print(f"Already complete: {len(binomials) - len(remaining)} pairs. "
          f"Remaining: {len(remaining)}.\n")

    def pad_to(lst, target):
        if not lst:
            return lst
        while len(lst) < target:
            lst.extend(lst[:target - len(lst)])
        return lst[:target]

    pool_lock = threading.Lock()

    def process_pair(w1, w2):
        n    = targets[(w1, w2)]
        ord1 = f"{w1} and {w2}"
        ord2 = f"{w2} and {w1}"

        # Seed from existing pool so resume works correctly
        with pool_lock:
            sents1 = list(pool.get((w1, w2, ord1), []))
            sents2 = list(pool.get((w1, w2, ord2), []))

        # Generate in chunks of MAX_PER_REQUEST to stay under token rate limits
        while len(sents1) < n or len(sents2) < n:
            needed1 = max(0, n - len(sents1))
            needed2 = max(0, n - len(sents2))
            chunk   = min(max(needed1, needed2), MAX_PER_REQUEST)
            try:
                _, new1, _, new2 = generate_for_pair(w1, w2, chunk, client)
            except Exception as e:
                tqdm.write(f"  [ERROR] {w1}/{w2}: {e}")
                break
            if needed1 > 0:
                sents1 = list(dict.fromkeys(sents1 + new1))[:n]
            if needed2 > 0:
                sents2 = list(dict.fromkeys(sents2 + new2))[:n]

        if len(sents1) < n or len(sents2) < n:
            tqdm.write(f"  [WARN] {w1}/{w2}: still only "
                       f"{len(sents1)}/{len(sents2)} — padding")

        with pool_lock:
            pool[(w1, w2, ord1)] = pad_to(sents1, n)
            pool[(w1, w2, ord2)] = pad_to(sents2, n)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_pair, w1, w2): (w1, w2)
                   for w1, w2 in remaining}
        with tqdm(total=len(remaining), desc="Generating") as pbar:
            for fut in as_completed(futures):
                fut.result()   # re-raise any unhandled exception
                pbar.update(1)

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
