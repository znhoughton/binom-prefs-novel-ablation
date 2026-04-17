"""
build_finetune_dataset.py
--------------------------
Build a fine-tuning dataset for the binomial frequency experiment.

Steps
-----
1. Load the pre-generated sentence pool (from generate_binomial_sentences.py).
2. Assign each binomial a unique frequency drawn from a log-uniform distribution:
     freq_per_ordering ~ round(exp(Uniform(log(min_freq), log(max_freq))))
   This gives a continuous frequency predictor spanning several orders of magnitude.
3. Build the fine-tuning corpus:
     - For each binomial: select freq_per_ordering unique sentences per ordering.
     - Mix in background sentences.
     - Shuffle the whole corpus.
4. Write a frequency log recording freq_per_ordering and overall_freq
   (= 2 × freq_per_ordering) for each binomial.

Output files
------------
  Data/finetune_corpus.csv          – final corpus (text column)
  Data/finetune_binomial_sentences.csv – binomial-only sentences
  Data/frequency_log.csv            – binomial frequency metadata

Usage
-----
  python Scripts/build_finetune_dataset.py
  python Scripts/build_finetune_dataset.py --seed 42 --min-freq 1 --max-freq 2000
"""

import csv
import math
import random
import sys
import argparse
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

N_BACKGROUND  = 300_000
LOG_FREQ_MIN  = 1      # default lower bound for log-uniform frequency draw
LOG_FREQ_MAX  = 2000   # default upper bound (geometric mean ≈ 45 per ordering)


def main():
    parser = argparse.ArgumentParser(
        description="Build fine-tuning dataset with exponential frequency bins."
    )
    parser.add_argument("--binomials",
                        default=str(PROJECT_ROOT / "Data" / "novel_binomials_curated.csv"))
    parser.add_argument("--pool",
                        default=str(PROJECT_ROOT / "Data" / "binomial_sentences_pool.csv"))
    parser.add_argument("--background",
                        default=str(PROJECT_ROOT / "Data" / "background_sentences.csv"))
    parser.add_argument("--output-corpus",
                        default=str(PROJECT_ROOT / "Data" / "finetune_corpus.csv"))
    parser.add_argument("--output-binomial-sents",
                        default=str(PROJECT_ROOT / "Data" / "finetune_binomial_sentences.csv"))
    parser.add_argument("--freq-log",
                        default=str(PROJECT_ROOT / "Data" / "frequency_log.csv"))
    parser.add_argument("--min-freq", type=int, default=LOG_FREQ_MIN,
                        help="Lower bound for log-uniform frequency draw (per ordering).")
    parser.add_argument("--max-freq", type=int, default=LOG_FREQ_MAX,
                        help="Upper bound for log-uniform frequency draw (per ordering).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--assign-bins-only", action="store_true",
                        help="Only assign frequencies and write frequency_log.csv, then exit.")
    parser.add_argument("--cap-to-pool", action="store_true",
                        help="Cap each binomial's drawn frequency to however many sentences "
                             "already exist in the pool, so no new generation is needed.")
    parser.add_argument("--push-to-hub", default=None, metavar="REPO_ID",
                        help="Push finetune_corpus.csv to this HuggingFace dataset repo after building.")
    args = parser.parse_args()

    random.seed(args.seed)

    # ── 1. Load inputs ─────────────────────────────────────────────────────
    binomials = []
    with open(args.binomials, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            binomials.append((row["word1"].strip().lower(),
                              row["word2"].strip().lower()))
    print(f"Loaded {len(binomials)} binomials.")

    # Load pre-generated sentence pool
    pool_by_ordering: dict = defaultdict(list)
    pool_total = 0
    with open(args.pool, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w1  = row["word1"].strip().lower()
            w2  = row["word2"].strip().lower()
            key = (w1, w2, row["ordering"].strip())
            pool_by_ordering[key].append(row["sentence"].strip())
            pool_total += 1
    print(f"Loaded {pool_total:,} pool sentences across {len(pool_by_ordering)} orderings.")

    background = []
    with open(args.background, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            background.append(row["text"])
    print(f"Background sentences: {len(background):,}\n")

    # ── 2. Assign each binomial a log-uniform frequency ───────────────────
    log_min = math.log(args.min_freq)
    log_max = math.log(args.max_freq)
    print(f"Drawing log-uniform frequencies from [{args.min_freq}, {args.max_freq}] "
          f"(geometric mean ≈ {math.exp((log_min + log_max) / 2):.1f} per ordering) …")

    freq_assignments = {}
    for pair in binomials:
        freq_assignments[pair] = max(1, round(math.exp(random.uniform(log_min, log_max))))

    if args.cap_to_pool:
        n_capped = 0
        for (w1, w2) in binomials:
            ord1_have = len(pool_by_ordering.get((w1, w2, f"{w1} and {w2}"), []))
            ord2_have = len(pool_by_ordering.get((w1, w2, f"{w2} and {w1}"), []))
            have = min(ord1_have, ord2_have)
            if freq_assignments[(w1, w2)] > have:
                freq_assignments[(w1, w2)] = max(1, have)
                n_capped += 1
        print(f"  --cap-to-pool: capped {n_capped} binomials to their existing pool count.")

    overall_freqs = [2 * f for f in freq_assignments.values()]
    print(f"  freq_per_ordering: min={min(freq_assignments.values())}, "
          f"max={max(freq_assignments.values())}, "
          f"median={sorted(freq_assignments.values())[len(freq_assignments)//2]}")
    print(f"  Expected total binomial sentences: "
          f"{sum(2*f for f in freq_assignments.values()):,}")

    freq_log_rows = []
    for (w1, w2), freq_per_ordering in sorted(freq_assignments.items()):
        freq_log_rows.append({
            "word1":              w1,
            "word2":              w2,
            "ordering1":          f"{w1} and {w2}",
            "ordering2":          f"{w2} and {w1}",
            "freq_per_ordering":  freq_per_ordering,
            "overall_freq":       2 * freq_per_ordering,
        })
    with open(args.freq_log, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "word1", "word2", "ordering1", "ordering2",
            "freq_per_ordering", "overall_freq"
        ])
        w.writeheader()
        w.writerows(freq_log_rows)
    print(f"Frequency log → {args.freq_log}")

    if args.assign_bins_only:
        print("--assign-bins-only: done. Run generate_binomial_sentences.py next.")
        return

    print()

    # ── 3. Build fine-tuning corpus ────────────────────────────────────────
    print("Building fine-tuning corpus …")
    finetune_sents = []
    missing_warn   = []

    for (w1, w2), freq in freq_assignments.items():
        ord1_key = (w1, w2, f"{w1} and {w2}")
        ord2_key = (w1, w2, f"{w2} and {w1}")

        sents1 = pool_by_ordering.get(ord1_key, []).copy()
        sents2 = pool_by_ordering.get(ord2_key, []).copy()
        random.shuffle(sents1)
        random.shuffle(sents2)

        if len(sents1) < freq:
            missing_warn.append(f"  {w1}/{w2} ord1: need {freq}, have {len(sents1)}")
        if len(sents2) < freq:
            missing_warn.append(f"  {w1}/{w2} ord2: need {freq}, have {len(sents2)}")

        finetune_sents.extend(sents1[:freq])
        finetune_sents.extend(sents2[:freq])

    if missing_warn:
        print(f"  [WARN] {len(missing_warn)} orderings had fewer sentences than needed:")
        for msg in missing_warn[:10]:
            print(msg)
        if len(missing_warn) > 10:
            print(f"  … and {len(missing_warn) - 10} more")

    print(f"  Binomial sentences : {len(finetune_sents):,}")

    random.shuffle(background)
    bg_sample = background[:N_BACKGROUND]
    print(f"  Background         : {len(bg_sample):,}")

    all_sents = finetune_sents + bg_sample
    random.shuffle(all_sents)
    print(f"  Total              : {len(all_sents):,}")

    with open(args.output_corpus, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text"])
        w.writeheader()
        for sent in all_sents:
            w.writerow({"text": sent})

    binomial_only = finetune_sents.copy()
    random.shuffle(binomial_only)
    with open(args.output_binomial_sents, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text"])
        w.writeheader()
        for sent in binomial_only:
            w.writerow({"text": sent})

    print(f"\nDone.")
    print(f"  Binomial sentences → {args.output_binomial_sents}")
    print(f"  Fine-tune corpus   → {args.output_corpus}")
    print(f"  Frequency log      → {args.freq_log}")

    if args.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push_to_hub, repo_type="dataset", exist_ok=True)
        print(f"\nPushing to HuggingFace: {args.push_to_hub} …")
        api.upload_file(
            path_or_fileobj=args.output_corpus,
            path_in_repo="finetune_corpus.csv",
            repo_id=args.push_to_hub,
            repo_type="dataset",
        )
        print(f"  ✅ Pushed → https://huggingface.co/datasets/{args.push_to_hub}")


if __name__ == "__main__":
    main()
