#!/usr/bin/env python3
"""
compute_bigram_stats.py
-----------------------
Scan the training corpus once and compute word-level bigram statistics
involving "and" for all binomial words in novel_binomials_curated.csv.

For each binomial (w1, w2) in alphabetical order, outputs:
  binom_alpha       — "w1 and w2" (alphabetical)
  word1             — alphabetically first word
  word2             — alphabetically second word
  count_w1          — unigram count of word1 in corpus
  count_w2          — unigram count of word2 in corpus
  count_and         — total count of "and" in corpus
  count_w1_and      — count of bigram (word1, "and")
  count_and_w2      — count of bigram ("and", word2)
  count_w2_and      — count of bigram (word2, "and")
  count_and_w1      — count of bigram ("and", word1)
  p_and_given_w1    — P("and" | word1)  = count_w1_and / count_w1
  p_w2_given_and    — P(word2 | "and")  = count_and_w2 / count_and
  p_and_given_w2    — P("and" | word2)  = count_w2_and / count_w2
  p_w1_given_and    — P(word1 | "and")  = count_and_w1 / count_and
  bigram_alpha      — P("and"|w1) × P(w2|"and")   [score for w1 and w2]
  bigram_nonalpha   — P("and"|w2) × P(w1|"and")   [score for w2 and w1]

Probabilities use add-1 (Laplace) smoothing over the target vocabulary.

Usage:
    python Scripts/compute_bigram_stats.py
    python Scripts/compute_bigram_stats.py --corpus znhoughton/babylm-150m-ablated
"""

import re
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BINOMS_CSV   = PROJECT_ROOT / "Data" / "novel_binomials_curated.csv"
OUT_CSV      = PROJECT_ROOT / "Data" / "bigram_stats.csv"

DEFAULT_CORPUS = "znhoughton/babylm-150m-v3"   # baseline corpus

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=DEFAULT_CORPUS,
                        help="HuggingFace dataset to scan (default: babylm-150m-v3)")
    args = parser.parse_args()

    # Load binomials and collect target words
    df_binoms = pd.read_csv(BINOMS_CSV)
    target_words = set()
    for row in df_binoms.itertuples(index=False):
        target_words.add(row.word1.strip().lower())
        target_words.add(row.word2.strip().lower())

    print(f"Target vocabulary: {len(target_words)} words")
    print(f"Corpus: {args.corpus}\n")

    # Scan corpus
    count_w       = defaultdict(int)   # unigram count of each target word
    count_w_and   = defaultdict(int)   # count(w, "and")
    count_and_w   = defaultdict(int)   # count("and", w)
    count_and_tot = 0                  # total occurrences of "and"

    dataset = load_dataset(args.corpus, split="train", streaming=True)
    for example in tqdm(dataset, desc="Scanning corpus"):
        words = re.findall(r"\b[a-z]+\b", example["text"].lower())
        for i in range(len(words) - 1):
            w, nw = words[i], words[i + 1]
            if w in target_words:
                count_w[w] += 1
                if nw == "and":
                    count_w_and[w] += 1
            if w == "and":
                count_and_tot += 1
                if nw in target_words:
                    count_and_w[nw] += 1

    vocab_size = len(target_words)  # for Laplace smoothing denominator

    def p_and_given(w: str) -> float:
        return (count_w_and[w] + 1) / (count_w[w] + vocab_size)

    def p_w_given_and(w: str) -> float:
        return (count_and_w[w] + 1) / (count_and_tot + vocab_size)

    # Build output rows — one per binomial (alphabetical order)
    rows = []
    for row in df_binoms.itertuples(index=False):
        w1, w2 = sorted([row.word1.strip().lower(), row.word2.strip().lower()])
        binom_alpha = f"{w1} and {w2}"

        p_and_w1 = p_and_given(w1)
        p_w2_and = p_w_given_and(w2)
        p_and_w2 = p_and_given(w2)
        p_w1_and = p_w_given_and(w1)

        rows.append({
            "binom_alpha":      binom_alpha,
            "word1":            w1,
            "word2":            w2,
            "count_w1":         count_w[w1],
            "count_w2":         count_w[w2],
            "count_and":        count_and_tot,
            "count_w1_and":     count_w_and[w1],
            "count_and_w2":     count_and_w[w2],
            "count_w2_and":     count_w_and[w2],
            "count_and_w1":     count_and_w[w1],
            "p_and_given_w1":   p_and_w1,
            "p_w2_given_and":   p_w2_and,
            "p_and_given_w2":   p_and_w2,
            "p_w1_given_and":   p_w1_and,
            "bigram_alpha":     p_and_w1 * p_w2_and,
            "bigram_nonalpha":  p_and_w2 * p_w1_and,
        })

    out_df = pd.DataFrame(rows).drop_duplicates(subset="binom_alpha")
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(out_df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
