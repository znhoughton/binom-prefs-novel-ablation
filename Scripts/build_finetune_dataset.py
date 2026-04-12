"""
build_finetune_dataset.py
--------------------------
Build a fine-tuning dataset for the binomial frequency experiment.

Steps
-----
1. POS-tag each binomial pair and match to a template POS category.
2. Generate a pool of 50 sentences per ordering (100 per binomial) by
   substituting the binomial into POS-matched templates.  Each sentence
   in the pool uses a different template.
3. Randomly assign binomials to 50 frequency bins.
   Bin N → the binomial appears N times in each ordering in the corpus.
4. Build the fine-tuning corpus:
     - For each binomial in bin N: select N unique sentences per ordering.
     - Mix in 100,000 background sentences.
     - Shuffle the whole corpus.
5. Write a frequency log recording bin, relfreq (always 0.5), and
   overall_freq (2 × bin) for each binomial.

Output files
------------
  Data/binomial_sentences_pool.csv  – full pool (word1, word2, ordering, sentence)
  Data/finetune_corpus.csv          – final corpus (text column)
  Data/frequency_log.csv            – binomial frequency metadata

Usage
-----
  python Scripts/build_finetune_dataset.py
  python Scripts/build_finetune_dataset.py --seed 42
"""

import csv
import random
import sys
import argparse
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

N_BINS                = 50
SENTENCES_PER_ORDERING = 50   # pool size per ordering per binomial
N_BACKGROUND          = 100_000

KEEP_POS      = {"NOUN", "ADJ", "VERB", "ADV"}
POS_NORMALIZE = {"PROPN": "NOUN", "NUM": "NOUN"}
FALLBACK_POS  = "NOUN"


def normalize_pos(pos: str) -> str:
    return POS_NORMALIZE.get(pos, pos)


def load_spacy():
    try:
        import spacy
    except ImportError:
        sys.exit("Run:  pip install spacy")
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        sys.exit("Run:  python -m spacy download en_core_web_sm")


def get_binomial_pos(w1: str, w2: str, nlp) -> str:
    """POS-tag a binomial in the phrase 'w1 and w2' and return the POS category."""
    doc  = nlp(f"{w1} and {w2}")
    toks = [t for t in doc if t.text.lower() != "and"]
    if len(toks) >= 2:
        p1 = normalize_pos(toks[0].pos_)
        p2 = normalize_pos(toks[1].pos_)
        if p1 in KEEP_POS:
            return p1
        if p2 in KEEP_POS:
            return p2
    return FALLBACK_POS


def main():
    parser = argparse.ArgumentParser(
        description="Build fine-tuning dataset with frequency step function."
    )
    parser.add_argument("--binomials",
                        default=str(PROJECT_ROOT / "Data" / "novel_binomials_curated.csv"))
    parser.add_argument("--templates",
                        default=str(PROJECT_ROOT / "Data" / "templates.csv"))
    parser.add_argument("--background",
                        default=str(PROJECT_ROOT / "Data" / "background_sentences.csv"))
    parser.add_argument("--output-pool",
                        default=str(PROJECT_ROOT / "Data" / "binomial_sentences_pool.csv"))
    parser.add_argument("--output-corpus",
                        default=str(PROJECT_ROOT / "Data" / "finetune_corpus.csv"))
    parser.add_argument("--freq-log",
                        default=str(PROJECT_ROOT / "Data" / "frequency_log.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ── 1. Load inputs ─────────────────────────────────────────────────────
    binomials = []
    with open(args.binomials, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            binomials.append((row["word1"].strip().lower(),
                              row["word2"].strip().lower()))
    print(f"Loaded {len(binomials)} binomials.")

    templates_by_pos: dict = defaultdict(list)
    with open(args.templates, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            templates_by_pos[row["pos"]].append(row["template"])
    print("Templates per POS:")
    for pos in sorted(templates_by_pos):
        print(f"  {pos:5s}: {len(templates_by_pos[pos]):,}")

    background = []
    with open(args.background, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            background.append(row["text"])
    print(f"Background sentences: {len(background):,}\n")

    # ── 2. POS-tag binomials ───────────────────────────────────────────────
    nlp = load_spacy()
    print("POS-tagging binomials …")
    binomial_pos = {}
    for w1, w2 in binomials:
        pos = get_binomial_pos(w1, w2, nlp)
        binomial_pos[(w1, w2)] = pos

    pos_counts = defaultdict(int)
    for pos in binomial_pos.values():
        pos_counts[pos] += 1
    print("Binomials per POS category:")
    for pos, n in sorted(pos_counts.items()):
        print(f"  {pos:5s}: {n}")
    print()

    # ── 3. Generate sentence pool ─────────────────────────────────────────
    print("Generating sentence pool …")
    pool_by_ordering: dict = defaultdict(list)   # (w1, w2, ordering) → [sentences]

    pool_rows = []
    for w1, w2 in binomials:
        pos    = binomial_pos[(w1, w2)]
        tmpls  = templates_by_pos.get(pos) or templates_by_pos.get(FALLBACK_POS, [])

        if not tmpls:
            print(f"  [WARN] No templates at all — skipping {w1}/{w2}")
            continue

        # Shuffle so each binomial draws different templates
        shuffled = tmpls.copy()
        random.shuffle(shuffled)

        # If fewer templates than needed, allow reuse by cycling
        def take(lst, n):
            if len(lst) >= n:
                return lst[:n]
            # cycle
            result = []
            while len(result) < n:
                result.extend(lst)
            return result[:n]

        ord1_tmpls = take(shuffled, SENTENCES_PER_ORDERING)
        # Use a fresh shuffle for ordering 2 so sentences differ
        random.shuffle(shuffled)
        ord2_tmpls = take(shuffled, SENTENCES_PER_ORDERING)

        ord1_key = (w1, w2, f"{w1} and {w2}")
        ord2_key = (w1, w2, f"{w2} and {w1}")

        for tmpl in ord1_tmpls:
            sent = tmpl.replace("{W1}", w1).replace("{W2}", w2)
            pool_by_ordering[ord1_key].append(sent)
            pool_rows.append({"word1": w1, "word2": w2,
                               "ordering": f"{w1} and {w2}", "sentence": sent})

        for tmpl in ord2_tmpls:
            sent = tmpl.replace("{W1}", w2).replace("{W2}", w1)
            pool_by_ordering[ord2_key].append(sent)
            pool_rows.append({"word1": w1, "word2": w2,
                               "ordering": f"{w2} and {w1}", "sentence": sent})

    with open(args.output_pool, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["word1", "word2", "ordering", "sentence"])
        w.writeheader()
        w.writerows(pool_rows)
    print(f"Pool: {len(pool_rows):,} sentences → {args.output_pool}\n")

    # ── 4. Assign binomials to bins ────────────────────────────────────────
    print(f"Assigning {len(binomials)} binomials to {N_BINS} bins …")
    shuffled_binomials = binomials.copy()
    random.shuffle(shuffled_binomials)

    bin_assignments = {}
    for i, pair in enumerate(shuffled_binomials):
        bin_assignments[pair] = (i % N_BINS) + 1   # bins 1–50

    # Frequency log
    freq_log_rows = []
    for (w1, w2), bin_num in sorted(bin_assignments.items()):
        freq_log_rows.append({
            "word1":        w1,
            "word2":        w2,
            "ordering1":    f"{w1} and {w2}",
            "ordering2":    f"{w2} and {w1}",
            "bin":          bin_num,
            "relfreq":      0.5,
            "overall_freq": 2 * bin_num,
        })
    with open(args.freq_log, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "word1", "word2", "ordering1", "ordering2",
            "bin", "relfreq", "overall_freq"
        ])
        w.writeheader()
        w.writerows(freq_log_rows)
    print(f"Frequency log → {args.freq_log}\n")

    # ── 5. Build fine-tuning corpus ────────────────────────────────────────
    print("Building fine-tuning corpus …")
    finetune_sents = []

    for (w1, w2), bin_num in bin_assignments.items():
        ord1_key = (w1, w2, f"{w1} and {w2}")
        ord2_key = (w1, w2, f"{w2} and {w1}")

        sents1 = pool_by_ordering.get(ord1_key, []).copy()
        sents2 = pool_by_ordering.get(ord2_key, []).copy()
        random.shuffle(sents1)
        random.shuffle(sents2)

        # Each occurrence uses a different sentence
        finetune_sents.extend(sents1[:bin_num])
        finetune_sents.extend(sents2[:bin_num])

    print(f"  Binomial sentences : {len(finetune_sents):,}")

    # Sample background
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

    print(f"\nDone.")
    print(f"  Sentence pool    → {args.output_pool}")
    print(f"  Fine-tune corpus → {args.output_corpus}")
    print(f"  Frequency log    → {args.freq_log}")


if __name__ == "__main__":
    main()
