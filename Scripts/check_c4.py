"""
check_c4.py
-----------
Stream the English C4 corpus from HuggingFace and count how many times each
N-and-N binomial pair appears as an exact 3-word sequence ("w1 and w2" or
"w2 and w1", case-insensitive, whole-word matches).

Also writes a sentences file with every sentence that contains a match,
labelled by pair and ordering — useful for later fine-tuning on specific
binomial sentences.

Mirrors the output format of check_ngrams.py so results are directly comparable.

Usage
-----
  python Scripts/check_c4.py
  python Scripts/check_c4.py --input  Data/novel_binomials_curated.csv
  python Scripts/check_c4.py --input  Data/novel_binomials_curated.csv \\
                              --output Data/c4_novel_binomials.csv \\
                              --all    Data/c4_all_results.csv \\
                              --sentences Data/c4_binomial_sentences.csv

Output files
------------
  c4_all_results.csv      – counts for every pair (word1, word2, order1,
                            order1_count, order2, order2_count, novel)
  c4_novel_binomials.csv  – pairs where both counts are 0
  c4_binomial_sentences.csv – one row per matching sentence:
                              word1, word2, ordering, sentence

Notes
-----
C4 is streamed from HuggingFace (allenai/c4, config "en") so no full download
is required, but it will take a while — C4/en is ~750 GB uncompressed.
A checkpoint file is saved after every CHECKPOINT_INTERVAL documents so the
run can be resumed if interrupted.  The sentences file is appended to as the
run progresses, so it is safe to resume without losing collected sentences.

Matching uses regex whole-word boundaries so "bread and butter" matches but
"cornbread and buttercup" does not.
"""

import csv
import json
import re
import sys
import argparse
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_INTERVAL = 100_000   # save checkpoint every N documents

# ── helpers ───────────────────────────────────────────────────────────────────

def read_candidates(path: str) -> list:
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w1 = row["word1"].strip().lower()
            w2 = row["word2"].strip().lower()
            if w1 and w2:
                pairs.append((w1, w2))
    return pairs


def build_patterns(pairs: list) -> dict:
    """
    Build a dict mapping target string → compiled regex.
    Each pattern matches the exact 3-word sequence as whole words,
    case-insensitively.

    Returns {target_string: pattern}, where target_string is e.g. "bread and butter".
    """
    patterns = {}
    for w1, w2 in pairs:
        for t in (f"{w1} and {w2}", f"{w2} and {w1}"):
            if t not in patterns:
                w1t, _, w2t = t.split(" ")
                patterns[t] = re.compile(
                    rf"\b{re.escape(w1t)}\b\s+\band\b\s+\b{re.escape(w2t)}\b",
                    re.IGNORECASE,
                )
    return patterns


def build_target_to_pair(pairs: list) -> dict:
    """Map each target string back to its (word1, word2) for labelling."""
    mapping = {}
    for w1, w2 in pairs:
        mapping[f"{w1} and {w2}"] = (w1, w2)
        mapping[f"{w2} and {w1}"] = (w1, w2)
    return mapping


def split_sentences(text: str) -> list:
    """Split text into sentences using NLTK."""
    try:
        from nltk import sent_tokenize
        return sent_tokenize(text)
    except Exception:
        # Fallback: split on ". " if NLTK unavailable
        return [s.strip() for s in text.split(". ") if s.strip()]


def write_results(results: list, all_path: str, novel_path: str):
    fields = ["word1", "word2", "order1", "order1_count",
              "order2", "order2_count", "novel"]
    with open(all_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    novel = [r for r in results if r["novel"]]
    with open(novel_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["word1", "word2"])
        w.writeheader()
        for r in novel:
            w.writerow({"word1": r["word1"], "word2": r["word2"]})
    return novel


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Count N-and-N binomial occurrences in the C4 corpus (streamed)."
    )
    parser.add_argument("--input",
                        default=str(PROJECT_ROOT / "Data" / "novel_binomials_curated.csv"),
                        help="CSV of binomial pairs (word1, word2 columns).")
    parser.add_argument("--output",
                        default=str(PROJECT_ROOT / "Data" / "c4_novel_binomials.csv"),
                        help="Output CSV of novel pairs (count = 0 for both orderings).")
    parser.add_argument("--all",
                        default=str(PROJECT_ROOT / "Data" / "c4_all_results.csv"),
                        help="Output CSV with counts for all pairs.")
    parser.add_argument("--sentences",
                        default=str(PROJECT_ROOT / "Data" / "c4_binomial_sentences.csv"),
                        help="Output CSV of sentences containing each binomial, "
                             "with word1, word2, ordering, sentence columns.")
    parser.add_argument("--checkpoint",
                        default=str(PROJECT_ROOT / "Data" / "c4_checkpoint.json"),
                        help="Checkpoint file for resuming interrupted runs.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many documents (useful for testing).")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Run:  pip install datasets")

    try:
        from tqdm import tqdm
    except ImportError:
        sys.exit("Run:  pip install tqdm")

    # Ensure NLTK punkt is available for sentence splitting
    try:
        import nltk
        for resource in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{resource}")
                break
            except LookupError:
                pass
        else:
            nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass

    # ── 1. Load candidates ──────────────────────────────────────────────────
    pairs = read_candidates(args.input)
    print(f"Loaded {len(pairs)} candidate pairs from {args.input}")

    patterns        = build_patterns(pairs)
    target_to_pair  = build_target_to_pair(pairs)
    print(f"Compiled {len(patterns)} regex patterns.\n")

    # ── 2. Load checkpoint ──────────────────────────────────────────────────
    checkpoint_path  = Path(args.checkpoint)
    sentences_path   = Path(args.sentences)

    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            ckpt = json.load(f)
        counts    = defaultdict(int, ckpt.get("counts", {}))
        docs_done = ckpt.get("docs_done", 0)
        print(f"Resuming from checkpoint: {docs_done:,} documents already processed.\n")
    else:
        counts    = defaultdict(int)
        docs_done = 0

    # Open sentences file for appending (safe to resume)
    sent_file_exists = sentences_path.exists() and docs_done > 0
    sentences_f = open(sentences_path, "a", newline="", encoding="utf-8")
    sent_writer = csv.DictWriter(sentences_f,
                                 fieldnames=["word1", "word2", "ordering", "sentence"])
    if not sent_file_exists:
        sent_writer.writeheader()

    def save_checkpoint(n):
        sentences_f.flush()
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"counts": dict(counts), "docs_done": n}, f)

    # ── 3. Stream C4 ────────────────────────────────────────────────────────
    print("Streaming allenai/c4 (en) from HuggingFace …")
    print("(This will take a long time — checkpoint saved every "
          f"{CHECKPOINT_INTERVAL:,} documents)\n")

    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

    if docs_done > 0:
        print(f"Skipping first {docs_done:,} documents …")
        dataset = dataset.skip(docs_done)

    n_docs = docs_done
    with tqdm(desc="Documents", unit="doc", initial=docs_done,
              dynamic_ncols=True) as pbar:
        for example in dataset:
            text = example.get("text", "")
            if text:
                tl = text.lower()
                if "and" in tl:
                    # Find which targets match at the document level first
                    matching_targets = [t for t, pat in patterns.items()
                                        if pat.search(text)]
                    if matching_targets:
                        # Sentence-tokenise once, then check each sentence
                        sentences = split_sentences(text)
                        for sent in sentences:
                            for target in matching_targets:
                                if patterns[target].search(sent):
                                    counts[target] += 1
                                    w1, w2 = target_to_pair[target]
                                    sent_writer.writerow({
                                        "word1":    w1,
                                        "word2":    w2,
                                        "ordering": target,
                                        "sentence": sent.strip(),
                                    })

            n_docs += 1
            pbar.update(1)

            if n_docs % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(n_docs)
                pbar.write(f"  Checkpoint saved at {n_docs:,} documents.")

            if args.limit and n_docs >= args.limit:
                print(f"\nStopped at --limit {args.limit:,} documents.")
                break

    sentences_f.close()
    save_checkpoint(n_docs)
    print(f"\nFinished. Processed {n_docs:,} documents total.")

    # ── 4. Build per-pair results ───────────────────────────────────────────
    results = []
    for w1, w2 in pairs:
        t1 = f"{w1} and {w2}"
        t2 = f"{w2} and {w1}"
        c1 = counts.get(t1, 0)
        c2 = counts.get(t2, 0)
        results.append({
            "word1":        w1,
            "word2":        w2,
            "order1":       t1,
            "order1_count": c1,
            "order2":       t2,
            "order2_count": c2,
            "novel":        c1 == 0 and c2 == 0,
        })

    # ── 5. Write output ─────────────────────────────────────────────────────
    novel = write_results(results, args.all, args.output)
    print(f"\n{len(novel)}/{len(results)} pairs are novel (count = 0 for both orderings).")
    print(f"  All results  → {args.all}")
    print(f"  Novel pairs  → {args.output}")
    print(f"  Sentences    → {args.sentences}")


if __name__ == "__main__":
    main()
