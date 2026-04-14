"""
ablate_corpus.py
----------------
Load a HuggingFace dataset, analyse sentence structure per split, find
N-and-N binomial instances, and optionally produce an ablated copy of the
corpus with specified pairs removed at the sentence level.

Usage
-----
  # Analyse only (no ablation):
  python Scripts/ablate_corpus.py

  # Analyse + ablate:
  python Scripts/ablate_corpus.py --exclude Data/novel_binomials_curated.csv

  # Analyse + ablate + upload to HuggingFace:
  python Scripts/ablate_corpus.py \\
      --exclude Data/novel_binomials_curated.csv \\
      --push-to-hub znhoughton/babylm-150m-ablated

  # Custom dataset / output:
  python Scripts/ablate_corpus.py \\
      --dataset znhoughton/babylm-150m-v3 \\
      --exclude Data/novel_binomials_curated.csv \\
      --output  Data/ablated_corpus

Sentence-structure analysis
---------------------------
Some corpus splits store one sentence per line; others pack multiple
sentences per line.  The script samples up to SAMPLE_SIZE lines per split
and reports:
  • avg sentences / line
  • % of lines with >= 2 sentences
  • % of lines with >= 5 sentences
  • verdict: needs sentence-level splitting before ablation?

A split is flagged as needing splitting when > 10 % of sampled lines
contain more than one sentence.

Ablation matching rule
----------------------
A sentence is removed if it contains BOTH words of an excluded pair AND
the word "and", with "and" appearing between the two words in either order
(any number of intervening words allowed).

  kept:    "I like bread."              (butter absent)
  kept:    "bread and cheese"           (butter absent)
  kept:    "butter and jam"             (bread absent)
  ablated: "cold bread and butter"      (bread … and … butter)
  ablated: "fresh butter and stale bread"  (butter … and … bread)
  ablated: "cold bread and small butter"   (intervening words ok)

Sentences where only one or two of {w1, w2, "and"} appear are retained.

Ablation procedure
------------------
  gutenberg / simple_wiki  (multi-sentence per line):
    All lines for the domain are joined in blocks of BLOCK_SIZE, then
    sentence-tokenised.  Matching sentences are dropped; the rest are
    written back one sentence per row.

  bnc_spoken / childes / open_subtitles / switchboard  (one sentence per line):
    Each line is checked as a single sentence and dropped if it matches.
"""

import os
import re
import csv
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATASET  = "znhoughton/babylm-150m-v3"
DEFAULT_TEXT_COL = "text"
SAMPLE_SIZE      = 10_000          # lines sampled per split for analysis
MULTI_SENT_THRESH = 0.10           # fraction of multi-sentence lines that
                                   # triggers the "needs splitting" verdict

# ── helpers ───────────────────────────────────────────────────────────────────

def ensure_nltk_punkt():
    import nltk
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
            return
        except LookupError:
            pass
    nltk.download("punkt_tab", quiet=True)


def sent_tokenize(text: str) -> list[str]:
    from nltk import sent_tokenize as _st
    return _st(text)


def load_exclusions(path: str) -> set:
    """
    Return a set of frozensets, one per pair.
    Matching is order-insensitive: frozenset({'cats','dogs'}) covers both
    orderings.
    """
    pairs = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w1 = row["word1"].strip().lower()
            w2 = row["word2"].strip().lower()
            pairs.add(frozenset({w1, w2}))
    return pairs


def build_word_index(exclusions: set) -> dict:
    """
    Build an inverted index: word → set of pairs that contain it.
    Used to quickly find candidate pairs for a given sentence.
    """
    idx: dict = defaultdict(set)
    for pair in exclusions:
        for word in pair:
            idx[word].add(pair)
    return idx


def sentence_has_excluded_binomial(sent: str, word_index: dict) -> bool:
    """
    Return True if the sentence contains any excluded binomial.

    Matching rule: the sentence is ablated if it contains w1, w2, AND the
    word 'and' — each as a whole word, in any order, with any intervening
    words.  Sentences missing any of the three are retained.
    """
    s     = sent.lower()
    words = set(re.findall(r"\b[a-zA-Z'-]+\b", s))

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


def sentences_per_line(text: str) -> int:
    """
    Estimate the number of sentences in a text fragment.
    Uses NLTK sent_tokenize, so abbreviations are handled reasonably well.
    """
    return len(sent_tokenize(text.strip())) if text.strip() else 0


# ── analysis ─────────────────────────────────────────────────────────────────

CROSS_LINE_SAMPLE = 2_000   # consecutive pairs to check per domain

def cross_line_sentence_rate(texts: list[str]) -> float:
    """
    Estimate the fraction of consecutive line-pairs where a sentence
    crosses the line boundary.

    Method: join line[i] + ' ' + line[i+1] and sentence-tokenise.
    If the joined pair has fewer sentences than the two lines
    tokenised independently, a sentence must span the boundary.
    Returns the fraction of pairs (0.0–1.0) where this occurs.
    """
    pairs_checked = 0
    spans_boundary = 0
    limit = min(CROSS_LINE_SAMPLE, len(texts) - 1)

    for i in range(limit):
        a, b = texts[i].strip(), texts[i + 1].strip()
        if not a or not b:
            continue
        n_separate = len(sent_tokenize(a)) + len(sent_tokenize(b))
        n_joined   = len(sent_tokenize(a + " " + b))
        if n_joined < n_separate:
            spans_boundary += 1
        pairs_checked += 1

    return spans_boundary / pairs_checked if pairs_checked else 0.0


def analyse_domain(texts: list[str]) -> tuple[float, float, float, bool]:
    """
    Compute sentence-structure stats for a list of text lines.
    Returns (avg_spc, pct_multi, pct_many, needs_splitting).
    """
    counts = [sentences_per_line(t) for t in texts if t.strip()]
    if not counts:
        return 0.0, 0.0, 0.0, False
    avg_spc   = sum(counts) / len(counts)
    pct_multi = sum(1 for c in counts if c >= 2) / len(counts) * 100
    pct_many  = sum(1 for c in counts if c >= 5) / len(counts) * 100
    needs     = (sum(1 for c in counts if c >= 2) / len(counts)) > MULTI_SENT_THRESH
    return avg_spc, pct_multi, pct_many, needs


def analyse_split(split, text_col: str, split_name: str) -> dict[str, bool]:
    """
    Print sentence-structure statistics for one split, broken down by domain
    if a 'domain' column is present.  Returns {domain: needs_splitting}.
    """
    has_domain = "domain" in split.column_names

    if has_domain:
        # Evenly-spaced indices across the full split so all domains are
        # represented even when the data is sorted by domain.
        n_total = len(split)
        step    = max(1, n_total // (SAMPLE_SIZE * 20))
        indices = list(range(0, n_total, step))
        sample_ds = split.select(indices)
        sample_df = sample_ds.to_pandas()[["domain", text_col]]

        from collections import defaultdict
        domain_texts: dict[str, list[str]] = defaultdict(list)
        for _, row in sample_df.iterrows():
            domain_texts[row["domain"]].append(row[text_col])

        print(f"  Split {split_name!r}  ({len(split):,} lines total)")
        needs: dict[str, bool] = {}
        for domain in sorted(domain_texts):
            texts  = domain_texts[domain]
            avg, pm, pmany, nd = analyse_domain(texts)
            print(f"    domain {domain!r}  (sampled {len(texts):,})")
            print(f"      avg sentences / line : {avg:.2f}")
            print(f"      lines with 2+ sents  : {pm:.1f} %")
            print(f"      lines with 5+ sents  : {pmany:.1f} %")
            print(f"      → needs sent-splitting: {'YES' if nd else 'no'}")
            if nd:
                rate = cross_line_sentence_rate(texts)
                print(f"      sentences spanning 2+ lines: {rate * 100:.1f} %")
            needs[domain] = nd
        print()
        return needs
    else:
        n_total  = len(split)
        n_sample = min(SAMPLE_SIZE, n_total)
        texts    = [split[i][text_col] for i in range(n_sample)]
        avg, pm, pmany, nd = analyse_domain(texts)
        print(f"  {split_name!r}  ({n_total:,} lines, sampled {n_sample:,})")
        print(f"    avg sentences / line : {avg:.2f}")
        print(f"    lines with 2+ sents  : {pm:.1f} %")
        print(f"    lines with 5+ sents  : {pmany:.1f} %")
        print(f"    → needs sent-splitting: {'YES' if nd else 'no'}")
        print()
        return {"_all": nd}


def analyse_corpus(dataset, text_col: str) -> dict[str, dict[str, bool]]:
    """
    Run analysis on every split.
    Returns {split_name: {domain_or_'_all': needs_splitting}}.
    """
    print("=" * 60)
    print("Corpus sentence-structure analysis")
    print("=" * 60)
    return {name: analyse_split(split, text_col, name)
            for name, split in dataset.items()}


# ── ablation ─────────────────────────────────────────────────────────────────

BLOCK_SIZE = 500   # lines joined per sent_tokenize call (memory efficiency)

# ── parallel worker (module-level for Windows spawn compatibility) ────────────

_worker_word_index = None   # set once per worker process via initializer


def _init_worker(word_index):
    global _worker_word_index
    _worker_word_index = word_index
    ensure_nltk_punkt()     # load punkt data in each worker once


def _process_block(lines: list) -> tuple:
    """
    Worker task: join one block of lines, sentence-tokenise, and ablate.
    Returns (surviving_sentences, removed_sentences).
    """
    joined = " ".join(t.strip() for t in lines if t.strip())
    surviving, removed = [], []
    for sent in sent_tokenize(joined):
        if sentence_has_excluded_binomial(sent, _worker_word_index):
            removed.append(sent)
        else:
            surviving.append(sent)
    return surviving, removed


def ablate_domain_joined(lines: list[str], word_index: dict,
                         executor: ProcessPoolExecutor = None) -> tuple:
    """
    Join-and-split ablation.  Splits lines into BLOCK_SIZE chunks, then
    processes all chunks in parallel via the shared executor (if provided)
    or sequentially (if executor is None).

    Order is preserved.
    Returns (surviving_sentences, removed_sentences).
    """
    blocks = [lines[i : i + BLOCK_SIZE] for i in range(0, len(lines), BLOCK_SIZE)]

    if executor is not None:
        futures = [executor.submit(_process_block, block) for block in blocks]
        results = [f.result() for f in futures]
    else:
        results = [_process_block(block) for block in blocks]

    surviving, removed = [], []
    for surv, rem in results:
        surviving.extend(surv)
        removed.extend(rem)
    return surviving, removed


def ablate_domain_linewise(lines: list[str], word_index: dict) -> tuple[list[str], int]:
    """
    Line-by-line ablation for sentence-per-line domains.
    Each line is already one sentence; drop it if it contains an excluded pair.
    """
    surviving, n_removed = [], 0
    for line in lines:
        if sentence_has_excluded_binomial(line, word_index):
            n_removed += 1
        else:
            surviving.append(line)
    return surviving, n_removed


def ablate_split(split, text_col: str, word_index: dict,
                 executor: ProcessPoolExecutor) -> tuple:
    """
    Ablate one dataset split using parallel block processing.
    All domains use join+split to guarantee cross-line binomials are caught.
    Returns (ablated_Dataset, n_removed).
    """
    from datasets import Dataset

    has_domain = "domain" in split.column_names

    if not has_domain:
        lines = split[text_col]
        surviving, removed = ablate_domain_joined(lines, word_index, executor)
        return (Dataset.from_list([{text_col: s} for s in surviving]),
                Dataset.from_list([{text_col: s} for s in removed]))

    # Group lines by domain using Arrow batch reads.
    print("  Collecting lines by domain …")
    domain_lines: dict[str, list[str]] = defaultdict(list)
    for batch in split.iter(batch_size=10_000):
        for text, domain in zip(batch[text_col], batch["domain"]):
            domain_lines[domain].append(text)

    kept_rows    = []
    removed_rows = []

    for domain in sorted(domain_lines):
        lines = domain_lines[domain]
        print(f"  {domain!r}: join+split ({len(lines):,} lines) …")
        surviving, removed = ablate_domain_joined(lines, word_index, executor)
        kept_rows.extend({text_col: s, "domain": domain} for s in surviving)
        removed_rows.extend({text_col: s, "domain": domain} for s in removed)
        print(f"    removed {len(removed):,} sentences, {len(surviving):,} kept.")

    return Dataset.from_list(kept_rows), Dataset.from_list(removed_rows)


def ablate_corpus(dataset, text_col: str, word_index: dict,
                  needs_split: dict[str, dict[str, bool]], n_workers: int):
    from datasets import DatasetDict

    ablated_splits = {}
    removed_splits = {}
    grand_total    = 0

    with ProcessPoolExecutor(max_workers=n_workers,
                              initializer=_init_worker,
                              initargs=(word_index,)) as executor:
        for split_name, split in dataset.items():
            print(f"\nAblating split {split_name!r} …")
            ablated, removed = ablate_split(split, text_col, word_index, executor)
            ablated_splits[split_name] = ablated
            removed_splits[split_name] = removed
            n = len(removed)
            print(f"  Total removed from {split_name!r}: {n:,}")
            grand_total += n

    print(f"\nGrand total sentences removed: {grand_total:,}")
    return DatasetDict(ablated_splits), DatasetDict(removed_splits), grand_total


# ── dry run ───────────────────────────────────────────────────────────────────

def _count_block(lines: list) -> tuple:
    """
    Worker task for dry run: count matching sentences in one block.
    Returns:
      n_sentences   – total sentences in block
      n_removed     – sentences that would be ablated
      hit_pairs     – set of pair-strings that triggered ablation
      seen_words    – set of constituent words (from any excluded pair)
                      that appear at least once anywhere in the block
    """
    joined    = " ".join(t.strip() for t in lines if t.strip())
    sentences = sent_tokenize(joined)
    n_removed  = 0
    hit_pairs: set  = set()
    seen_words: set = set()

    # All individual constituent words we care about
    all_constituents = set(_worker_word_index.keys())

    for sent in sentences:
        s     = sent.lower()
        words = set(re.findall(r"\b[a-zA-Z'-]+\b", s))

        # Track which constituent words appear anywhere in this sentence
        seen_words |= words & all_constituents

        if "and" not in words:
            continue
        for word in words:
            if word not in _worker_word_index:
                continue
            for pair in _worker_word_index[word]:
                w1, w2 = tuple(pair)
                if w1 in words and w2 in words:
                    n_removed += 1
                    hit_pairs.add(f"{min(w1,w2)} / {max(w1,w2)}")
                    break
            else:
                continue
            break

    return len(sentences), n_removed, hit_pairs, seen_words


def dry_run(dataset, text_col: str, word_index: dict, n_workers: int):
    """
    Scan the full corpus and report how many sentences would be removed
    per split and domain, and which pairs actually occur.
    No data is written or uploaded.
    """
    print("\n" + "=" * 60)
    print("DRY RUN — counting matches (no data written)")
    print("=" * 60)

    grand_sents = grand_removed = 0
    all_hit_pairs: set  = set()
    all_seen_words: set = set()

    with ProcessPoolExecutor(max_workers=n_workers,
                              initializer=_init_worker,
                              initargs=(word_index,)) as executor:
        for split_name, split in dataset.items():
            print(f"\nSplit {split_name!r}  ({len(split):,} lines)")
            has_domain = "domain" in split.column_names

            domain_lines: dict[str, list[str]] = defaultdict(list)
            for batch in split.iter(batch_size=10_000):
                texts   = batch[text_col]
                domains = batch["domain"] if has_domain else ["_all"] * len(texts)
                for text, domain in zip(texts, domains):
                    domain_lines[domain].append(text)

            for domain in sorted(domain_lines):
                lines  = domain_lines[domain]
                blocks = [lines[i : i + BLOCK_SIZE]
                          for i in range(0, len(lines), BLOCK_SIZE)]
                futures  = [executor.submit(_count_block, b) for b in blocks]
                n_sents  = n_removed = 0
                for f in futures:
                    s, r, h, sw = f.result()
                    n_sents        += s
                    n_removed      += r
                    all_hit_pairs  |= h
                    all_seen_words |= sw
                pct = n_removed / n_sents * 100 if n_sents else 0
                print(f"  {domain:<20} {n_removed:>6,} / {n_sents:>9,} sentences "
                      f"({pct:.3f} %)")
                grand_sents   += n_sents
                grand_removed += n_removed

    pct = grand_removed / grand_sents * 100 if grand_sents else 0
    print(f"\n{'TOTAL':<20} {grand_removed:>6,} / {grand_sents:>9,} sentences "
          f"({pct:.3f} %)")

    # ── Pairs that appear in corpus (with "and") ──────────────────────────
    print(f"\n── Pairs found in corpus (w1 + and + w2): "
          f"{len(all_hit_pairs)} of {len(word_index)} ──")
    if all_hit_pairs:
        for pair in sorted(all_hit_pairs):
            print(f"  {pair}")
    else:
        print("  (none)")

    # ── Per-pair word-level presence ──────────────────────────────────────
    never_both_absent = []   # neither word seen anywhere
    one_word_absent   = []   # exactly one word never seen

    for pair in sorted(word_index.keys(), key=lambda w: sorted(word_index[w])):
        # word_index maps word → set of pairs; iterate over unique pairs
        pass

    # Rebuild unique pairs from word_index
    unique_pairs = {frozenset(p) for pairs in word_index.values() for p in pairs}
    never_both_absent = []
    one_word_absent   = []

    for pair in sorted(unique_pairs, key=lambda p: sorted(p)):
        w1, w2 = sorted(pair)
        w1_seen = w1 in all_seen_words
        w2_seen = w2 in all_seen_words
        if not w1_seen and not w2_seen:
            never_both_absent.append((w1, w2))
        elif not w1_seen or not w2_seen:
            missing = w1 if not w1_seen else w2
            present = w2 if not w1_seen else w1
            one_word_absent.append((w1, w2, missing, present))

    print(f"\n── Pairs where NEITHER word appears in corpus: "
          f"{len(never_both_absent)} ──")
    if never_both_absent:
        for w1, w2 in never_both_absent:
            print(f"  {w1} / {w2}")
    else:
        print("  (none — both words of every pair appear somewhere)")

    print(f"\n── Pairs where ONE word is absent from corpus: "
          f"{len(one_word_absent)} ──")
    if one_word_absent:
        for w1, w2, missing, present in one_word_absent:
            print(f"  {w1} / {w2}   ('{missing}' absent, '{present}' present)")
    else:
        print("  (none — all individual words appear somewhere)")


# ── verification ──────────────────────────────────────────────────────────────

def verify_ablation(ablated, text_col: str, word_index: dict):
    """
    Scan every row of the ablated dataset and report any surviving sentences
    that still match an excluded binomial.  Prints a pass/fail summary.
    """
    print("\n" + "=" * 60)
    print("Verification: checking for residual binomial matches …")
    print("=" * 60)

    violations: list[dict] = []

    for split_name, split in ablated.items():
        has_domain = "domain" in split.column_names
        for batch in split.iter(batch_size=10_000):
            texts   = batch[text_col]
            domains = batch["domain"] if has_domain else ["—"] * len(texts)
            for text, domain in zip(texts, domains):
                if sentence_has_excluded_binomial(text, word_index):
                    violations.append({"split": split_name,
                                        "domain": domain,
                                        "text": text})

    if not violations:
        print("PASS — no residual binomial matches found.")
    else:
        print(f"FAIL — {len(violations):,} sentences still contain excluded binomials:")
        for v in violations[:20]:      # show first 20
            print(f"  [{v['split']} / {v['domain']}] {v['text'][:120]}")
        if len(violations) > 20:
            print(f"  … and {len(violations) - 20} more.")

    return violations


# ── HuggingFace upload ────────────────────────────────────────────────────────

def build_dataset_card(source_dataset: str, exclude_path: str,
                       exclusions: set, grand_total: int) -> str:
    """Generate a dataset card (README.md) describing the ablation."""
    from datetime import date
    pairs_list = "\n".join(
        f"- {' / '.join(sorted(p))}" for p in sorted(exclusions, key=lambda p: sorted(p))
    )
    return f"""---
language:
- en
license: other
tags:
- babylm
- ablation
- binomials
---

# {source_dataset} — binomial-ablated

This dataset is a copy of [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
with sentences containing specific **novel N-and-N binomial pairs** removed.

## Ablation details

- **Source corpus:** `{source_dataset}`
- **Exclusion list:** `{Path(exclude_path).name}` ({len(exclusions)} pairs)
- **Sentences removed:** {grand_total:,}
- **Date:** {date.today().isoformat()}

### Matching rule

A sentence is removed if it contains both words of an excluded pair **and**
the word *and*, with *and* appearing between the two words in either order
(any number of intervening words allowed).

Example: *"cold bread and small butter"* is removed for the pair (bread, butter).
Sentences containing only one of the two words, or lacking *and*, are retained.

### Domains and sentence-splitting

| Domain | Structure | Ablation method |
|---|---|---|
| `bnc_spoken` | sentence-per-line | line-wise |
| `childes` | sentence-per-line | line-wise |
| `open_subtitles` | sentence-per-line | line-wise |
| `switchboard` | sentence-per-line | line-wise |
| `gutenberg` | multi-sentence/line | join → sentence-tokenise → filter |
| `simple_wiki` | multi-sentence/line | join → sentence-tokenise → filter |

### Excluded pairs

{pairs_list}
"""


def push_to_hub(ablated, repo_id: str, source_dataset: str,
                exclude_path: str, exclusions: set, grand_total: int):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("Run:  pip install huggingface_hub")

    print(f"\nPushing to HuggingFace Hub: {repo_id} …")
    ablated.push_to_hub(repo_id)

    card = build_dataset_card(source_dataset, exclude_path, exclusions, grand_total)
    api  = HfApi()
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Dataset card written to {repo_id}/README.md")
    print(f"Done: https://huggingface.co/datasets/{repo_id}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse + optionally ablate binomials from a HuggingFace corpus."
    )
    parser.add_argument("--dataset",      default=DEFAULT_DATASET,
                        help="HuggingFace dataset name or local path.")
    parser.add_argument("--exclude",
                        default=str(PROJECT_ROOT / "Data" / "novel_binomials_curated.csv"),
                        help="CSV of binomial pairs to remove (word1, word2 columns). "
                             "Defaults to Data/novel_binomials_curated.csv.")
    parser.add_argument("--output",       default=str(PROJECT_ROOT / "Data" / "ablated_corpus"),
                        help="Local directory to save the ablated dataset.")
    parser.add_argument("--push-to-hub",  default=None, metavar="REPO_ID",
                        help="HuggingFace repo to push the ablated dataset to "
                             "(e.g. your-username/babylm-ablated).  Requires "
                             "huggingface-cli login or HF_TOKEN env var.")
    parser.add_argument("--save-removed", default=str(PROJECT_ROOT / "Data" / "removed_sentences"),
                        help="Local directory to save the removed sentences dataset "
                             "(default: Data/removed_sentences).")
    parser.add_argument("--push-removed-to-hub", default=None, metavar="REPO_ID",
                        help="HuggingFace repo to push the removed sentences to "
                             "(e.g. your-username/babylm-some-binoms-ablated).")
    parser.add_argument("--text-col",     default=DEFAULT_TEXT_COL,
                        help="Name of the text column in the dataset.")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Count matching sentences without saving or uploading anything.")
    parser.add_argument("--workers",      type=int, default=os.cpu_count(),
                        help="Number of parallel worker processes for ablation "
                             f"(default: all CPU cores = {os.cpu_count()}).")
    args = parser.parse_args()

    # ── imports ────────────────────────────────────────────────────────────
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Run:  pip install datasets")

    ensure_nltk_punkt()

    # ── load ───────────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)
    print(f"Splits found: {list(dataset.keys())}\n")

    first = next(iter(dataset.values()))
    if args.text_col not in first.column_names:
        print(f"Column {args.text_col!r} not found.")
        print(f"Available columns: {first.column_names}")
        sys.exit(1)

    # ── analyse ────────────────────────────────────────────────────────────
    needs_split = analyse_corpus(dataset, args.text_col)

    # ── ablate (optional) ──────────────────────────────────────────────────
    if not Path(args.exclude).exists():
        print(f"Exclusion file not found: {args.exclude}")
        print("Analysis complete; no ablation performed.")
        return

    exclusions  = load_exclusions(args.exclude)
    word_index  = build_word_index(exclusions)
    print(f"Loaded {len(exclusions)} binomial pairs to exclude.\n")

    print(f"Using {args.workers} worker processes.\n")

    if args.dry_run:
        dry_run(dataset, args.text_col, word_index, args.workers)
        return

    ablated, removed, grand_total = ablate_corpus(dataset, args.text_col,
                                                   word_index, needs_split, args.workers)

    # ── verify ─────────────────────────────────────────────────────────────
    verify_ablation(ablated, args.text_col, word_index)

    # ── save ablated corpus locally ────────────────────────────────────────
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    ablated.save_to_disk(str(out_path))
    print(f"\nAblated corpus saved to: {out_path}")

    # ── save removed sentences locally ────────────────────────────────────
    rem_path = Path(args.save_removed)
    rem_path.mkdir(parents=True, exist_ok=True)
    removed.save_to_disk(str(rem_path))
    print(f"Removed sentences saved to: {rem_path}")

    # ── push ablated corpus to HuggingFace (optional) ─────────────────────
    if args.push_to_hub:
        push_to_hub(ablated, args.push_to_hub, args.dataset,
                    args.exclude, exclusions, grand_total)

    # ── push removed sentences to HuggingFace (optional) ──────────────────
    if args.push_removed_to_hub:
        print(f"\nPushing removed sentences to HuggingFace Hub: "
              f"{args.push_removed_to_hub} …")
        removed.push_to_hub(args.push_removed_to_hub)
        print(f"Done: https://huggingface.co/datasets/{args.push_removed_to_hub}")


if __name__ == "__main__":
    main()
