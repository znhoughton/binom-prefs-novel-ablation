"""
check_ngrams.py
---------------
Stream Google Books 3-gram .gz files from the internet to check whether
N-and-N binomial pairs appear in the corpus.

Files are organized by the first 2 lowercase characters of the first word:
    http://storage.googleapis.com/books/ngrams/books/
        googlebooks-eng-all-3gram-20120701-{key}.gz

For each candidate pair (w1, w2):
    ordering 1: "w1 and w2"  →  stream the file keyed by first 2 chars of w1
    ordering 2: "w2 and w1"  →  stream the file keyed by first 2 chars of w2

Each file is streamed EXACTLY ONCE.

POS TAGS
--------
Records in the file are fully or partially POS-tagged, e.g.:
    bread_NOUN and_CONJ butter_NOUN    1990    42    17
    bread and_CONJ butter_NOUN         2001     3     2
    bread and butter                   1985     1     1
All three should count toward "bread and butter". We strip the _TAG suffix
from every token before matching, and lowercase everything, so all variants
map to the same cleaned string.

OUTPUT COLUMNS
--------------
  word1, word2            – the candidate pair
  order1 / order2         – the two orderings as strings
  order1_count / order2_count  – raw match_count summed across all years
                                 and all capitalisation/POS-tag variants
  novel                   – True if both counts are 0
"""

import csv
import gzip
import json
import urllib.request
import sys
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_URL = (
    "http://storage.googleapis.com/books/ngrams/books/"
    "googlebooks-eng-all-3gram-20120701-{key}.gz"
)

TOTALCOUNTS_URL = (
    "http://storage.googleapis.com/books/ngrams/books/"
    "googlebooks-eng-all-totalcounts-20120701.txt"
)

TOTAL_CORPUS_WORDS    = 128_126_390_255  # exact token count, Google Ngrams 2012 English
REFERENCE_CORPUS_SIZE = 350_000_000     # 350M tokens — threshold denominator
NOVELTY_THRESHOLD     = TOTAL_CORPUS_WORDS / REFERENCE_CORPUS_SIZE  # ≈ 366


# ── total corpus word count ───────────────────────────────────────────────────

def fetch_total_word_count() -> int:
    """
    Stream the Google Ngrams 2012 total-counts file and return the sum of
    all unigram tokens across all years.

    File format (tab-separated):
        year  total_tokens  total_pages  total_volumes
    """
    print(f"Fetching total word count from Google Ngrams …")
    req = urllib.request.Request(
        TOTALCOUNTS_URL,
        headers={"User-Agent": "Mozilla/5.0 (research; binomial-novelty-checker)"},
    )
    total = 0
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")

    # The file is a single line of comma-separated values grouped in fours:
    # year,match_count,page_count,volume_count,year,match_count,…
    entries = raw.strip().split(",")

    # Entries come in groups of 4: year, match_count, page_count, volume_count
    for i in range(0, len(entries) - 3, 4):
        try:
            total += int(entries[i + 1])
        except (ValueError, IndexError):
            pass

    print(f"Total tokens in Google Ngrams 2012 corpus: {total:,}")
    return total


# ── token / ngram helpers ────────────────────────────────────────────────────

def strip_pos(token: str) -> str:
    """
    Remove a POS-tag suffix and lowercase the token.
      'wolves_NOUN'  →  'wolves'
      'and_CONJ'     →  'and'
      'wolves'       →  'wolves'
      '_END_'        →  '_end_'   (won't match our targets)
    A suffix counts as a POS tag only if it starts with an uppercase letter.
    """
    idx = token.rfind("_")
    if idx > 0 and token[idx + 1 : idx + 2].isupper():
        return token[:idx].lower()
    return token.lower()


def clean_ngram(raw: str) -> str:
    """
    Strip POS tags from all three tokens and lowercase.
    Uses find() instead of split() to avoid list allocation.
    Returns '' for records that aren't exactly 3 space-separated tokens.
    """
    s1 = raw.find(" ")
    if s1 < 0:
        return ""
    s2 = raw.find(" ", s1 + 1)
    if s2 < 0:
        return ""
    if raw.find(" ", s2 + 1) >= 0:   # 4+ tokens → skip
        return ""
    return strip_pos(raw[:s1]) + " " + strip_pos(raw[s1 + 1:s2]) + " " + strip_pos(raw[s2 + 1:])


def file_key(word: str) -> str:
    """Return the 2-char (or 1-char) file key for a word."""
    w = word.lower()
    if len(w) >= 2 and w[:2].isalpha():
        return w[:2]
    return w[:1]


# ── streaming ────────────────────────────────────────────────────────────────

def stream_and_count(key: str, targets: set) -> dict:
    """
    Stream the .gz file for `key` and count all occurrences of each target
    (targets are lowercase, untagged 3-gram strings like 'wolves and thimbles').

    Raw counts are summed across all years and all POS/capitalisation variants.

    Returns {target_string: total_match_count}.
    """
    # Index targets by their first word for fast per-line rejection.
    # ~99.9% of lines are skipped after checking just one token.
    first_word_index: dict = defaultdict(set)
    for t in targets:
        sp = t.find(" ")
        first_word_index[t[:sp] if sp >= 0 else t].add(t)

    url = BASE_URL.format(key=key)
    counts = defaultdict(int)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (research; binomial-novelty-checker)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            with gzip.open(resp, "rb") as gz:
                for line in gz:
                    # Work in bytes — defer decode until both filters pass (~99.9% of lines never decoded)
                    tab = line.find(b"\t")
                    if tab < 0:
                        continue
                    ngram = line[:tab]
                    s1 = ngram.find(b" ")
                    if s1 < 0:
                        continue
                    s2 = ngram.find(b" ", s1 + 1)
                    if s2 < 0:
                        continue
                    # Middle-token "and" check (bytes)
                    if ngram[s1 + 1:s1 + 4].lower() != b"and":
                        continue
                    # First-word check — decode only this token
                    fw = strip_pos(ngram[:s1].decode("utf-8", errors="ignore"))
                    if fw not in first_word_index:
                        continue
                    # Full match — decode third token and build clean string
                    clean = fw + " and " + strip_pos(ngram[s2 + 1:].decode("utf-8", errors="ignore"))
                    if clean in targets:
                        t2 = line.find(b"\t", tab + 1)
                        t3 = line.find(b"\t", t2 + 1)
                        if t2 >= 0 and t3 >= 0:
                            counts[clean] += int(line[t2 + 1:t3])
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(
                f"  [INFO] No file for key '{key}' (HTTP 404) "
                "— treating all its targets as absent.",
                file=sys.stderr,
            )
        else:
            print(f"  [WARNING] HTTP {e.code} for key '{key}'", file=sys.stderr)
    except Exception as e:
        print(f"  [WARNING] Error streaming key '{key}': {e}", file=sys.stderr)

    return dict(counts)


# ── I/O ──────────────────────────────────────────────────────────────────────

def read_candidates(path: str) -> list:
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            w1, w2 = row[0].strip().lower(), row[1].strip().lower()
            if w1 in ("word1", ""):
                continue
            pairs.append((w1, w2))
    return pairs


def write_results(results: list, all_path: str, novel_path: str) -> list:
    all_fields = [
        "word1", "word2",
        "order1", "order1_count",
        "order2", "order2_count",
        "novel",
    ]
    with open(all_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        w.writerows(results)

    novel = [r for r in results if r["novel"]]
    with open(novel_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["word1", "word2"])
        w.writeheader()
        for r in novel:
            w.writerow({"word1": r["word1"], "word2": r["word2"]})
    return novel


def apply_frequency_threshold(results: list) -> list:
    """
    Re-label results using a frequency-based novelty threshold.

    A pair is marked novel if the COMBINED count of both orderings is below
    NOVELTY_THRESHOLD — i.e. fewer than 1 occurrence per REFERENCE_CORPUS_SIZE
    tokens when summing across both orders.
    """
    print(f"\nNovelty threshold: {NOVELTY_THRESHOLD:.1f} combined occurrences "
          f"(1 per {REFERENCE_CORPUS_SIZE:,} of {TOTAL_CORPUS_WORDS:,} tokens)")
    for r in results:
        r["novel"] = (r["order1_count"] + r["order2_count"]) < NOVELTY_THRESHOLD
    return results


# ── parallel worker (must be at module level for Windows spawn) ───────────────

def _worker(args):
    key, targets = args
    return key, stream_and_count(key, targets)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check N-and-N binomial pairs against Google Books 3-gram corpus "
            "(streamed from the internet, raw counts)."
        )
    )
    parser.add_argument("--input",     default=str(PROJECT_ROOT / "Data" / "candidates.csv"))
    parser.add_argument("--output",    default=str(PROJECT_ROOT / "Data" / "novel_binomials.csv"))
    parser.add_argument("--all",       default=str(PROJECT_ROOT / "Data" / "all_results.csv"))
    parser.add_argument("--checkpoint",default=str(PROJECT_ROOT / "Data" / "checkpoint.json"))
    args = parser.parse_args()

    # ── 1. Load candidates ──────────────────────────────────────────────────
    pairs = read_candidates(args.input)
    print(f"Loaded {len(pairs)} candidate pairs.")

    # ── 2. Group targets by file key ────────────────────────────────────────
    # target strings are lowercase + untagged: "w1 and w2"
    # Capitalised variants all clean to the same string, so we need only one.
    file_to_targets: dict = defaultdict(set)
    pair_to_targets: dict = {}

    for w1, w2 in pairs:
        t1 = f"{w1} and {w2}"   # ordering 1 target (lowercase)
        t2 = f"{w2} and {w1}"   # ordering 2 target (lowercase)
        file_to_targets[file_key(w1)].add(t1)
        file_to_targets[file_key(w2)].add(t2)
        pair_to_targets[(w1, w2)] = (t1, t2)

    n_files = len(file_to_targets)
    print(f"Need to stream {n_files} unique files (each streamed exactly once).\n")

    # ── 3. Stream each file once (parallelised across all cores) ───────────
    checkpoint_path = Path(args.checkpoint)

    # Load checkpoint if it exists
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            checkpoint = json.load(f)
        all_counts = checkpoint.get("counts", {})
        done_keys  = set(checkpoint.get("done_keys", []))
        print(f"Resuming: {len(done_keys)}/{n_files} files already done.\n")
    else:
        all_counts = {}
        done_keys  = set()

    jobs = [(k, v) for k, v in sorted(file_to_targets.items()) if k not in done_keys]
    remaining = len(jobs)
    completed = n_files - remaining

    def save_checkpoint():
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"counts": all_counts, "done_keys": list(done_keys)}, f)

    with ProcessPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(_worker, job): job[0] for job in jobs}
        with tqdm(total=n_files, initial=completed, desc="Files", unit="file", position=0, dynamic_ncols=True) as outer:
            for fut in as_completed(futures):
                key = futures[fut]
                targets = file_to_targets[key]
                completed += 1
                try:
                    _, counts = fut.result()
                except Exception as e:
                    tqdm.write(f"  [ERROR] key '{key}': {e}", file=sys.stderr)
                    counts = {}
                all_counts.update(counts)
                done_keys.add(key)
                save_checkpoint()
                found = sum(1 for t in targets if counts.get(t, 0) > 0)
                outer.update(1)
                tqdm.write(f"  '{key}' done -> {found}/{len(targets)} targets found")

    # ── 4. Build per-pair results ───────────────────────────────────────────
    results = []
    for w1, w2 in pairs:
        t1, t2 = pair_to_targets[(w1, w2)]
        c1 = all_counts.get(t1, 0)
        c2 = all_counts.get(t2, 0)
        results.append(
            {
                "word1":        w1,
                "word2":        w2,
                "order1":       f"{w1} and {w2}",
                "order1_count": c1,
                "order2":       f"{w2} and {w1}",
                "order2_count": c2,
                "novel":        c1 == 0 and c2 == 0,   # updated below
            }
        )

    # ── 5. Apply frequency threshold ────────────────────────────────────────
    results = apply_frequency_threshold(results)

    # ── 6. Write output ─────────────────────────────────────────────────────
    novel = write_results(results, args.all, args.output)
    n_novel = len(novel)
    n_total = len(results)

    print(f"\nDone. {n_novel}/{n_total} pairs are novel "
          f"(combined count < {NOVELTY_THRESHOLD:.1f}).")
    print(f"  All results -> {args.all}")
    print(f"  Novel pairs -> {args.output}")


if __name__ == "__main__":
    main()
