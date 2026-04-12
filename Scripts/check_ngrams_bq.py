"""
check_ngrams_bq.py
------------------
Check all candidate binomial pairs against the Google Books 3-gram corpus
using BigQuery's public dataset.  Runs in seconds rather than hours.

Setup (one-time):
    pip install google-cloud-bigquery
    gcloud auth application-default login

Usage:
    python check_ngrams_bq.py [--project YOUR_GCP_PROJECT_ID]

The Google Cloud free tier includes 1 TB of query data/month — more than enough.
"""

import csv
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Public BigQuery table for Google Books 1-grams through 5-grams (2012 corpus).
# If this exact name has changed, browse:
#   https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=books_ngrams_2012
BQ_TABLE = "bigquery-public-data.google_books_ngrams_2020.eng_3"


# ── helpers ───────────────────────────────────────────────────────────────────

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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Check binomial novelty via BigQuery (Google Books 3-gram corpus)."
    )
    parser.add_argument("--input",   default=str(PROJECT_ROOT / "Data" / "candidates.csv"))
    parser.add_argument("--output",  default=str(PROJECT_ROOT / "Data" / "novel_binomials.csv"))
    parser.add_argument("--all",     default=str(PROJECT_ROOT / "Data" / "all_results.csv"))
    parser.add_argument(
        "--project", default=None,
        help="Google Cloud project ID to bill the query to (uses your default if omitted)."
    )
    args = parser.parse_args()

    # ── 1. Load candidates ────────────────────────────────────────────────────
    pairs = read_candidates(args.input)
    print(f"Loaded {len(pairs)} candidate pairs.")

    # ── 2. Build target phrase list (both orderings) ──────────────────────────
    pair_to_targets: dict = {}
    targets_set: set = set()
    for w1, w2 in pairs:
        t1 = f"{w1} and {w2}"
        t2 = f"{w2} and {w1}"
        pair_to_targets[(w1, w2)] = (t1, t2)
        targets_set.add(t1)
        targets_set.add(t2)

    all_targets = sorted(targets_set)
    print(f"Checking {len(all_targets)} target phrases (both orderings).")

    # ── 3. Run BigQuery ───────────────────────────────────────────────────────
    try:
        from google.cloud import bigquery
    except ImportError:
        sys.exit(
            "google-cloud-bigquery not installed.\n"
            "Run:  pip install google-cloud-bigquery"
        )

    client = bigquery.Client(project=args.project)

    # Strip POS tags in SQL:  wolves_NOUN and_CONJ thimbles_NOUN  →  wolves and thimbles
    # REGEXP_REPLACE removes any _UPPERCASE suffix from each token.
    # The WHERE clause uses a parameterised array to avoid SQL injection and
    # to keep the query string short regardless of how many targets there are.
    query = f"""
        SELECT
            LOWER(term)            AS clean_ngram,
            SUM(term_frequency)    AS total_count
        FROM `{BQ_TABLE}`
        WHERE LOWER(term) IN UNNEST(@targets)
        GROUP BY LOWER(term)
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("targets", "STRING", all_targets)
        ]
    )

    print(f"Submitting query to BigQuery table: {BQ_TABLE}")
    print("(Full table scan — typically completes in under a minute.)\n")

    job = client.query(query, job_config=job_config)
    rows = job.result()           # blocks until done

    bytes_processed = job.total_bytes_processed or 0
    print(f"Query complete.  Data scanned: {bytes_processed / 1e9:.1f} GB")

    # ── 4. Collect counts ─────────────────────────────────────────────────────
    counts: dict = {}
    for row in rows:
        counts[row.clean_ngram] = row.total_count

    print(f"Phrases with count > 0: {len(counts)}\n")

    # ── 5. Build per-pair results ─────────────────────────────────────────────
    all_rows = []
    novel = []
    for w1, w2 in pairs:
        t1, t2 = pair_to_targets[(w1, w2)]
        c1 = counts.get(t1, 0)
        c2 = counts.get(t2, 0)
        is_novel = (c1 == 0 and c2 == 0)
        result = {
            "word1":        w1,
            "word2":        w2,
            "order1":       t1,
            "order1_count": c1,
            "order2":       t2,
            "order2_count": c2,
            "novel":        is_novel,
        }
        all_rows.append(result)
        if is_novel:
            novel.append(result)

    # ── 6. Write outputs ──────────────────────────────────────────────────────
    all_fields = ["word1", "word2", "order1", "order1_count", "order2", "order2_count", "novel"]
    with open(args.all, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        w.writerows(all_rows)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["word1", "word2"])
        w.writeheader()
        for r in novel:
            w.writerow({"word1": r["word1"], "word2": r["word2"]})

    print(f"Novel pairs : {len(novel)}/{len(pairs)}")
    print(f"All results -> {args.all}")
    print(f"Novel pairs -> {args.output}")


if __name__ == "__main__":
    main()
