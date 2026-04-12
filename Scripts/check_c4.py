"""
check_c4.py
-----------
Stream the English C4 corpus from HuggingFace and count how many times each
N-and-N binomial pair appears as an exact 3-word sequence ("w1 and w2" or
"w2 and w1", case-insensitive, whole-word matches).

Also writes a sentences file with every document that contains a match,
labelled by pair and ordering.

Parallelised across 12 workers, each processing independent C4 shards
(C4/en has 1024 shards).  Checkpoint tracks completed shards so the run
can be safely resumed.

Usage
-----
  python Scripts/check_c4.py
  python Scripts/check_c4.py --workers 12
  python Scripts/check_c4.py --input Data/novel_binomials_curated.csv

Output files
------------
  c4_all_results.csv         – counts for every pair
  c4_novel_binomials.csv     – pairs where both counts are 0
  c4_binomial_sentences.csv  – word1, word2, ordering, sentence (document text)
"""

import csv
import json
import re
import os
import sys
import time
import argparse
import multiprocessing
import threading
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

C4_TOTAL_SHARDS  = 1024     # fixed property of allenai/c4 en train split
C4_TOTAL_DOCS    = 364_868_892   # approximate total documents in C4/en train
COUNTER_INTERVAL = 500      # worker increments shared counter every N docs

# ── module-level worker state (set once per process via initializer) ──────────

_worker_patterns       = None
_worker_target_to_pair = None
_worker_doc_counter    = None   # shared multiprocessing.Value


def _init_worker(pattern_strings, target_to_pair, doc_counter):
    global _worker_patterns, _worker_target_to_pair, _worker_doc_counter
    _worker_patterns = {
        t: re.compile(p, re.IGNORECASE)
        for t, p in pattern_strings.items()
    }
    _worker_target_to_pair = target_to_pair
    _worker_doc_counter    = doc_counter


def _process_shard(shard_index: int) -> tuple:
    """
    Worker task: stream one C4 shard and collect counts + matching documents.
    Returns (shard_index, counts_dict, sentences_list, error_or_None, n_docs).
    """
    from datasets import load_dataset

    url = (f"hf://datasets/allenai/c4/en/"
           f"c4-train.{shard_index:05d}-of-01024.json.gz")

    counts    = defaultdict(int)
    sentences = []
    n_docs    = 0
    _local    = 0   # local accumulator before flushing to shared counter

    try:
        dataset = load_dataset("json", data_files=[url],
                               split="train", streaming=True)
        for example in dataset:
            n_docs  += 1
            _local  += 1

            # Flush to shared counter periodically
            if _local >= COUNTER_INTERVAL:
                with _worker_doc_counter.get_lock():
                    _worker_doc_counter.value += _local
                _local = 0

            text = example.get("text", "")
            if text and "and" in text.lower():
                for target, pat in _worker_patterns.items():
                    if pat.search(text):
                        counts[target] += 1
                        w1, w2 = _worker_target_to_pair[target]
                        sentences.append({
                            "word1":    w1,
                            "word2":    w2,
                            "ordering": target,
                            "sentence": text.strip(),
                        })

        # Flush remainder
        if _local:
            with _worker_doc_counter.get_lock():
                _worker_doc_counter.value += _local

    except Exception as e:
        return shard_index, {}, [], str(e), n_docs

    return shard_index, dict(counts), sentences, None, n_docs


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


def build_pattern_strings(pairs: list) -> dict:
    """Returns {target_string: regex_string} — serialisable for worker init."""
    patterns = {}
    for w1, w2 in pairs:
        for t in (f"{w1} and {w2}", f"{w2} and {w1}"):
            if t not in patterns:
                w1t, _, w2t = t.split(" ")
                patterns[t] = (rf"\b{re.escape(w1t)}\b\s+\band\b"
                               rf"\s+\b{re.escape(w2t)}\b")
    return patterns


def build_target_to_pair(pairs: list) -> dict:
    mapping = {}
    for w1, w2 in pairs:
        mapping[f"{w1} and {w2}"] = (w1, w2)
        mapping[f"{w2} and {w1}"] = (w1, w2)
    return mapping


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
        description="Count N-and-N binomial occurrences in C4 (parallelised by shard)."
    )
    parser.add_argument("--input",
                        default=str(PROJECT_ROOT / "Data" / "novel_binomials_curated.csv"))
    parser.add_argument("--output",
                        default=str(PROJECT_ROOT / "Data" / "c4_novel_binomials.csv"))
    parser.add_argument("--all",
                        default=str(PROJECT_ROOT / "Data" / "c4_all_results.csv"))
    parser.add_argument("--sentences",
                        default=str(PROJECT_ROOT / "Data" / "c4_binomial_sentences.csv"))
    parser.add_argument("--checkpoint",
                        default=str(PROJECT_ROOT / "Data" / "c4_checkpoint.json"))
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help=f"Parallel worker processes (default: {os.cpu_count()}).")
    parser.add_argument("--limit-shards", type=int, default=None,
                        help="Stop after this many shards (useful for testing).")
    args = parser.parse_args()

    try:
        from tqdm import tqdm
    except ImportError:
        sys.exit("Run:  pip install tqdm")

    # ── 1. Load candidates ──────────────────────────────────────────────────
    pairs           = read_candidates(args.input)
    pattern_strings = build_pattern_strings(pairs)
    target_to_pair  = build_target_to_pair(pairs)
    print(f"Loaded {len(pairs)} candidate pairs → {len(pattern_strings)} patterns.")

    # ── 2. Load checkpoint ──────────────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    sentences_path  = Path(args.sentences)

    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            ckpt = json.load(f)
        counts         = defaultdict(int, ckpt.get("counts", {}))
        done_shards    = set(ckpt.get("done_shards", []))
        docs_per_shard = ckpt.get("docs_per_shard", {})
        print(f"Resuming: {len(done_shards)}/{C4_TOTAL_SHARDS} shards already done.\n")
    else:
        counts         = defaultdict(int)
        done_shards    = set()
        docs_per_shard = {}

    total_shards  = args.limit_shards or C4_TOTAL_SHARDS
    remaining     = [i for i in range(total_shards) if i not in done_shards]
    docs_done_so_far = sum(docs_per_shard.get(str(i), 0) for i in done_shards)
    print(f"Shards remaining: {len(remaining)}  |  Workers: {args.workers}\n")

    # Open sentences file for appending (safe to resume)
    sent_file_new = not (sentences_path.exists() and done_shards)
    sentences_f   = open(sentences_path, "a", newline="", encoding="utf-8")
    sent_writer   = csv.DictWriter(sentences_f,
                                   fieldnames=["word1", "word2", "ordering", "sentence"])
    if sent_file_new:
        sent_writer.writeheader()

    def save_checkpoint():
        sentences_f.flush()
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"counts": dict(counts),
                       "done_shards": sorted(done_shards),
                       "docs_per_shard": docs_per_shard}, f)

    # ── 3. Shared counter + polling thread ──────────────────────────────────
    doc_counter  = multiprocessing.Value('l', docs_done_so_far)
    stop_polling = threading.Event()

    def poll_doc_counter(doc_pbar):
        """Background thread: update the document tqdm bar in real time."""
        last = docs_done_so_far
        while not stop_polling.is_set():
            time.sleep(2)
            current = doc_counter.value
            delta   = current - last
            if delta > 0:
                doc_pbar.update(delta)
                last = current

    # ── 4. Process shards in parallel ───────────────────────────────────────
    print(f"Processing {len(remaining)} shards across {args.workers} workers …\n")

    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(pattern_strings, target_to_pair,
                                       doc_counter)) as pool:

        futures = {pool.submit(_process_shard, i): i for i in remaining}

        with tqdm(total=total_shards, initial=len(done_shards),
                  desc="Shards   ", unit="shard",
                  position=0, dynamic_ncols=True) as shard_pbar, \
             tqdm(total=C4_TOTAL_DOCS, initial=docs_done_so_far,
                  desc="Documents", unit="doc", unit_scale=True,
                  position=1, dynamic_ncols=True) as doc_pbar:

            poll_thread = threading.Thread(target=poll_doc_counter,
                                           args=(doc_pbar,), daemon=True)
            poll_thread.start()

            for fut in as_completed(futures):
                shard_idx, shard_counts, shard_sentences, err, n_docs = fut.result()

                if err:
                    shard_pbar.write(f"  [ERROR] shard {shard_idx:04d}: {err}")
                else:
                    for target, n in shard_counts.items():
                        counts[target] += n
                    for row in shard_sentences:
                        sent_writer.writerow(row)
                    docs_per_shard[str(shard_idx)] = n_docs

                done_shards.add(shard_idx)
                save_checkpoint()
                shard_pbar.update(1)

                if shard_sentences:
                    shard_pbar.write(f"  shard {shard_idx:04d}: "
                                     f"{sum(shard_counts.values())} matches")

            stop_polling.set()
            poll_thread.join()

    sentences_f.close()
    print(f"\nFinished. {len(done_shards)} shards processed.")

    # ── 5. Build and write results ───────────────────────────────────────────
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

    novel = write_results(results, args.all, args.output)
    print(f"\n{len(novel)}/{len(results)} pairs novel (both counts = 0).")
    print(f"  All results  → {args.all}")
    print(f"  Novel pairs  → {args.output}")
    print(f"  Sentences    → {args.sentences}")


if __name__ == "__main__":
    main()
