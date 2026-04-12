"""
collect_c4_templates.py
-----------------------
Stream C4 shards and collect:
  1. Template sentences per POS category — sentences with "X and Y" where
     X and Y are the same POS, with the pair replaced by {W1}/{W2}.
  2. Background sentences — clean sentences with no "and", for mixing into
     the fine-tuning corpus.

Stops as soon as both targets are met (typically finishes in 1–2 shards).

Output
------
  Data/templates.csv          – pos, template, orig_w1, orig_w2
  Data/background_sentences.csv – text

Usage
-----
  python Scripts/collect_c4_templates.py
  python Scripts/collect_c4_templates.py --target-per-pos 300
"""

import csv
import re
import sys
import argparse
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGET_PER_POS    = 200
TARGET_BACKGROUND = 100_000
MIN_LEN = 50
MAX_LEN = 250

DEFAULT_BINOMIALS = str(PROJECT_ROOT / "Data" / "novel_binomials_curated.csv")

# POS categories we want templates for
KEEP_POS = {"NOUN", "ADJ", "VERB", "ADV"}
POS_NORMALIZE = {"PROPN": "NOUN", "NUM": "NOUN"}

AND_RE = re.compile(r'\b\w+\s+and\s+\w+\b', re.IGNORECASE)


def load_binomial_index(path: str) -> dict:
    """
    Load binomials and build an inverted index word → set of frozensets.
    Used to quickly check whether a sentence contains any excluded binomial.
    """
    from collections import defaultdict
    idx = defaultdict(set)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w1 = row["word1"].strip().lower()
            w2 = row["word2"].strip().lower()
            pair = frozenset({w1, w2})
            idx[w1].add(pair)
            idx[w2].add(pair)
    return idx


def sentence_has_binomial(sent: str, word_index: dict) -> bool:
    """Return True if the sentence contains any binomial (w1 + and + w2)."""
    words = set(re.findall(r"\b[a-zA-Z'-]+\b", sent.lower()))
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


def normalize_pos(pos: str) -> str:
    return POS_NORMALIZE.get(pos, pos)


def load_spacy():
    try:
        import spacy
    except ImportError:
        sys.exit("Run:  pip install spacy")
    try:
        # Keep parser so we can use dependency relations for better coord detection
        return spacy.load("en_core_web_sm", disable=["ner"])
    except OSError:
        sys.exit("Run:  python -m spacy download en_core_web_sm")


def split_sentences(text: str) -> list:
    """Fast sentence splitter — no NLTK, just punctuation + newlines."""
    sentences = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            for part in re.split(r"(?<=[.!?])\s+", line):
                part = part.strip()
                if part:
                    sentences.append(part)
    return sentences


def find_coord_pairs(doc):
    """
    Use spaCy dependency parse to find coordinated word pairs joined by 'and'.
    Returns list of (left_token, right_token, pos_category).

    Looks for tokens with dep='conj' whose head is connected via a 'cc' 'and'.
    Falls back to simple adjacent-token heuristic if no conj relations found.
    """
    results = []
    seen = set()

    # Primary: dependency-based coordination
    for token in doc:
        if token.dep_ == "conj":
            head = token.head
            # Check if they're linked by 'and'
            has_and = any(
                c.text.lower() == "and" and c.dep_ == "cc"
                for c in head.children
            )
            if not has_and:
                continue
            left, right = head, token
            lpos = normalize_pos(left.pos_)
            rpos = normalize_pos(right.pos_)
            pair_key = (left.i, right.i)
            if (pair_key not in seen
                    and lpos == rpos
                    and lpos in KEEP_POS
                    and left.is_alpha and right.is_alpha
                    and not left.is_stop and not right.is_stop
                    and left.text.lower() != right.text.lower()):
                results.append((left, right, lpos))
                seen.add(pair_key)

    # Fallback: simple adjacent-token heuristic
    if not results:
        for i, tok in enumerate(doc):
            if tok.text.lower() == "and" and tok.pos_ == "CCONJ":
                if 0 < i < len(doc) - 1:
                    left  = doc[i - 1]
                    right = doc[i + 1]
                    lpos  = normalize_pos(left.pos_)
                    rpos  = normalize_pos(right.pos_)
                    pair_key = (left.i, right.i)
                    if (pair_key not in seen
                            and lpos == rpos
                            and lpos in KEEP_POS
                            and left.is_alpha and right.is_alpha
                            and not left.is_stop and not right.is_stop
                            and left.text.lower() != right.text.lower()):
                        results.append((left, right, lpos))
                        seen.add(pair_key)

    return results


def make_template(sent_text: str, left_tok, right_tok) -> str:
    """Replace left_tok with {W1} and right_tok with {W2}."""
    ls, le = left_tok.idx, left_tok.idx + len(left_tok.text)
    rs, re_ = right_tok.idx, right_tok.idx + len(right_tok.text)
    offset = len("{W1}") - (le - ls)
    out = sent_text[:ls] + "{W1}" + sent_text[le:]
    rs2, re2 = rs + offset, re_ + offset
    return out[:rs2] + "{W2}" + out[re2:]


def main():
    parser = argparse.ArgumentParser(
        description="Collect template and background sentences from C4."
    )
    parser.add_argument("--templates",
                        default=str(PROJECT_ROOT / "Data" / "templates.csv"))
    parser.add_argument("--background",
                        default=str(PROJECT_ROOT / "Data" / "background_sentences.csv"))
    parser.add_argument("--binomials", default=DEFAULT_BINOMIALS,
                        help="CSV of binomial pairs to exclude from background sentences.")
    parser.add_argument("--target-per-pos", type=int, default=TARGET_PER_POS)
    parser.add_argument("--target-background", type=int, default=TARGET_BACKGROUND)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Run:  pip install datasets")

    try:
        from tqdm import tqdm
    except ImportError:
        sys.exit("Run:  pip install tqdm")

    nlp = load_spacy()
    print("spaCy loaded.")

    binomial_index = load_binomial_index(args.binomials)
    print(f"Loaded binomial index ({len(binomial_index)} words) for background filtering.")

    tmpl_counts = defaultdict(int)   # pos → n collected
    bg_count    = 0

    templates_f  = open(args.templates,  "w", newline="", encoding="utf-8")
    background_f = open(args.background, "w", newline="", encoding="utf-8")
    tmpl_writer  = csv.DictWriter(templates_f,
                                  fieldnames=["pos", "template", "orig_w1", "orig_w2"])
    bg_writer    = csv.DictWriter(background_f, fieldnames=["text"])
    tmpl_writer.writeheader()
    bg_writer.writeheader()

    def templates_done():
        return all(tmpl_counts[p] >= args.target_per_pos for p in KEEP_POS)

    def background_done():
        return bg_count >= args.target_background

    print("Streaming C4 shards until targets met …\n")

    for shard_idx in range(1024):
        if templates_done() and background_done():
            break

        url = (f"hf://datasets/allenai/c4/en/"
               f"c4-train.{shard_idx:05d}-of-01024.json.gz")
        try:
            dataset = load_dataset("json", data_files=[url],
                                   split="train", streaming=True)
        except Exception as e:
            print(f"  [WARN] shard {shard_idx}: {e}")
            continue

        with tqdm(desc=f"Shard {shard_idx:04d}", unit="doc",
                  dynamic_ncols=True) as pbar:
            for example in dataset:
                if templates_done() and background_done():
                    break

                text = example.get("text", "")
                if not text:
                    pbar.update(1)
                    continue

                for sent in split_sentences(text):
                    if not (MIN_LEN <= len(sent) <= MAX_LEN):
                        continue

                    has_and = "and" in sent.lower()

                    # Background: clean sentences that contain no binomials
                    if not background_done() and not sentence_has_binomial(sent, binomial_index):
                        bg_writer.writerow({"text": sent})
                        bg_count += 1

                    # Templates: sentences with "and" that match our regex
                    if not templates_done() and has_and and AND_RE.search(sent):
                        doc = nlp(sent)
                        for left_tok, right_tok, pos in find_coord_pairs(doc):
                            if tmpl_counts[pos] < args.target_per_pos:
                                tmpl = make_template(sent, left_tok, right_tok)
                                tmpl_writer.writerow({
                                    "pos":      pos,
                                    "template": tmpl,
                                    "orig_w1":  left_tok.text,
                                    "orig_w2":  right_tok.text,
                                })
                                tmpl_counts[pos] += 1

                pbar.update(1)

        print(f"  After shard {shard_idx:04d} — templates: "
              + ", ".join(f"{p}:{tmpl_counts[p]}" for p in sorted(KEEP_POS))
              + f"  background: {bg_count:,}")

    templates_f.close()
    background_f.close()

    print("\nDone.")
    print("Template counts per POS:")
    for pos in sorted(KEEP_POS):
        print(f"  {pos:5s}: {tmpl_counts[pos]:,}")
    print(f"Background: {bg_count:,}")
    print(f"  Templates   → {args.templates}")
    print(f"  Background  → {args.background}")


if __name__ == "__main__":
    main()
