"""
curate_binomials.py
-------------------
Filter novel binomials to only those where both words are highly familiar.
Uses wordfreq (pip install wordfreq) to get word frequency scores.

Run:
    pip install wordfreq
    python Scripts/curate_binomials.py
"""

import csv
from pathlib import Path

try:
    from wordfreq import word_frequency
except ImportError:
    raise SystemExit("Run:  pip install wordfreq")

ROOT     = Path(__file__).resolve().parent.parent
IN_PATH  = ROOT / "Data" / "novel_binomials.csv"
OUT_PATH = ROOT / "Data" / "novel_binomials_familiar.csv"

# Threshold: words below this frequency are considered unfamiliar.
# Adjust to taste — lower = more permissive (keeps more pairs).
# Some reference points (approximate):
#   1e-5  : volcano, diamond, hurricane
#   1e-6  : gravedigger, peasant, avalanche
#   1e-7  : cordwainer, hosier, seneschal
THRESHOLD = 1e-7


REFERENCE_WORDS = [
    "gravedigger", "volcano", "hurricane", "peasant", "sapphire",
    "cordwainer", "hosier", "seneschal", "trebuchet", "seamstress",
    "kazoo", "trombone", "avalanche", "rhinoceros", "mammoth",
]


def main():
    print("Reference frequencies:")
    for w in REFERENCE_WORDS:
        f = word_frequency(w, "en")
        kept = "KEPT" if f >= THRESHOLD else "CUT"
        print(f"  {w:<20} {f:.2e}  [{kept}]")
    print()
    pairs = []
    with open(IN_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs.append((row["word1"], row["word2"]))

    familiar, excluded = [], []
    for w1, w2 in pairs:
        f1 = word_frequency(w1, "en")
        f2 = word_frequency(w2, "en")
        if f1 >= THRESHOLD and f2 >= THRESHOLD:
            familiar.append((w1, w2, f1, f2))
        else:
            excluded.append((w1, w2, f1, f2))

    # Print words that were cut so user can review
    print(f"\nKept {len(familiar)}/{len(pairs)} pairs (threshold={THRESHOLD:.0e})\n")

    # Show which words caused exclusions
    cut_words = {}
    for w1, w2, f1, f2 in excluded:
        if f1 < THRESHOLD:
            cut_words[w1] = f1
        if f2 < THRESHOLD:
            cut_words[w2] = f2
    print("Words that caused exclusions (word: frequency):")
    for w, f in sorted(cut_words.items(), key=lambda x: x[1]):
        print(f"  {w:<25} {f:.2e}")

    # Write output
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["word1", "word2"])
        for w1, w2, *_ in familiar:
            w.writerow([w1, w2])

    print(f"\nFamiliar novel pairs -> {OUT_PATH}")


if __name__ == "__main__":
    main()
