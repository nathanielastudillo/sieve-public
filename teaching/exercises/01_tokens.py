"""Lesson 01 — tokens.

Run: python teaching/exercises/01_tokens.py
"""
from pathlib import Path
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[2]
TOK_PATH = REPO / "data/tokenizers/v005/tokenizer.json"

tok = Tokenizer.from_file(str(TOK_PATH))
vocab_size = tok.get_vocab_size()
print(f"Loaded tokenizer: {TOK_PATH.relative_to(REPO)}")
print(f"Vocab size: {vocab_size}")
print()

sentences = [
    "In Lacan's formulation, the subject is split.",
    "The transference structures the analytic encounter.",
    "the cat sat",
    "xylophonist countertransference",
    "Michael Mann distinguishes four sources of social power.",
]

for s in sentences:
    enc = tok.encode(s)
    ids = enc.ids
    pieces = enc.tokens
    # Round-trip: decode back to text
    decoded = tok.decode(ids)
    print(f"text:    {s!r}")
    print(f"ids:     {ids}")
    print(f"pieces:  {pieces}")
    print(f"decoded: {decoded!r}")
    print(f"n_tokens: {len(ids)}  (chars: {len(s)})")
    print()

# Inspect a few specific vocab entries so you can SEE what a "token" is
print("=" * 60)
print("A few tokens from the vocabulary (id -> piece):")
print("=" * 60)
vocab = tok.get_vocab()  # dict: piece -> id
by_id = {v: k for k, v in vocab.items()}
for i in [0, 1, 2, 3, 4, 100, 500, 1000, 4000, 7999]:
    if i in by_id:
        print(f"  {i:5d} -> {by_id[i]!r}")

print()
print("Try changing `sentences` above and rerunning.")
print("Observe: which of your words become single tokens? Which fragment?")
