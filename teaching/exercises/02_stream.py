"""Lesson 02 — the stream.

Run: python teaching/exercises/02_stream.py
"""
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "data/corpus_v008"

# 1. Read metadata — this tells you how to interpret the bytes
meta = json.loads((CORPUS / "metadata.json").read_text())
print("metadata.json says:")
print(f"  tokenizer    = {meta['tokenizer_path']}")
print(f"  vocab_size   = {meta['vocab_size']}")
print(f"  dtype        = {meta['dtype']}")
print(f"  train_tokens = {meta['train_tokens']}")
print(f"  val_tokens   = {meta['val_tokens']}")
print(f"  test_tokens  = {meta['test_tokens']}")
print()

# 2. Open train.bin as a memory-mapped numpy array
#    Notice: no parsing, no headers, no delimiters. The bytes ARE the token IDs.
train = np.memmap(CORPUS / "train.bin", dtype="uint16", mode="r")
val   = np.memmap(CORPUS / "val.bin",   dtype="uint16", mode="r")

print(f"train.bin on disk: {(CORPUS/'train.bin').stat().st_size} bytes")
print(f"train[] as uint16: shape={train.shape}  (expect {meta['train_tokens']})")
print(f"val[]   as uint16: shape={val.shape}    (expect {meta['val_tokens']})")
print(f"  → {meta['train_tokens']} tokens × 2 bytes = {meta['train_tokens']*2} bytes ✓")
print()

# 3. Look at raw integers
print("train[0:30]:", list(train[:30]))
print()

# 4. Decode a window back to text using the tokenizer
tok_path = REPO / meta["tokenizer_path"]
tok = Tokenizer.from_file(str(tok_path))

def decode(arr):
    return tok.decode([int(x) for x in arr])

print("Decoded train[0:80]:")
print(f"  {decode(train[:80])!r}")
print()
print("Decoded train[10000:10080]  (somewhere in the middle):")
print(f"  {decode(train[10000:10080])!r}")
print()
print("Decoded val[0:80]:")
print(f"  {decode(val[:80])!r}")
print()

# 5. The cruical realization: document boundaries are invisible
#    Somewhere in this buffer, one book ends and another begins with no marker.
#    The model just sees the transition as "two tokens that happen to be adjacent."
print("Notice: there is NO delimiter between documents.")
print("At some index (we don't know where), Mann ends and Fink begins mid-stream.")
print("This is fine — the model learns from every sliding window.")
