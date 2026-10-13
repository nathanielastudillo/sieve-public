# 01 — Tokens

## Bridge

You already know the Lacanian move: language is not a mirror of things, it's a chain of differences between signifiers. A word means what it means because it is *not* the other words. No signifier has inherent content; it's pure positional value in a system of symbols.

A tokenizer operationalizes exactly this. It doesn't know what "transference" *means*. It only knows: there are ~8,000 possible symbols in this universe, and "transference" happens to be symbol #4,217. Every other symbol is "not-4,217." The model will learn which symbols tend to follow which — pure second-order statistics over a system of differences.

In programming terms: a tokenizer is a **compression dictionary** for text. It maps arbitrary unicode strings into a finite alphabet of small integers. A vocabulary of 8,000 means there are 8,000 possible "atoms" your model can ever see. That's it.

## Why not just use characters? Or whole words?

**Characters** (vocab ≈ 100): the model wastes most of its capacity learning that "t-h-e" is a word. Inefficient.

**Whole words** (vocab ≈ millions): every rare word is a new token nobody's seen. Can't generalize.

**BPE (byte-pair encoding)** is the middle path: start with characters, repeatedly merge the most common adjacent pair into a new token, stop at `vocab_size` tokens. Result: common words get one token, rare words decompose into pieces.

"psychoanalysis" might be one token if it appears often in the corpus. Or it might be `psycho` + `analysis`. Or `psych` + `oan` + `alysis`. *Whatever the corpus suggests.* The tokenizer is trained on your corpus — different corpus, different merges, different vocabulary.

## ❓ Predict

Before running the exercise:

1. The sentence `"In Lacan's formulation, the subject is split"` — how many tokens do you think this becomes for an 8,000-vocab BPE trained on psychoanalytic/theory text? Guess a single number.
2. Which word do you expect to be a single token: `"Lacan"` or `"the"`? Why?
3. If a word is NOT in the vocab (say, `"xylophonist"`), what happens? Error? Unknown token? Something else?

Write your guesses down. Then:

## Run

```bash
cd ~/dev/sieve
python teaching/exercises/01_tokens.py
```

## What it means

Look at what came back:

- `ids` — a Python list of integers. This is what the model sees. Nothing else.
- `tokens` — the string pieces each ID corresponds to. Most have a leading `Ġ` or `▁` or space — that's the tokenizer's way of encoding "this piece follows a word boundary."
- Notice some words collapse to one token (common ones); others explode into multiple pieces (rare ones).

**The decisive insight**: the tokenizer *learned* this vocabulary by finding the most statistically frequent substrings in the training corpus. A different corpus produces a different tokenizer, which produces a different token stream, which trains a different model. This is why SIEVE versions tokenizers (`v001`, `v002`, ...) — it's a hyperparameter.

The vocab size is also a trade-off: smaller = more pieces per word = longer sequences = more compute per document, but better handling of rare strings. Larger = fewer pieces = shorter sequences, but each embedding vector is rarer (less training signal per vector).

## The pitfall to avoid

The tokenizer is **coupled to the model**. If you train a model with vocab 8000 and then try to `load_state_dict` from a checkpoint that used vocab 50257, the embedding matrix shape doesn't match and loading fails. The tokenizer is part of the model's identity, not a detachable preprocessing step.

This is why every SIEVE corpus directory has a `metadata.json` naming the exact tokenizer it used.

## Check

Before moving on, make sure you can answer:

1. If vocab_size = 8000 and your tokenizer was trained on psychoanalytic text, how would you expect it to handle a mathematical physics paper? (Hint: think about which substrings are frequent.)
2. Why does the dataloader use `np.uint16` when vocab_size < 65536 but `np.uint32` when it's larger?
3. What does a single integer like `4217` *mean*, intrinsically? (Trick question.)

→ Next: [02_the_stream.md](02_the_stream.md) — what happens after you tokenize everything.
