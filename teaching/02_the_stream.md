# 02 — The stream

## Bridge

You've worked with memory-mapped files before (basind segments, Steel pages). You know the move: instead of loading a big file into RAM, you tell the OS "this file is logically a numpy array" and it pages bytes in as you access them. Zero-copy. Constant memory footprint regardless of file size.

That's exactly what `train.bin` is. It's a single flat array of `uint16` integers, memory-mapped, concatenated from every document in the corpus. No delimiters. No document boundaries. One continuous river of token IDs.

Parallel from Lacan: in the real of language, there is no "end of document." Speech doesn't punctuate itself; the subject's discourse runs on. The corpus-as-stream treats all texts as one continuous chain of signifiers, and the model has to learn where the natural breaks are (if they matter at all) from statistical regularity alone.

## The decisive move: stratified split

You have 35 source documents. You need to hold some out for validation (to measure the model honestly — loss on text it trained on is meaningless). How?

**Naive option A**: hold out 3 whole documents. Problem: your val set only measures generalization to *those specific authors*. If Lacan is in train and Collins in val, you'd really be measuring cross-domain transfer, not honest val loss.

**Naive option B**: randomly remove 10% of tokens anywhere. Problem: incoherent val text. Half a sentence here, a word there. You can't interpret val loss.

**What SIEVE does (stratified)**: for each document independently, carve out a contiguous 10% slice from a random starting position. Put those slices together → val. Everything else → train. Now val has coherent passages from *every* source, so val loss measures "how well does the model model this corpus." Each source contributes proportionally.

Test set (5%) is carved the same way, held apart, and **never used for selection**. It's the court of last appeal — if ensemble selection started overfitting the val set across 200 generations, the test set would reveal it.

## ❓ Predict

Before running:

1. Look at `metadata.json` in `corpus_v008`: 81,920 train tokens, 9,216 val tokens, 4,608 test tokens. If each token is a uint16 (2 bytes), how many bytes is `train.bin` on disk? Compute it.
2. If you open `train.bin` with `np.memmap(..., dtype="uint16")`, what is `len(arr)`?
3. If `arr[42] == 317`, what does that integer mean? What do you need in order to turn it back into text?

Write your answers.

## Run

```bash
python teaching/exercises/02_stream.py
```

## What it means

A few things should click:

- **The file size is exactly `train_tokens × 2` bytes.** No headers, no delimiters. The bytes ARE the token IDs. That's it.
- The first 30 or so tokens, decoded, give you some readable text — the beginning of whichever document happened to sort first. But the next 30 after some big index is a totally different book, butted up against it with no seam.
- The val set is smaller but structured the same way: concatenated random slices, one per source.

**Why this matters for the training loop**: Stage 3 (next lesson) is "draw a random position `i` and take tokens `[i : i+block_size+1]`." Because the stream has no boundaries, some training windows will straddle the seam between two documents. The model just sees "text that transitions from Mann to Fink mid-sequence." It learns a distribution over all of it.

This is why the split is stratified in the FIRST place: you don't want the val set to contain text that also appears in the train set, even indirectly. Since each doc's val slice is disjoint from its train slices, that's guaranteed.

## The pitfall to avoid

Don't think of `train.bin` as "a dataset." It's **a buffer**. The actual dataset is the set of all `(x, y)` pairs you can form by sliding a window over it. At block_size=512, a 81,920-token buffer yields `81,920 - 512 = 81,408` distinct training windows — many of which overlap by 511 tokens. That redundancy is fine; epoch shuffling randomizes which ones you sample.

## Check

1. If you added a new document to the corpus and rebuilt, would the bytes at `train.bin[0:1000]` be the same as before? Why / why not?
2. If you wanted to *know* that your val set doesn't leak into train, what invariant would you check?
3. What would go wrong if you used raw random 10% token removal instead of stratified contiguous slices?

→ Next: [03_the_window.md](03_the_window.md) — turning the stream into (input, target) pairs.
