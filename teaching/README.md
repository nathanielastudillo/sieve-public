# Teaching — SIEVE from the ground up

A hands-on path from "bag of PDFs" to "text coming out of a .pt file." Built for deliberate practice: every lesson has a small script you actually run, and questions you answer before reading the explanation.

## How to use this

Each lesson is short. The format is always:

1. **Bridge** — how this maps to something you already understand
2. **❓ Predict** — one or two questions. *Actually stop and answer them before reading on.* The prediction is the learning; reading the answer afterward is just calibration.
3. **Run the exercise** — `python teaching/exercises/NN_name.py`. It prints things. Look at them.
4. **What it means** — the concept, now that you have the concrete object in front of you.
5. **Check** — a small question you should be able to answer now. If you can't, reread.

The path builds. Don't skip — each lesson uses primitives from the previous one.

## Setup

```bash
cd ~/dev/sieve
# Everything should already be installed. If an import fails:
#   pip install torch numpy tiktoken tokenizers pymupdf
```

The exercises use real artifacts from this repo:

- `data/corpus_v008/` — 95,749 tokens of cleaned corpus (clinical + theory)
- `data/tokenizers/v005/tokenizer.json` — a BPE tokenizer, vocab 8000
- `runs/smoke_v003/checkpoints/*.pt` — real trained checkpoints

## The path

| # | Lesson | The question it answers |
|---|---|---|
| 01 | [Tokens](01_tokens.md) | What does a tokenizer learn, and why turn language into integers? |
| 02 | [The stream](02_the_stream.md) | Why is the corpus one long int array with no document boundaries? |
| 03 | [The sliding window](03_the_window.md) | What does "next-token prediction" actually look like in memory? |
| 04 | [The lookup table](04_the_lookup.md) | How does the model turn an integer into a vector? Why weight tying? |
| 05 | [Attention and the block](05_attention.md) | What does a transformer block actually do to the residual stream? |
| 06 | [Loss and learning](06_learning.md) | How does cross-entropy become a gradient that updates the weights? |
| 07 | [The .pt file](07_the_pt_file.md) | What is actually stored, and how do we reconstruct a model from it? |
| 08 | [Generation](08_generation.md) | Temperature, top-k, multinomial — what does each knob do? |
| 09 | [The ensemble](09_ensemble.md) | Why does averaging *logits* approximate Solomonoff induction? |

## The invariants

By the end you should hold these three ideas tightly:

1. **After tokenization, everything is integer arithmetic on a token stream.** Documents, sentences, whitespace — gone. The model lives in `int → int`.
2. **The target `y` is just `x` shifted by one position.** Cross-entropy on this IS the language model. There is no other objective.
3. **A `.pt` file is a Python dict of named tensors.** Reconstructing a model = `GPT(config)` allocates empty tensors; `load_state_dict` fills them by name. The class definition must match what was saved.

Start with [01_tokens.md](01_tokens.md).
