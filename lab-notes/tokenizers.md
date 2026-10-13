# Tokenizers

BPE tokenizer versions live in `data/tokenizers/v{NNN}/`. Each dir contains
`tokenizer.json` (HuggingFace `tokenizers` format), `config.json` (training
parameters), `metrics.json` (quality metrics).

## Summary table

| Version | Vocab | Trained on | Compression | Fertility | Domain integrity | Notes |
|---------|------:|------------|------------:|----------:|-----------------:|-------|
| v001    | 8,000 | 18 cleaned files, 3.05M tokens | 4.88 b/tok | 1.54 tok/word | 38.6% | First cut, too small |
| v002    | 16,000 | 18 cleaned files, 3.06M tokens | 5.31 b/tok | 1.42 tok/word | 65.7% | Used by corpus_v003 + v005 |
| v003    | 16,000 | 37 cleaned files, 3.41M tokens | 4.92 b/tok | 1.48 tok/word | 75.7% | 2026-04-17. Used by corpus_v006. |
| **v004** | **4,000** | **37 cleaned files** | **4.11 b/tok** | **1.77 tok/word** | **31.4%** | **2026-04-17. Used by corpus_v007. Compute-optimal vocab for ~7.4M params per Tao 2024.** |
| **v005** | **8,000** | **37 cleaned files** | **4.54 b/tok** | **1.60 tok/word** | **61.4%** | **2026-04-17. Used by corpus_v008. Middle-ground vocab.** |

## Metrics defined

- **Compression ratio** — bytes per token. Higher = more information
  packed per token at inference/training time. Bounded above by the
  average word length (~5-6 bytes/word in English).
- **Fertility** — tokens per word. 1.0 = every word is a single token.
  Common English text under a well-matched BPE runs ~1.3-1.5.
- **Domain integrity %** — fraction of a curated list of 70 domain terms
  (psychoanalytic, social-theory, physics vocab) that survive as single
  tokens. The headline quality metric for this corpus.

## v001 (historical)

- 8K vocab was too aggressive — most domain terms broken into 3-5 subwords:
  `deterritorialization` → 4 tokens, `reterritorialization` → 5, `eigenvector` → 5
- Fertility 1.54 is noticeably high; keeps happening because 8K vocab
  can't reach the frequency threshold for long compound terms
- Deprecated once v002 showed much better metrics

## v002 (current)

- 16K vocab — sweet spot for 3M-token corpus
- 65.7% of domain terms are single-token (46 of 70)
- Still-fragmenting terms: `deterritorialization`, `reterritorialization`,
  `hermeneutics`, `schrodinger`, `eigenvector`, `renormalization`
  (all 4-5 tokens). Acceptable — these are rare and very long.
- Both active corpora (v003, v005) tokenize against this. Because v005's
  source pool includes authors (Deleuze, Kristeva, Schuster) that *weren't*
  in v002's training data, there's a mild OOD tax: idiosyncratic terms
  like `jouissance`, `rhizome`, `chora` have fertilities > 1 in v005 but
  weren't measured at training time.

## v003 (2026-04-17)

Trained on the expanded 37-source pool (the same `data/cleaned/` that
corpus_v004 draws from). Same vocab size, same config as v002 —
apples-to-apples.

- **Domain integrity: 75.7%** (53/70 single-token, up from 46/70 for v002)
- **Compression: 4.92 b/tok** (down from 5.31) — slightly fewer bytes per
  token, i.e. tokens are a bit shorter on average. Expected given more
  diverse vocabulary competes for the same 16K slots.
- **Fertility: 1.48 tok/word** (up from 1.42). Same explanation.
- **Newly single-token** (relative to v002): `deleuze`, `rhizome`,
  `jouissance`, `schizophrenia`, `assemblage`, `deterritorialization`,
  `reterritorialization`, `schizo`, and ~10 others that live in the
  new authors.
- **Still multi-token**: `eigenvector` (5), `electromagnetic` (5),
  `renormalization` (4), `schrodinger` (4), `lagrangian` (3),
  `eigenvalue` (3). Physics terms lose out — very few physics sources
  in the pool.
- Train time: ~1 min on gpu-01 CPU.

Used by **corpus_v006** (experiment 004). Direct test of
whether the v002 tokenizer mismatch was the cause of v005's worse
ensemble loss vs v003.

## v004 (4K) and v005 (8K) — 2026-04-17 (Magikarp ablation)

Added after the Magikarp diagnostic (see `findings.md` 2026-04-17 later).
At 85K training tokens, the 16K vocab has 40% never-seen entries. Tao
2024's scaling law recommends vocab in the low thousands for ~7.4M
params. These tokenizers test the lower end.

- **v004 (4K vocab):** domain integrity collapses to 31.4% — most
  theoretical terms now fragment heavily. But only 326 vocab entries
  are never-seen in a 85K training split, and 16.4% of test tokens
  land on rare (≤10 occurrences) vocab entries vs 28.7% for 16K.
- **v005 (8K vocab):** domain integrity 61.4% — similar to v002. Middle
  ground. 23.5% test Magikarp exposure.

**Naming collision note:** tokenizer-v004/v005 vs corpus-v004/v005.
`data/tokenizers/v{N}/` and `data/corpus_v{N}/` are different
filesystem trees so no conflict, but be careful reading references.

## Vocab-size ↔ Magikarp summary

From `scripts/magikarp_check.py` on the 100K-token corpora:

| Tokenizer | Corpus | Vocab | Never-seen | Test-tok @freq≤10 | ln(V) |
|-----------|--------|------:|-----------:|------------------:|------:|
| v002 | v003 | 16K | 40.3% | 28.7% | 9.68 |
| v002 | v005 | 16K | 41.9% | 28.4% | 9.68 |
| v003 | v006 | 16K | 40.8% | 28.7% | 9.68 |
| **v005** | **v008** | **8K** | **19.9%** | **23.5%** | **8.99** |
| **v004** | **v007** | **4K** | **8.2%** | **16.4%** | **8.29** |
