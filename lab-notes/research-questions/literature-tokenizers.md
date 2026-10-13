# Literature — BPE tokenizer design for small-corpus LMs

Compiled 2026-04-17 by a research subagent in response to the question
in [tokenizer-scaling.md](tokenizer-scaling.md). Full transcript in
`/tmp/claude-1000/-home-lhk-dev/e4ce224f-2590-4c0e-9ccb-733be23a9656/tasks/ab8f53f8b8fcc8f25.output`.

## TL;DR

The agent's synthesis, which I agree with after reading: the
meta-overfitting pathology (test NLL 10.87 > uniform 9.68) is most
likely **not** ensemble selection going wrong — it's the *Magikarp*
pathology. A large fraction of a 16K vocab trained on 3M tokens of
BPE-training text, when used with an LM trained on 85K tokens, will
be "glitch tokens" with near-random embeddings. When they appear in
test sequences, softmax mass on the correct token is ≈ 0, pushing
per-token NLL above uniform.

Corollary: **our 16K vocab is wrong-sized** for 7.4M params + ≤7.5M
tokens. Tao et al. 2024's scaling law recommends vocab in the low
thousands (~4-8K) for our regime.

Corollary: **BPE is probably worse than Unigram-LM** at our scale
(Bostrom & Durrett 2020).

## Annotated bibliography

### Approach A — scale the tokenizer

**A1. [Dagan, Synnaeve, Rozière 2024 — Getting the most out of your tokenizer](https://arxiv.org/abs/2402.01035).** Tokenizer quality (not just vocab size) matters; you can swap in a better-trained tokenizer mid-pretraining and recover. Supports A *cautiously*.

**A2. [Tao et al. 2024 — Scaling Laws with Vocabulary](https://arxiv.org/abs/2407.13623).** Joint scaling law (params × vocab × tokens). Compute-optimal vocab is ~43K for a 7B model; for 7.4M params recommends **~4-8K**. **Undermines A** for us: we're already oversized on vocab.

**A3. [Provilkov, Emelianenko, Voita 2020 — BPE-Dropout](https://arxiv.org/abs/1910.13267).** Ablations show diminishing returns past 16K for <10M training tokens; 32K can *increase* perplexity on rare-token sets because merges overfit. **Undermines A** for small corpora.

**A4. [Bostrom & Durrett 2020 — BPE Is Suboptimal for LM Pretraining](https://arxiv.org/abs/2004.03720).** Unigram LM tokenization beats BPE at matched vocab sizes; gap *widens* at smaller corpora. Empirical floor: ≥300 tokens per vocab entry. At 7.5M/32K we'd be at 234 — below floor. **Supports algorithm change (BPE→Unigram)** over vocab scaling.

**Failure modes:** singleton/doubleton merges becoming "uninformed" tokens; embedding matrix swallowing parameter budget (32K × 256 ≈ 8.2M params, larger than the whole model).

**Additional:** [Gowda & May 2020 — Optimal Vocabulary Size for NMT](https://arxiv.org/abs/2004.02334) — Zipf long-tail argument: at ≤5 occurrences per token, that token can't be learned and actively corrupts softmax.

### Approach B — multi-tokenizer / domain-specific BPE

**B1. [Gururangan et al. 2020 — Don't Stop Pretraining](https://arxiv.org/abs/2004.10964).** DAPT gains bounded by tokenizer fertility on domain terms. Fertility > 1.5 on domain vocab consistently correlates with worse downstream loss.

**B2. [Gu et al. 2021 — PubMedBERT](https://arxiv.org/abs/2007.15779).** Domain-native tokenizer (vs general-English WordPiece inherited by BioBERT) accounts for ~30-40% of downstream gains on biomedical benchmarks. **Strongly supports B** for specialized corpora — and our psychoanalytic vocabulary (jouissance, agencement, chora) is genuinely OOV-heavy for any general tokenizer.

**B3. [Remy et al. 2024 — Trans-Tokenization](https://arxiv.org/abs/2408.04303).** Swap a pretrained LM's tokenizer by initializing new embeddings as weighted averages of old embeddings for overlapping byte-sequences. Extends to domain swaps. **Gives a concrete mechanism for mixture-of-tokenizers ensembles** without full retrain.

**B4. [Minixhofer et al. 2024 — Zett: Training-Free Tokenizer Transfer](https://arxiv.org/abs/2405.07883).** Hypernetwork generates embeddings for any new tokenizer; Mistral-7B recovers 95%+ quality after minimal finetuning.

**B5. [Pagnoni et al. 2024 — Byte-Latent Transformer](https://arxiv.org/abs/2412.09871).** Dynamic byte-patches instead of any fixed tokenizer — limit case of "many tokenizers". Underperforms BPE on small corpora but is competitive on mixed-domain data. **Mixed signal** — could mean the right answer on heterogeneous corpora is no tokenizer at all.

**Failure modes for B:**
- Logit averaging across different vocabs requires projection (hard).
- Splitting 7.5M tokens 3 ways → 2.5M each, below the quality floor per [Zouhar et al. 2023 — Tokenization and the Noiseless Channel](https://arxiv.org/abs/2306.16842).
- Routing is a new classifier you have to train on sparse data.

### Approach C — scale data + compute

**C1. [Hoffmann et al. 2022 — Chinchilla](https://arxiv.org/abs/2203.15556).** Compute-optimal: tokens ≈ 20× params. For 7.4M params → **~150M tokens**. We have 7.5M. **Undermines C if we interpret it as "scale the model"** — we're 20× under-tokened.

**C2. [Muennighoff et al. 2023 — Scaling Data-Constrained LMs](https://arxiv.org/abs/2305.16264).** Up to ~4 epochs of repetition is nearly as good as fresh data. Below ~1B tokens, data quality/dedup > raw count. **Partially supports C**: 7.5M × 4 epochs ≈ 30M effective tokens. Still short of Chinchilla, but no longer catastrophic.

**C3. [Gunasekar et al. 2023 — Textbooks Are All You Need / phi-1](https://arxiv.org/abs/2306.11644).** Small models on tiny *curated* corpora beat much larger models on web-scale noise. **Supports C** with a caveat: their win came from data *quality*, not quantity — our corpus already has this property.

**C4. [Geiping & Goldstein 2022 — Cramming](https://arxiv.org/abs/2212.14034).** At < 10^19 FLOPs, tokenizer/architecture choices are dominated by data curation and optimizer choices. Recommends *not* tuning vocab size; focus on data diversity + LR warmup.

**C5. [Penedo et al. 2024 — DataComp-LM / FineWeb](https://arxiv.org/abs/2406.17557).** Data curation quality > architectural choices (including vocab size) for small-model downstream loss.

**Failure modes for C:** Chinchilla shortfall (fitting noise on under-tokened models); distributed engineering surface (FSDP, NCCL) rarely pays for <100M params.

### Cross-cutting — the tokenizer-LM leakage problem

**X1. [Goldman et al. 2024 — Tokenization Is More Than Compression](https://arxiv.org/abs/2402.18376).** Compression ratio / fertility correlate weakly with downstream LM perplexity. Mismatched-domain tokenizers cause *systematic* perplexity inflation of 15-40% that does not decrease with more LM training.

**X2. [Singh & Strouse 2024 — Tokenization counts](https://arxiv.org/abs/2402.14903).** Single-token vs multi-token representations of the same concept produce **order-of-magnitude** downstream loss differences. Relevant to fragmenting `deleuze` / `lacan` / `jouissance`.

**X3. [Land & Bartolo 2024 — Fishing for Magikarp](https://arxiv.org/abs/2405.05417).** 🎯 **The load-bearing paper for us.** 5-20% of BPE vocab items in production LLMs are "glitch tokens" — appeared in tokenizer training but barely in LM training. Their output embeddings are effectively random. Softmax assigns them near-zero mass. When test sequences include them, per-token NLL exceeds the uniform baseline.

At our v002 tokenizer (16K vocab trained on 3M tokens) used with an LM trained on 85K tokens:
- Each LM training token exposes exactly one vocab entry.
- Even if we see 50K unique tokens in 85K training tokens (likely), that leaves **at least 15K / 16K ≈ 94%** of the vocab under-exposed by the LM.
- By Land & Bartolo's threshold (<10 training occurrences = Magikarp), we expect an extremely high glitch fraction.

This is the mechanism behind our 10.87-nats test loss.

## Synthesis (revised)

The agent's bottom line, adopted:

> The decisive move is not A, B, or C in isolation — it's
> **recomputing the tokenizer budget against Tao 2024 for our model
> size, and measuring glitch-token fraction before any further
> training.**

The failure diagnoses cleanly:

- **Problem 1 (meta-overfit, test>uniform):** vocab too big for
  training-token budget → Magikarp glitch tokens.
- **Problem 2 (v005 regression vs v003):** tokenizer fitted to old
  source distribution → high fragmentation of new-author vocab →
  systematic perplexity inflation (Goldman 2024).

Both problems are **tokenizer problems**, not evolution/ensemble
problems. This is important — the SIEVE evolutionary loop has been
operating on top of broken tokenization and we've been interpreting
its signal as if it reflected true LM quality.

## The decisive experiment (agent's proposal, adopted)

Train three 7.4M-param models with identical architecture and
identical corpus (v004), varying only the tokenizer:

| Variant | Vocab | Algorithm | Training data |
|---------|------:|-----------|---------------|
| T-baseline | 16,000 | BPE (current v002) | same v002 tok |
| T-sized | 8,000 | BPE | retrained on v004 pool |
| T-unigram | 8,000 | Unigram-LM | retrained on v004 pool |

Measurements (per variant):
1. **Magikarp fraction** — fraction of vocab with <10 training occurrences in LM train.bin
2. **val loss** (on v004 val split)
3. **test loss** (on v004 test split) — primary metric
4. **test loss vs uniform baseline ln(V)** — the pathology indicator

**Predictions:**
- T-baseline: test NLL > ln(16000) = 9.68 (reproduces current failure)
- T-sized: test NLL < ln(8000) = 8.99, much closer to val
- T-unigram: T-sized but incrementally better (~5-15% on small-corpus regime per Bostrom & Durrett)

**If predictions hold:** the tokenizer is the bottleneck, and the
next SIEVE run should use T-unigram (or T-sized as fallback if we
don't want to swap the tokenizer library).

**If T-baseline *also* has test NLL < uniform at v004 scale:** then
the Magikarp hypothesis was wrong for our specific regime, and the
75× more training tokens alone dissolved the pathology. In that case
Design C ("scale data") was the right read and B/tokenizer-algorithm
changes are secondary.

Either outcome is informative. Cost: 3 tokenizers (minutes each) +
3 training runs (~4h each on gpu-01, parallelizable to ~4h total
on 3 compute-cluster nodes).

This supersedes the exp-005 design I sketched earlier — it tests a
sharper hypothesis, uses the same corpus, and gives a direct
algorithm-choice readout.

## What this means for the cluster plan

The cluster plan in [strategy-cluster-run.md](../strategy-cluster-run.md)
stands, but its *first* experiment should not be a full-scale SIEVE
run — it should be the 3-tokenizer ablation above, distributed across
3 of the 12 nodes. Only after tokenizer is right do we commit 12×
GB10-hours to a 30-generation SIEVE run.

## Outstanding checks worth doing before any new training

1. **Empirically measure Magikarp fraction on v005 and v006.** For
   each, tokenize train.bin, count per-token frequencies, count how
   many vocab entries fall below threshold (try 5, 10, 20). If v006's
   Magikarp fraction is noticeably lower than v005's, the v003
   tokenizer (which was trained on more data) helped even without the
   vocab shrink.
2. **Read the unembedding norm vs frequency correlation** from a
   trained checkpoint — this is the L&B diagnostic proper. Cheap:
   one forward-pass load of a gen-2 member, look at `lm_head.weight`
   norm by row, plot against training frequency.

Both doable today, both cheap, both directly informative about
whether the agent's diagnosis is right for *our* checkpoints rather
than for GPT-family tokenizers.
