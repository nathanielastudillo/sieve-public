# SIEVE Operator Guide

You have a codebase that trains populations of small transformers, evaluates them as logit-averaged ensembles, and evolves the population using ensemble-level fitness. The output artifact is not a single model — it's an ensemble specification: a list of checkpoint paths and configs that, when loaded together and their logits averaged, produce the best predictor found.

This guide gets you from zero to a running evolutionary loop.

---

## 1. Mental Model

```
                    YOU ARE HERE
                         │
    ┌────────────────────▼────────────────────┐
    │              PREPARE CORPUS              │
    │   PDFs/text → clean → tokenize → split  │
    │   Output: train.bin, val.bin             │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │           GENERATION 0 (INIT)            │
    │   Sample 32 random hyperparameter configs│
    │   Train each as a small GPT (minutes)    │
    │   Save checkpoints                       │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │          ENSEMBLE EVALUATION             │
    │   Try thousands of 8-model subsets       │
    │   Average their logits on val set        │
    │   Score each combination by val loss     │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │         SELECTION & REPRODUCTION         │
    │   Keep models that improve ensembles     │
    │   Kill models that never help            │
    │   Mutate/crossover → next generation     │
    └────────────────────┬────────────────────┘
                         │
                    loops back to TRAIN
                    until loss plateaus
                    (50-200 generations)
                         │
    ┌────────────────────▼────────────────────┐
    │            FINAL ARTIFACT                │
    │   best_ensemble.json                     │
    │   = list of 8 checkpoint paths + configs │
    │   Inference: load all 8, average logits  │
    └─────────────────────────────────────────┘
```

The key insight: the unit of selection is the **ensemble combination**, not the individual model. A mediocre model that covers blind spots of other models is more valuable than a strong model that's redundant with the rest. This selects for diversity, which is the Solomonoff approximation.

---

## 2. Prerequisites

```bash
cd ~/dev/sieve
source .venv/bin/activate

# Verify
python -c "from model import GPT, ModelConfig; print('OK')"
```

Dependencies already installed in `.venv/`: torch, tiktoken, numpy, pymupdf, sentencepiece, pyyaml, tqdm, matplotlib.

For GPU training: needs CUDA-capable PyTorch (already installed on GB10s).

---

## 3. Prepare Your Corpus

You need a directory of source documents (PDFs, text files, markdown). At 100K tokens, every token matters — curate aggressively.

### 3a. Curate first (optional but recommended)

```python
from data.curate import compute_density_score, deduplicate, score_passages

# Check information density of candidate texts
texts = [open(f).read() for f in my_files]
for f, t in zip(my_files, texts):
    print(f"{compute_density_score(t):.3f}  {f}")

# Remove near-duplicates
unique = deduplicate(texts, threshold=0.8)

# Rank by density — put the best ones in your corpus dir
ranked = score_passages(texts)
for score, text in ranked[:10]:
    print(f"  {score:.3f}  {text[:80]}...")
```

### 3b. Build train/val split

```bash
# Put your source docs in a directory
mkdir -p corpus/sources
cp /path/to/your/pdfs/*.pdf corpus/sources/
cp /path/to/your/texts/*.txt corpus/sources/

# Run the preparation pipeline
python -c "
from data.prepare import prepare_corpus
meta = prepare_corpus(
    source_dir='corpus/sources',
    output_dir='corpus',
    tokenizer_name='gpt2',   # or 'cl100k_base' for GPT-4 tokenizer
    val_fraction=0.1,
)
print(f'Train: {meta[\"train_tokens\"]:,} tokens')
print(f'Val:   {meta[\"val_tokens\"]:,} tokens')
print(f'Sources: {meta[\"num_sources\"]}')
"
```

This creates:
- `corpus/train.bin` — memory-mapped token array (90%)
- `corpus/val.bin` — held-out validation set (10%)
- `corpus/metadata.json` — token counts, source hashes, split info

### 3c. Optional: train a custom tokenizer

For domain-specific corpora (legal, clinical, physics), a custom BPE tokenizer may outperform GPT-2's:

```python
from data.prepare import train_custom_tokenizer

# First, dump all cleaned text to a single file
model_path = train_custom_tokenizer(
    text_path='corpus/all_text.txt',
    vocab_size=8192,        # smaller vocab = denser encoding for small corpora
)
print(f"Tokenizer saved to {model_path}")

# Then re-prepare corpus with custom tokenizer
from data.prepare import prepare_corpus
prepare_corpus(
    source_dir='corpus/sources',
    output_dir='corpus',
    custom_tokenizer_path=str(model_path),
)
```

---

## 4. Run the Evolutionary Loop

### 4a. Single-GPU (local, simplest)

```bash
python -m controller.orchestrator \
    --config config/default.yaml \
    --corpus-dir corpus \
    --checkpoint-dir checkpoints \
    --results-dir results \
    -v
```

This runs the full loop on your local GPU:
1. Initializes 32 random configs (population)
2. Trains each model sequentially (~minutes each at 20M params / 100K tokens)
3. Evaluates 5,000 random 8-model ensemble combinations
4. Selects elites, mutates, crossovers, immigrants → next generation
5. Repeats until loss plateaus or 200 generations

**Time estimate (single GB10):** ~5-15 min per generation × 50-200 generations = hours to a couple days.

### 4b. With a profile override

Profiles scale the evolutionary parameters to your cluster size:

```bash
# For 4-8 GB10 workers: conservative (pop=16, ensemble=4)
python -m controller.orchestrator --config config/default.yaml --profile small ...

# For 8-16 workers: balanced (pop=32, ensemble=8)
python -m controller.orchestrator --config config/default.yaml --profile medium ...

# For 16-32 workers: aggressive (pop=64, ensemble=12)
python -m controller.orchestrator --config config/default.yaml --profile large ...

# For 32-64 workers: brute-force (pop=128, ensemble=16)
python -m controller.orchestrator --config config/default.yaml --profile full ...
```

### 4c. Via the shell wrapper (includes node discovery + deployment)

```bash
./scripts/run_sieve.sh \
    --profile auto \
    --corpus-dir corpus \
    --deploy
```

This will:
1. Discover GB10 nodes on the <fabric>.x subnet
2. Auto-select a profile based on worker count
3. Deploy the sieve codebase to each worker via rsync
4. Launch the controller

---

## 5. What's Happening Inside

### The search space

Each population member gets a random config drawn from these ranges:

| Parameter | Range | Type |
|-----------|-------|------|
| `n_layer` | [4, 6, 8, 10, 12] | discrete |
| `n_head` | [4, 6, 8] | discrete |
| `n_embd` | [256, 384, 512] | discrete |
| `dropout` | [0.0, 0.3] | continuous (log-uniform) |
| `weight_decay` | [0.1, 3.0] | continuous |
| `learning_rate` | [1e-4, 1e-2] | continuous |
| `optimizer` | [adamw, muon] | discrete |
| `epochs` | [10, 200] | continuous |
| `warmup_fraction` | [0.01, 0.1] | continuous |
| `lr_schedule` | [cosine, linear, constant_with_warmup] | discrete |

The Muon optimizer is the default and generally outperforms AdamW on data-constrained training (Slowrun finding). It applies Newton-Schulz orthogonalization to gradients of 2D weight matrices — momentum in orthogonalized gradient space. Embeddings and layer norms fall back to AdamW automatically.

### Selection pressure

This is the novel piece. Standard PBT selects individual models. SIEVE selects *ensembles*.

After training a generation, the system:
1. Evaluates thousands of random 8-model subsets
2. For each subset: averages their logits on the val set, measures cross-entropy
3. Ranks members by **how often they appear in the top 10% of ensembles**
4. Elites (top 25%) survive and reproduce with mutated hyperparams
5. Complementary pairs (co-occur in good ensembles + maximally different configs) produce crossover children
6. 10% of slots go to fresh random immigrants (prevents diversity collapse)
7. Dead weight (never in any top ensemble) is killed

### Convergence detection

The loop stops when:
- Best ensemble loss hasn't improved in `stagnation_patience` generations (default: 20)
- Or `max_generations` reached (default: 200)

Watch for **diversity collapse** — if all members converge to similar configs, the ensemble degenerates toward a single solution. The immigration mechanism fights this, but if you see it happening, increase `immigration_fraction` or `mutation_strength`.

---

## 6. Outputs & Artifacts

Everything lands in `results/`:

```
results/
├── best_ensemble.json     # THE artifact — checkpoint list + configs
├── lineage.db             # SQLite: every member, ensemble, generation
└── (analysis plots if you generate them)
```

### best_ensemble.json

```json
{
    "ensemble_id": "sieve-gen-147-top1",
    "members": [
        {"checkpoint": "checkpoints/gen089/member_a1b2c3d4.pt", "config": {...}},
        {"checkpoint": "checkpoints/gen102/member_e5f6g7h8.pt", "config": {...}},
        ...
    ],
    "ensemble_size": 8,
    "val_loss": 1.847,
    "aggregation": "logit_average"
}
```

### Inference with the ensemble

```python
import torch
import json
from worker.checkpoint import load_model_from_checkpoint

spec = json.load(open("results/best_ensemble.json"))
device = "cuda"

# Load all members
models = []
for m in spec["members"]:
    model = load_model_from_checkpoint(m["checkpoint"], device=device)
    models.append(model)

# Inference: average logits
input_ids = torch.tensor([[...]], device=device)  # your input tokens
with torch.no_grad():
    all_logits = [model(input_ids)[0] for model in models]
    avg_logits = torch.stack(all_logits).mean(dim=0)
    probs = torch.softmax(avg_logits[:, -1, :], dim=-1)
    next_token = torch.multinomial(probs, 1)
```

---

## 7. Analysis & Visualization

After a run (or during — the lineage DB is updated live):

```python
from analysis.convergence import ConvergenceTracker
from analysis.visualize import (
    plot_loss_curve,
    plot_diversity_vs_loss,
    plot_lineage_tree,
    plot_hyperparameter_evolution,
    plot_ensemble_composition,
)

db = "results/lineage.db"

# Convergence curve — are we still improving?
plot_loss_curve(db, "results/loss_curve.png")

# Diversity vs loss — is diversity collapsing?
plot_diversity_vs_loss(db, "results/diversity_vs_loss.png")

# Lineage tree — who descended from whom?
plot_lineage_tree(db, "results/lineage_tree.png", max_generations=50)

# How did learning rate evolve across generations?
plot_hyperparameter_evolution(db, "learning_rate", "results/lr_evolution.png")

# Which members appear in the best ensembles?
plot_ensemble_composition(db, "results/ensemble_heatmap.png")

# Programmatic check
tracker = ConvergenceTracker(db)
print(f"Converged: {tracker.is_converged()}")
print(f"Diversity collapsed: {tracker.is_diversity_collapsed()}")
print(f"Estimated remaining improvement: {tracker.estimate_remaining_improvement():.4f}")
print(tracker.generation_summary(tracker.loss_curve()[0][-1]))
```

---

## 8. Key Config Knobs

Edit `config/default.yaml` or pass a custom config:

| Knob | Default | What it does | When to change |
|------|---------|--------------|----------------|
| `population_size` | 32 | Members per generation | More workers → bigger population |
| `ensemble_size` | 8 | Models per ensemble | Fewer if memory-constrained; more if you have many workers for inference |
| `elite_fraction` | 0.25 | What fraction survives | Lower = more aggressive selection, faster convergence but riskier |
| `immigration_fraction` | 0.1 | Fresh random blood | Increase if diversity is collapsing |
| `mutation_strength` | 0.2 | How much configs change | Higher = more exploration, lower = more exploitation |
| `eval_combinations` | 5000 | Ensemble combos to test | Higher = better selection signal but slower per generation |
| `stagnation_patience` | 20 | Generations without improvement before stopping | Lower for faster experiments, higher for thorough search |
| `max_generations` | 200 | Hard cap | Adjust based on compute budget |
| `training.epochs` | 50 | Epochs per member | More epochs = better-trained individuals but slower generations |
| `training.weight_decay` | 0.3 | Regularization strength | Slowrun says go up to 16x standard (i.e., 1.6+) for small corpora |

---

## 9. Gotchas & Operational Notes

**Val set is sacred.** The ensemble fitness function evaluates against val.bin. Over many generations, you're meta-optimizing against it. Mitigations: keep a tertiary test set that's never used for selection; if running very long (200+ generations), consider rotating the val set.

**Checkpoint disk usage.** Each 20M-param model is ~80MB. With pop=32 and 200 generations: 32 × 200 × 80MB = ~500GB. The system saves checkpoints aggressively (every `checkpoint_every` epochs). Prune old generations' checkpoints if disk is tight — only the best ensemble's checkpoints matter at the end.

**shuffle_per_epoch must be True.** Slowrun's key finding: data ordering has outsized impact on multi-epoch training of small corpora. The EpochShuffler re-seeds the shuffle each epoch (`base_seed + epoch`). Turning this off is almost always worse.

**Muon vs AdamW.** Muon generally wins on data-constrained training, but the search space includes AdamW so the evolution can discover if AdamW is better for specific architecture points. Don't force one or the other — let the evolution decide.

**torch.compile.** The trainer tries `torch.compile` automatically on CUDA. This adds ~30s compilation overhead per model but gives 20-30% training speedup. Worth it for longer training runs (50+ epochs). If it causes issues, it falls back gracefully.

**Memory.** A single 20M-param model in bf16 fits easily in a GB10's memory. Ensemble evaluation loads models one at a time and caches logits — memory scales with val set size × vocab size × ensemble size. For 8 models with vocab 50257 and 10K val tokens: 8 × 10K × 50K × 4 bytes ≈ 16GB — fine for a GB10.

**Lineage DB is live.** You can query `results/lineage.db` with sqlite3 while the run is in progress. The analysis/visualize tools read from it non-destructively.

---

## 10. Quick Reference

```bash
# Prepare corpus
python -c "from data.prepare import prepare_corpus; prepare_corpus('corpus/sources', 'corpus')"

# Run evolution (local, single GPU)
python -m controller.orchestrator --config config/default.yaml --corpus-dir corpus -v

# Run with profile
python -m controller.orchestrator --config config/default.yaml --profile small --corpus-dir corpus -v

# Check convergence mid-run
python -c "
from analysis.convergence import ConvergenceTracker
t = ConvergenceTracker('results/lineage.db')
gens, losses = t.loss_curve()
print(f'Generation {gens[-1]}: best ensemble loss = {losses[-1]:.4f}')
print(f'Converged: {t.is_converged()}')
"

# Generate all plots
python -c "
from analysis.visualize import *
db = 'results/lineage.db'
plot_loss_curve(db, 'results/loss.png')
plot_diversity_vs_loss(db, 'results/div.png')
plot_lineage_tree(db, 'results/tree.png')
plot_ensemble_composition(db, 'results/heatmap.png')
"

# Inspect a checkpoint
python -m worker.checkpoint info checkpoints/some_model.pt

# List all checkpoints
python -m worker.checkpoint list checkpoints/

# Evaluate a specific ensemble manually
python -m worker.evaluator \
    --checkpoints ckpt1.pt ckpt2.pt ckpt3.pt ckpt4.pt \
    --data-dir corpus --block-size 512

# Discover cluster nodes
./scripts/discover_nodes.sh

# Deploy to cluster + run
./scripts/run_sieve.sh --profile auto --corpus-dir corpus --deploy
```
