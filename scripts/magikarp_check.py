"""Magikarp diagnostic per Land & Bartolo 2024.

Counts per-token frequency in each corpus's train.bin, reports:
  - total vocab size
  - unique tokens actually seen in training (support)
  - tokens below threshold (Magikarp candidates)
  - fraction of vocab that's Magikarp at each threshold

Also computes uniform-baseline vs train-frequency-proportional baseline
loss — the latter is the floor a well-calibrated LM should beat.

Usage:  python scripts/magikarp_check.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

CORPORA = [
    ("corpus_v003 (tok=v002, 16K)", REPO / "data" / "corpus_v003"),
    ("corpus_v005 (tok=v002, 16K)", REPO / "data" / "corpus_v005"),
    ("corpus_v006 (tok=v003, 16K)", REPO / "data" / "corpus_v006"),
    ("corpus_v008 (tok=v005, 8K) ", REPO / "data" / "corpus_v008"),
    ("corpus_v007 (tok=v004, 4K) ", REPO / "data" / "corpus_v007"),
]

THRESHOLDS = [0, 1, 5, 10, 20, 50]


def analyze(name: str, corpus_dir: Path) -> dict:
    meta_path = corpus_dir / "metadata.json"
    train_path = corpus_dir / "train.bin"
    val_path = corpus_dir / "val.bin"
    test_path = corpus_dir / "test.bin"
    if not (meta_path.exists() and train_path.exists()):
        print(f"[{name}] missing files, skipping")
        return {}

    meta = json.loads(meta_path.read_text())
    V = int(meta["vocab_size"])

    train = np.memmap(train_path, dtype=np.uint16, mode="r")
    val = np.memmap(val_path, dtype=np.uint16, mode="r") if val_path.exists() else None
    test = np.memmap(test_path, dtype=np.uint16, mode="r") if test_path.exists() else None

    # Frequency histogram in train
    freq = np.bincount(train, minlength=V)
    assert freq.shape[0] == V, f"freq array {freq.shape[0]} != V={V}"

    unique_train = int((freq > 0).sum())
    magikarp_pct = {thr: float((freq <= thr).sum()) / V for thr in THRESHOLDS}

    # Coverage of val/test by training-frequency
    def coverage(arr):
        if arr is None:
            return None
        seen_in_train = (freq > 0)
        total = len(arr)
        hits = int(seen_in_train[np.asarray(arr, dtype=np.int64)].sum())
        return {"tokens": total, "seen_in_train": hits, "oov_to_train_pct": (total - hits) / total}

    # Uniform baseline NLL = ln(V)
    ln_V = math.log(V)

    # Train-frequency baseline NLL: what loss would a model predicting
    # p(token) = freq(token)/sum(freq) achieve on test set?
    def freq_baseline(arr):
        if arr is None:
            return None
        total = freq.sum()
        # Laplace smoothing (add 1) to avoid -inf on Magikarp tokens
        p = (freq + 1) / (total + V)
        log_p = np.log(p)
        nll = -float(log_p[np.asarray(arr, dtype=np.int64)].mean())
        return nll

    # What fraction of TEST *token occurrences* fall on a Magikarp vocab entry?
    # This is what actually causes test NLL to spike — each such token costs
    # ~ln(V) (worse, if its unembedding is pointed away).
    def magikarp_exposure(arr):
        if arr is None:
            return None
        ids = np.asarray(arr, dtype=np.int64)
        train_freq_of_test_tokens = freq[ids]
        return {
            thr: float((train_freq_of_test_tokens <= thr).sum()) / len(ids)
            for thr in THRESHOLDS
        }

    result = {
        "corpus": name,
        "vocab_size": V,
        "train_tokens": int(len(train)),
        "val_tokens": int(len(val)) if val is not None else None,
        "test_tokens": int(len(test)) if test is not None else None,
        "test_magikarp_exposure_by_threshold": magikarp_exposure(test),
        "val_magikarp_exposure_by_threshold": magikarp_exposure(val),
        "unique_train_tokens": unique_train,
        "unique_train_fraction": unique_train / V,
        "magikarp_pct_at_threshold": magikarp_pct,
        "ln_V_uniform_nll": ln_V,
        "train_freq_baseline_nll_val": freq_baseline(val),
        "train_freq_baseline_nll_test": freq_baseline(test),
        "val_coverage": coverage(val),
        "test_coverage": coverage(test),
        # Top and bottom of frequency distribution
        "top5_token_ids": freq.argsort()[-5:][::-1].tolist(),
        "top5_freqs": freq[freq.argsort()[-5:][::-1]].tolist(),
        "tokens_with_freq_0": int((freq == 0).sum()),
        "tokens_with_freq_1": int((freq == 1).sum()),
    }
    return result


def main():
    all_results = []
    for name, path in CORPORA:
        r = analyze(name, path)
        if not r:
            continue
        all_results.append(r)
        print(f"\n=== {name} (vocab={r['vocab_size']}, train={r['train_tokens']}) ===")
        print(f"  unique tokens in train: {r['unique_train_tokens']}"
              f" ({100*r['unique_train_fraction']:.1f}% of vocab)")
        print(f"  freq=0 (never seen):   {r['tokens_with_freq_0']}")
        print(f"  freq=1 (hapax):        {r['tokens_with_freq_1']}")
        print(f"  Magikarp fraction by threshold:")
        for thr, pct in r['magikarp_pct_at_threshold'].items():
            print(f"    freq <= {thr:>3}: {100*pct:>5.1f}% of vocab")
        print(f"  Uniform baseline NLL:        {r['ln_V_uniform_nll']:.3f} nats")
        if r['train_freq_baseline_nll_test'] is not None:
            print(f"  Train-freq baseline NLL test:{r['train_freq_baseline_nll_test']:.3f} nats "
                  f"(this is the floor a competent LM must beat)")
        if r['val_coverage']:
            vc = r['val_coverage']
            print(f"  Val OOV-to-train:            {100*vc['oov_to_train_pct']:.2f}% "
                  f"({vc['tokens']-vc['seen_in_train']}/{vc['tokens']})")
        if r['test_coverage']:
            tc = r['test_coverage']
            print(f"  Test OOV-to-train:           {100*tc['oov_to_train_pct']:.2f}% "
                  f"({tc['tokens']-tc['seen_in_train']}/{tc['tokens']})")
        if r.get('test_magikarp_exposure_by_threshold'):
            print(f"  Fraction of test-token occurrences landing on a rare-in-train vocab entry:")
            for thr, pct in r['test_magikarp_exposure_by_threshold'].items():
                print(f"    train-freq(tok) <= {thr:>3}: {100*pct:>5.1f}% of test tokens")

    out = REPO / "lab-notes" / "research-questions" / "magikarp_results.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
