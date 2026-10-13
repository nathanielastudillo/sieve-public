# Corpora

Every versioned corpus built under `data/corpus_v*/`. All tokens are
uint16 memmaps. `metadata.json` in each dir is authoritative — this page
is the running index.

## Summary table

| Version | Date built | Sources | Total tokens | Train / Val / Test | Tokenizer | Purpose |
|---------|-----------|---------|-------------:|--------------------:|-----------|---------|
| v001    | (pre-session) | ?  | ~3.05M   | —                   | tiktoken GPT-2 | initial pipeline test |
| v002    | (pre-session) | ?  | ~3.01M   | —                   | custom 16K BPE v002 | custom-tokenizer test |
| v003    | (pre-session) | 16 | 99,744   | 84,992 / 9,728 / 4,608 | v002 | 100K subsample for smoke loop |
| **v004** | **2026-04-17** | **37** | **7,524,509** | **6,396,416 / 752,128 / 375,808** | **v002** | **Full expanded pool after OCR/epub/re-extract work** |
| v005    | 2026-04-17 | 35 | 97,078 | 82,944 / 9,216 / 4,608 | v002 | 100K distilled from v004 |
| v006    | 2026-04-17 | 35 | 97,018 | 82,944 / 9,216 / 4,608 | v003 (16K) | 100K distilled, v003 BPE |
| **v007** | **2026-04-17** | **35** | **94,777** | **80,896 / 9,216 / 4,608** | **v004 (4K)** | **4K-vocab variant for Magikarp ablation** |
| **v008** | **2026-04-17** | **35** | **95,749** | **81,920 / 9,216 / 4,608** | **v005 (8K)** | **8K-vocab variant for Magikarp ablation** |

## v003 — narrow 16-source baseline (pre-session)

- Built via `data/build_corpus_subsample.py` with `--target-tokens 100000 --stratification sqrt`
- Seed=42, val_fraction=0.10, test_fraction=0.05, dedup_threshold=0.7
- 16 sources, but several were heavily under-extracted by pymupdf4llm
  (Fink Clinical Intro: 157 words, Mann Vol 1: 11K words, Laclau/Mouffe: 64 words)
- Density heuristic saturated at 1.0 for every source (known limitation)
- Used as training data for **experiments/001-smoke-v003.md**

## v004 — full expanded pool (2026-04-17)

- Built via `data/build_corpus.py` (no subsampling), vocab still v002 BPE
- 37 cleaned sources totalling ~5.16M words / 33.2M chars raw
- MinHash dedup removed 2,205 / 63,134 paragraphs (3.5%)
- Tokenized to **7,524,509 tokens** — 75× larger pool than v003
- Purpose: the raw expanded pool that v005 draws from, and that a future
  v006 tokenizer will be trained on

## v005 — 100K distilled from v004 (2026-04-17)

- Built via `data/build_corpus_subsample.py --target-tokens 100000`
- Same BPE (v002), same seed (42), same stratification (sqrt)
- 35 sources contributed (Collins IRC dropped for being image-only,
  Seminar XX Encore dropped on density threshold)
- Final: 97,078 tokens — 1,000 short of target due to MinHash dedup (6/697 paragraphs removed after assembly)
- Used as training data for **experiments/002-smoke-v005.md**

## Source expansion work (2026-04-17)

Between v003 and v004, the extracted/ pool grew from 21 → 38 sources via:

1. **EPUBs from corpus-host** (10 new, clean text, no OCR needed):
   Anti-Oedipus, A Thousand Plateaus, Difference & Repetition,
   Lines of Flight, Schuster *Trouble with Pleasure*,
   Zizek *Mapping Ideology*, Mitchell *Freud and Beyond*,
   Leader *Introducing Lacan*, Goodchild *Deleuze and Guattari*,
   Ansell-Pearson *Deleuze and Philosophy*, DeLanda *New Philosophy of Society*
2. **New text PDFs from corpus-host** (6 clean): Fink *Lacan on Love*,
   Latour *Politics of Nature* + *Thinking with Bruno Latour in Rhetoric*,
   Kristeva *Powers of Horror*, Lacan Seminar XVII (via Clemens), Fink *Lacanian Subject*
3. **Raw-pymupdf re-extraction** (7 fixed): pymupdf4llm's markdown converter
   had been silently dropping 95%+ of text on multi-column / unusual-layout
   PDFs. Raw `page.get_text()` recovered full content:
    - Fink Clinical Intro: 2.4K → 955K chars (390×)
    - Lacan Seminar XI (Four Fundamental Concepts): 3.6K → 620K chars (172×)
    - Fink Lacan to the Letter: 4.2K → 510K chars (121×)
    - Fink Lacanian Subject: 15K → 562K chars (37×)
    - Laclau/Mouffe Hegemony: 456 → 591K chars (1300×)
    - Mann Vol 1: 76K → 1.76M chars (23×)
    - Mann Vol 2: 33K → 2.29M chars (69×)

Net: extracted/ went from 11.5M → 35.3M chars.

## Known gaps / deferred work

- **Collins *Interaction Ritual Chains*** — image-only PDF, 0 bytes extractable
  from both pymupdf4llm and raw pymupdf; needs OCR (tesseract/ocrmypdf
  not currently installed on gpu-01)
- **Lacan Seminar XX *Encore*** — same; 0 bytes raw, 8K from pymupdf4llm
  (page-number garbage, correctly dropped by clean.py density filter)
- **Density heuristic saturation** — every source shows
  `min_density_kept=1.0, mean_density_kept=1.0`. The 3-gram density score
  is hitting ceiling because every unique 3-gram in a ≤100-paragraph subset
  is unique. Heuristic is not discriminating anything at this scale.
- **No within-book dedup across versions** — the tiny pymupdf4llm
  DeLanda extract (1043 words) still lives in extracted/ alongside the
  clean epub version (364K chars). MinHash dedup in build_corpus handles
  paragraph-level overlap but not file-level. Minor noise.
