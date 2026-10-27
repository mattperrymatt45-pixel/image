# Text–Image Interaction & Evaluation Lab

A complete, self-contained classroom demo for a **Multimodal Generative AI Laboratory**
experiment. Everything lives in one file: `index.html` (HTML + CSS + vanilla JavaScript).

## Run it

1. Download or clone this repository.
2. Open `index.html` in any modern browser (double-click it, or drag it into a tab).

That's it — no Python, Node.js, build step, installation, API key, internet connection or
external service is required. The page makes **zero network requests**; the eight gallery
images are SVG illustrations embedded in the file as `data:` URIs.

## What it does

- Text prompt box with 8 preset prompts, a random-preset button and `Ctrl+Enter` shortcut.
- Gallery of 8 clearly labelled **demonstration images** (each card says
  "Demo image · not AI-generated").
- A deterministic, fully visible, rule-based semantic relevance score for every image, with
  a per-image "Why this score?" breakdown (matched / missing concepts and point contributions).
- Selectable A/B comparison panels and a full comparison table.
- Evaluation dashboard: summary statistics, a bar chart, a score-distribution histogram,
  automatically generated observations and an experiment conclusion.
- One-click **CSV export** of all scores and sub-scores (generated locally with a `Blob`).
- Print / Save PDF, dark mode and reset controls.
- A written-up experiment report: aim, theory, algorithm (pseudocode), methodology, live
  results and limitations.

## The scoring rule (short version)

Words in the prompt are normalised and mapped through a small, visible synonym lexicon into
three categories (`object`, `attribute`, `scene`). Each bundled image has a hand-written list
of concepts in those same categories. For each category present in the prompt we compute
recall = matched ÷ requested, then combine them with weights 45 / 30 / 25, renormalised to
100 when a category is absent:

```
score = round( Σ (weight[c] × recall[c]) / Σ weight[c] × 100 )
```

Vocabulary coverage (recognised words ÷ meaningful words) is reported separately. If no word
is recognised, a clearly labelled literal-overlap fallback is used and capped at 40. Verdict
bands: 75–100 strong, 50–74 partial, 25–49 weak, 0–24 no meaningful match.

## Important disclaimer

**No diffusion model and no CLIP / neural inference is performed here.** The images are fixed,
hand-drawn SVG placeholders — the prompt never changes them. The score is a transparent
keyword-overlap proxy designed to teach evaluation methodology, not to measure the quality of
any generative model. See the "Limitations" section in the app for the full list of caveats.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | The entire application (single file, no dependencies). |
| `README.md` | These instructions. |
