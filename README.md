# Calibrated Three-Way Gating of Multimodal Auto-Insurance Claims with a Closed-Set Decision Model

Code, cached model outputs and the manuscript for a DSC 501 (SRM AP University) course project. The project routes motor-insurance claims to **Auto-Approve**, **Reject/Fraud-Audit** or **Escalate-Human** using three inputs: a damage photograph, a categorical claim description and eleven numerical fields.

- **Router.** One request to Google's Gemini Decision 1 asks a routing question and four yes/no checks.
- **Combiner.** A 24-parameter logistic combiner turns the answers into route probabilities.
- **Gate.** A cost-aware confidence gate escalates claims it is unsure about.

The manuscript is in `manuscript/` (Springer Nature LaTeX template), and the compiled paper is `manuscript/main.pdf`.

Repository: https://github.com/rajeshkadiyalaaa/Claim_detection_Mulimodel

## Main results (constructed 1,000-claim benchmark)

| | Proposed router | Trained early fusion (B4) |
|---|---|---|
| Test accuracy (n = 150, frozen configuration) | 0.853 | 0.913 |
| Cross-validated accuracy (5×5 folds) | 0.866 | 0.908 |
| Accuracy with 25 labelled claims | 0.825 | 0.609 |
| Median latency per decision | 0.80 s | local |

- Against a generative model given the same questions, Decision 1 is about 5–8× faster and 6.5–18× cheaper.
- The cost gain over escalating every claim holds only when correct audits and escalations are free (misrouting cost).
- The benchmark's labels follow documented construction rules, which every method partly exploits. The paper reports shortcut controls for all methods.

## Repository layout

```
manuscript/                    LaTeX sources, figures and compiled PDF
  sections/                    abstract, introduction, literature review, methodology, results, conclusion
  figures/                     architecture.drawio/.png, fig_learning_curve, fig_confusion
experiments/                   all code
  decision1.py                 score claims with Gemini Decision 1 (API)
  decision1_v2.py              post-hoc revised request (v2) (API)
  b5_generative.py, b5_fair.py generative baselines (API)
  stability_retest.py          re-send 100 test claims (API)
  degradation_control.py       degrade clear photos and re-score (API)
  clip_embed.py                CLIP ViT-B/32 image embeddings
  baselines.py                 B1–B4 supervised baselines
  pilot_analysis.py, combiner.py, val_comparison.py, v2_eval.py   validation-stage analyses
  final_test.py, v2_test.py    one-time test evaluations of the frozen configurations
  revision_stats.py            McNemar, ECE, cross-validation, learning curves, shortcut controls
  round2_stats.py, handling_cost_test.py, learning_curve_regularised.py   corrected CV tests, handling cost, rule-aware learners
  make_figures.py, make_round2_figures.py   paper figures
  FROZEN_CONFIG.md, FROZEN_CONFIG_v2.md     configurations frozen before the test runs
  requirements-lock.txt        full pinned environment
  results/                     parsed model outputs (*.jsonl) and analysis results (*.json)
data/README.md                 benchmark field documentation
dataset_creation_walkthrough.ipynb   how the benchmark records were built
```

## Reproducing the results

1. **Environment** (Python 3.12):
   ```bash
   python3.12 -m venv .venv && source .venv/bin/activate
   pip install -r experiments/requirements-lock.txt
   ```
2. **Data (not included).** The benchmark pairs photographs from CarDD (Wang et al., 2023) with records from the Kaggle *Auto Insurance Claims* data set. The CarDD licence forbids redistribution, and the Kaggle licence was not verified, so neither the images nor the records are in this repository.
   - Obtain both sources under their own terms.
   - Place the records at `data/multimodal_claims_benchmark.jsonl` and the photographs in `data/benchmark_images/`.
   - The field layout is in `data/README.md`, and the construction steps are in `dataset_creation_walkthrough.ipynb`.
3. **Analyses without API calls.** Every model output used in the paper is cached in `experiments/results/`, so the statistics and figures can be recomputed without a Gemini key. Some scripts also need the benchmark records: `revision_stats.py` and `round2_stats.py` need them for the labels and the text and image features.
   ```bash
   cd experiments
   python final_test.py            # frozen test evaluation  -> results/final_test.json
   python revision_stats.py        # CV, learning curves, controls -> results/revision_stats.json
   python round2_stats.py          # corrected CV tests, handling cost, rule-aware learners -> results/round2_stats.json
   python handling_cost_test.py
   python make_figures.py && python make_round2_figures.py
   ```
4. **Re-querying the models (optional, billed).**
   - Put a Gemini API key in `geminiapi.txt` at the repository root. The file is git-ignored; never commit it.
   - Then run, for example, `python decision1.py --split test`.
   - Raw responses are cached under `experiments/cache/`, which is also git-ignored.
   - Prices on 24 Nov 2026: US$0.10 per million input tokens for Decision 1.

5. **Manuscript.** Build it with `cd manuscript && tectonic main.tex`, or with any LaTeX distribution: run `pdflatex`, then `bibtex`, then `pdflatex` twice.

## Notes

- All random seeds are 42.
- The main configuration was frozen (`experiments/FROZEN_CONFIG.md`) before any test claim was scored. Every analysis added after that run is reported in the paper as exploratory.
- The labels are construction labels, not real claim outcomes. Results describe routing under the benchmark's rules.

## Authors

Kadiyala Rajesh, Matta Vikas, Mudigonda Krishna Siva Prasad (SRM AP University, Amaravati).
