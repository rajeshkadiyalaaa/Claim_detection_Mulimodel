# Frozen configuration for the one-time test evaluation

**Frozen on 24 Nov 2026, before any test-split claim was sent to any model.** Every choice below was made using train and val only.

## Main method: tri-modal claim gate
- **Request.** One Decision 1 request per claim (`gemini-decision-1`, Decisions REST API `v1beta`). It contains:
  - the instruction text;
  - the photo as JPEG, longer side ≤ 1,024 px, `resolution: medium`;
  - the `claim_description`;
  - 11 labelled numeric fields.
- **Questions.** q1 routing (choice) and q2–q5 checks (yes/no). Exact wording is in `experiments/decision1.py`.
- **Combiner.** Multinomial logistic regression on (p_A, p_R, p_E, s2, s3, s4, s5).
  - Trained on the 700 training claims.
  - C is chosen from {0.01, 0.1, 1, 10, 100} by validation log loss (C = 100).
- **Gate.**
  - Rule: automate the route ŷ ∈ {A, R} only if c ≥ τ_c and σ(ŷ) ≥ τ_s; otherwise escalate.
  - (τ_c, τ_s) are chosen on validation over the grid {0, 0.05, …, 1}², minimising mean cost with weights (wrong approval 10, wrong audit 4, unneeded review 1). Ties go to higher coverage.
  - Validation selection: τ_c = 0.85, τ_s = 0.
- **Seed:** 42.

## Reported variant: image + text (no numbers in the request)
This is the same pipeline with the numeric block removed from the request. Its combiner is refit on the training split, and its C and thresholds are chosen on validation.

## Baselines
Each baseline uses the same gate. Its confidence threshold is chosen on validation and τ_s = 0.

| ID | Inputs | Model | Selection on validation |
|---|---|---|---|
| B1 | text | TF-IDF (1–2 grams) + logistic regression | C by log loss |
| B2 | numbers | XGBoost | depth and number of trees by log loss |
| B3 | image | CLIP ViT-B/32 embedding + logistic regression | C by log loss |
| B4 | all three (early fusion) | XGBoost on [CLIP \| SVD-64 of TF-IDF \| standardised numbers] | depth and number of trees by log loss |
| B5 | all three | `gemini-3.8-flash`, zero-shot, JSON route + confidence, temperature 0 | none |

## Ablations reported on test
- Question ablations: q1 only without the combiner; combiner on q1 only; combiner on checks only; dropping each of s2–s5.
- Modality ablations: no image, no text, no numbers.

## Metrics
- Accuracy, macro-F1, per-class recall, ECE (10 bins, top-label), Brier score.
- Ungated and gated mean cost, coverage, error rate on automated decisions, wrong approvals.
- Median latency from sequential calls, and US$ per 1,000 claims at the listed prices.
- 95% CIs from 10,000 paired bootstrap resamples of test claims.
- Cost sensitivity: re-select thresholds on validation for other cost weights.

Test results will not be used to change any of the choices above.
