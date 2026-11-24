# Frozen configuration: v2 (post-hoc revision)

**Frozen on 24 Nov 2026, before any test claim was sent with the v2 request.**

**Status:** post-hoc. v2 was designed after the v1 test run, from the error analysis summarised in the paper (Section 4.6). v1 (`FROZEN_CONFIG.md`) remains the pre-registered main result. All v2 choices below used train and val only.

## Request
- Model: `gemini-decision-1`, via the Decisions REST API `v1beta`.
- Inputs: identical to v1 (photo at ≤ 1,024 px with medium resolution, `claim_description`, 11 labelled numeric fields).
- The instruction text states the handling policy.
- Seven questions are asked: q1 route, damage match, photo clear, police-report recorded, theft, vehicle match, amounts match.
- Exact wording: `experiments/decision1_v2.py`. The same request is used for every split.

## Combiner
- Multinomial logistic regression on (p_A, p_R, p_E, damage match, photo clear, police recorded, theft, amounts match). The **vehicle-match** answer is excluded.
- This feature set had the lowest validation log loss among the four candidates:

  | Candidate | Val log loss |
  |---|---:|
  | All answers | 0.307 |
  | **Without vehicle** | **0.297** |
  | Without amounts | 0.309 |
  | Without vehicle and amounts | 0.299 |

- Trained on the 700 training claims. C is chosen by validation log loss (C = 10).

## Gate
- Confidence gate: automate ŷ ∈ {A, R} only if c ≥ τ_c; otherwise escalate.
- τ_c is chosen on val with cost weights 10 / 4 / 1 (τ_c = 0.85). There is no support term, since τ_s = 0 was selected in v1.

## Reporting
- Test metrics and bootstrap CIs are computed exactly as in `final_test.py`.
- Results are labelled **post-hoc** in the paper.
