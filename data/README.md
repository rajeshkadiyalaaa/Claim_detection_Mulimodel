# Tri-Modal Insurance Claim Gating Benchmark

This dataset contains 1,000 synthetic claim–photo pairs for research on insurance claim routing. Each sample combines a vehicle-damage photo from CarDD with a categorical description and numerical features from a unique record in Auto Insurance Claims Data. The claim and photo come from independent sources. Labels describe how each pair was constructed.

## Files

| File | Contents |
| --- | --- |
| `data/multimodal_claims_benchmark.csv` | Benchmark records in CSV format |
| `data/multimodal_claims_benchmark.jsonl` | The same records, one JSON object per line |
| `data/benchmark_images/` | The 1,000 JPEGs referenced by the records |

Resolve each `image_path` relative to the repository root. The records and images are not included in the repository (see the main README).

## Modalities

- **Image** uses the JPEG referenced by `image_path`.
- **Text** uses `claim_description` for incident type, collision type, reported severity, vehicle make and model, and police-report status. Missing categorical values are written as `unrecorded`.
- **Numerical** uses the 11 numeric fields below, taken directly from the matched source claim. Values retain their source units. All 11 fields are complete across the 1,000 records.

Use `claim_text_summary` for compatibility with earlier experiments. It repeats some numerical values. For a tri-modal evaluation, use `claim_description` as the text input and pass the numerical features separately.

## Fields

Types below describe the JSONL values and the types to use when loading the CSV. The CSV writes annual premiums with two decimal places.

| Field | Type | Description |
| --- | --- | --- |
| `sample_id` | String | Unique sample identifier |
| `image_path` | String | Path to the paired JPEG |
| `claim_text_summary` | String | Original summary with claim amounts and injury count |
| `incident_severity` | String | Severity recorded on the original claim |
| `total_claim_amount` | Integer | Total amount recorded on the original claim |
| `decision_label` | String | Target routing class |
| `pairing_reason` | String | Explanation of the constructed match, mismatch, or escalation |
| `split` | String | Assigned partition, one of `train`, `val`, or `test` |
| `claim_description` | String | Categorical incident and vehicle description used as the text input |
| `vehicle_claim` | Integer | Amount claimed for vehicle damage |
| `injury_claim` | Integer | Amount claimed for injuries |
| `property_claim` | Integer | Amount claimed for property damage |
| `policy_deductable` | Integer | Policy deductible amount using the source column spelling |
| `policy_annual_premium` | Float | Annual policy premium amount |
| `age` | Integer | Policyholder age in years |
| `months_as_customer` | Integer | Time as a customer in months |
| `auto_year` | Integer | Vehicle model year |
| `bodily_injuries` | Integer | Reported count of bodily injuries |
| `witnesses` | Integer | Reported witness count |

## Labels and splits

- **Auto-Approve** pairs an eligible non-fraud claim with a clear photo whose estimated visible damage is compatible with the reported severity.
- **Reject_Fraud_Audit** pairs a fraud-flagged or severe claim with a photo that shows a different level of visible damage.
- **Escalate_Human** pairs an ambiguous or incomplete claim with a photo degraded through Gaussian blur, reduced brightness, or cropping.

| Split | Auto-Approve | Reject_Fraud_Audit | Escalate_Human | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 189 | 291 | 220 | 700 |
| Validation | 41 | 62 | 47 | 150 |
| Test | 41 | 62 | 47 | 150 |
| **Total** | **271** | **415** | **314** | **1,000** |

CarDD labels damage types and bounding boxes. The benchmark estimates visible damage extent from those annotations. The estimate is sensitive to camera framing. The benchmark evaluates synthetic routing decisions, and text alone can provide strong label cues. For model evaluation, use the JPEG, `claim_description`, and the numerical fields as inputs. Use `decision_label` as the target and follow the supplied split. Keep `pairing_reason` out of the inputs because it records how the label was assigned.

## Sources

- [Auto Insurance Claims Data](https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data)
- [Car Damage Detection dataset](https://cardd-ustc.github.io/) · [CarDD license](https://cardd-ustc.github.io/docs/CarDD_license.pdf)
