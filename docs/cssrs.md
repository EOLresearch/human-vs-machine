# `claude_bedrock_cssrs.py`: Suicide-Risk Screening (C-SSRS)

## What it does
Reads each participant's journal text and gives it a **severity score from 0 to 6** based on the
**Columbia-Suicide Severity Rating Scale (C-SSRS)**. It then turns that score into a risk level.
If the input has ground-truth labels, it also measures how accurate the model is. Every run is
logged to **MLflow**.

> Research screening aid only. A HIGH flag must be reviewed under the study's safety protocol.

## Severity scale
| Level | Meaning |
|---|---|
| 0 | Not related to suicidal ideation |
| 1 | Wish to be dead (passive) |
| 2 | Non-specific active thoughts of killing oneself |
| 3 | Active ideation **with a method** |
| 4 | Active ideation **with some intent** to act |
| 5 | Active ideation **with a specific plan and intent** |
| 6 | Suicidal **behaviour** (attempt, started, or preparing) |

**Risk mapping** (`severity_to_risk`): `4–6 → HIGH`, `3 → MODERATE`, `1–2 → LOW`, `0 → NONE`.
A failed call is `ERROR`. **ERROR is not the same as NONE.**

## How the prompt works
- **System prompt:** the model acts as a clinical psychologist. A single clear statement is
  enough; it does not need to be repeated. Calm, matter-of-fact wording counts as much as
  distressed wording. If the model is unsure, it picks the **higher** level, because missing risk
  is worse than over-flagging.
- **Few-shot examples:** 14 short labelled examples, two per level from 0 to 6, used to
  calibrate the model.
- **Output:** JSON with reasoning for `Q1`…`Q6` (or `"N/A"`) and one `severity` integer. The
  model is told to choose the **highest** level the text supports.

## Settings
| Constant | Default |
|---|---|
| `INPUT_FILE` | `Non_phi_data_with_risk.csv` |
| `OUTPUT_FILE` | `claude_bedrock_results_cssrs_10test.csv` |
| `MODEL_ID` | `us.anthropic.claude-sonnet-4-6` |
| `MAX_TOKENS` | `8000` |
| `PROMPT_VERSION` | `v4_cssrs_severity_fewshot`. Change it whenever you change the prompt. |
| `PILOT_N` | `None` runs everyone. Set a number to run only the first N. |

## Output CSV columns
| Column | Meaning |
|---|---|
| `REDCap_ID` | Participant |
| `severity` | 0–6 (empty if the call failed) |
| `risk` | HIGH / MODERATE / LOW / NONE / ERROR |
| `items_present` | Names of the C-SSRS items the model found evidence for |
| `Q1_reason` … `Q6_reason` | The model's reasoning for each item |

Prompt log: `claude_prompt_log_cssrs.jsonl`.

## Metrics and MLflow
- Experiment name: `CSSRS_Risk_Screening`. Stored in `mlflow.db` next to the script.
- Logged: model, prompt version, hashes of the prompt text, the prompts as artifacts, the
  output CSV, and the prompt log.
- If `At_Risk` is in the input, predicted at-risk = `severity > 0`. It logs **sensitivity,
  specificity, F1, MCC, Cohen's kappa, and AUC**.
- If there is a `GT_high_risk` column instead, predicted = `risk == HIGH`.
- To view the results: `mlflow ui` → http://localhost:5000

## Notes
- Prompt artifacts are written to `/tmp/cssrs_*.txt` before they are logged to MLflow.
- The input CSV must already contain **one row per participant**. The script does not group rows.
