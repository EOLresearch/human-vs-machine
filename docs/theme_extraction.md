# `claude_bedrock_theme_extraction.py`: Construct Ratings (0–3)

## What it does
The model rates each journal entry on **19 psychological constructs**, using a 0–3 scale with
fixed definitions for each point. It gives a short evidence quote for every rating above 0. The
result is a **feature vector**, meant as input to a downstream classifier (e.g. L1 logistic
regression). It is not a risk verdict by itself.

## Rating scale
| Rating | Meaning |
|---|---|
| 0 | Absent: no evidence |
| 1 | Hinted: vague or indirect |
| 2 | Present: clearly stated at least once |
| 3 | Dominant: a central, recurring theme |

The model must rate each construct separately, so one rating doesn't influence another.

## The 19 constructs
`negative_emotion`, `positive_emotion`, `hopelessness` (Beck), `absolutist_thinking`,
`death_suicide_ideation`, `burdensomeness` and `belongingness_absent` (Joiner), `entrapment` and
`defeat` (O'Connor IMV model), `psychache` (Shneidman), `reunification_longing`,
`identity_collapse`, `self_blame`, `agency_helplessness`, `protective_factors`,
`cognitive_rumination`, `social_withdrawal`, `caregiver_role_loss`, `meaning_loss`.

The one-sentence definition of each construct is in the `CONSTRUCTS` dict in the script.

## Settings
| Constant | Default |
|---|---|
| `INPUT_FILE` | `Non_phi_data_with_risk.csv` |
| `OUTPUT_FILE` | `claude_bedrock_theme_ratings_10test.csv` |
| `MODEL_ID` | `us.anthropic.claude-sonnet-4-6` |
| `MAX_TOKENS` | `8000` |

## Output CSV columns
- `REDCap_ID`, `At_Risk` (copied from the input)
- For each construct: `<construct>_rating` (0–3) and `<construct>_evidence` (quote or paraphrase)

Prompt log: `claude_prompt_log_theme_extraction.jsonl`.

## Notes
- A row counts as finished only if **all** 19 ratings are filled. Incomplete rows are retried
  on the next run.
- A 0–3 rating is a judgment call and can vary between runs. For more reproducible features, use
  [theme_frequency.md](theme_frequency.md), which counts quotes instead.
