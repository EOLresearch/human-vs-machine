# `claude_bedrock_theme_frequency.py`: Construct Frequency (19 constructs)

## What it does
Uses the **same 19 constructs** as [theme_extraction](theme_extraction.md). Instead of a 0–3
rating, the model lists **every separate quote** where each construct appears. The script then
counts them itself: `count = len(list)`.

**Why:** in the pilot, the 0–3 ratings varied between runs. A count based on actual quotes is
more reproducible and easier to check.

## Counting rules given to the model
- Each item must be a **separate** expression of the construct. A sentence that expresses it
  once is one item, however strong it is. Don't list the same passage twice, and don't split one
  statement into several items.
- Only include what is clearly stated in this text, and don't reuse evidence across constructs.
- If a construct doesn't appear, its list is empty.
- Re-read the full text for each construct (be exhaustive).

## The 19 constructs
`negative_emotion`, `positive_emotion`, `hopelessness`, `absolutist_thinking`,
`death_suicide_ideation`, `burdensomeness`, `belongingness_absent`, `entrapment`, `defeat`,
`psychache`, `reunification_longing`, `identity_collapse`, `self_blame`, `agency_helplessness`,
`protective_factors`, `cognitive_rumination`, `social_withdrawal`, `caregiver_role_loss`,
`meaning_loss`. (Definitions are in `CONSTRUCTS` in the script.)

## Settings
| Constant | Default |
|---|---|
| `INPUT_FILE` | `non_phi_risk_assessment.csv` |
| `OUTPUT_FILE` | `claude_bedrock_theme_frequency_full111.csv` (the full dataset of 111 participants) |
| `MODEL_ID` | `us.anthropic.claude-sonnet-4-6` |
| `MAX_TOKENS` | `32000`. The lists can be long. |

## Output CSV columns
- `REDCap_ID`, `At_Risk`
- For each construct: `<construct>_count` (integer) and `<construct>_instances` (JSON list of quotes)

Prompt log: `claude_prompt_log_theme_frequency.jsonl`.

## Notes
- **JSON parsing:** the model sometimes outputs a JSON block, notices a mistake, and outputs a
  corrected block. `extract_json_block` uses the **last** fenced JSON block that parses.
- A row counts as finished only if all 19 counts are filled.
