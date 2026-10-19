# `claude_bedrock_theme_frequency_77.py`: 77 Grief Themes (Frequency)

## What it does
Uses the same **quote-counting method** as [theme_frequency](theme_frequency.md), but with a
larger list of **77 bereavement and grief themes**. For each entry, the model lists every
separate quote for each theme, and `count = len(list)`.

Most themes will have zero or very few items for any one entry. That is expected, and the model is
told not to force matches.

## The 77 themes, grouped
(Full names and one-line definitions are in the `CONSTRUCTS` dict in the script.)

- **Core grief:** prolonged yearning, non-acceptance of death, continuing bonds, linking objects,
  reminder-triggered grief, counterfactual rumination, guilt/self-blame
- **Relationship with the deceased:** ambivalence, idealization, identification, dependency or
  enmeshment, loss of confidant, search for a substitute attachment, seeking their approval,
  unfinished business, unanswered questions, value discordance
- **Anger and blame:** anger/betrayal, grievance with medical providers
- **Emotional and physical symptoms:** depression, anxiety, sleep/dreams, somatic distress,
  numbing, crying/lability, intrusive death-scene memories, distress over their suffering
- **Risk:** passive death wishes or suicidal ideation, personal mortality salience, shame and
  self-devaluation, self-neglect
- **Spiritual and meaning:** religious coping, existential struggle, religious disillusionment,
  meaning-making, legacy and memorialization
- **Coping:** reminiscence, gratitude, avoidance/distraction, wish to detach, restoration-oriented
  coping, nature/exercise, help-seeking, substance use or recovery
- **Time and occasions:** anniversaries and holidays, milestones, lost shared future, their unlived
  future, untimely death, acceptance or relief, fear of forgetting
- **Social and family:** loneliness, family conflict, emotionally inhibited family of origin, abuse
  history, relationship strain, social support, disenfranchised grief, concern for other survivors,
  geographic separation
- **Roles and life stress:** caregiver burden, parental role, identity/role loss, work/school
  stress, perfectionism, financial strain, estate and household burden, relocation, own illness,
  cumulative losses
- **Other:** companion animals, longing for physical touch, food/cooking rituals, the deceased's
  death being substance-related, pandemic stressors, sociopolitical demoralization, comments on
  the journaling task itself

## Settings
| Constant | Default |
|---|---|
| `INPUT_FILE` | `Non_phi_data_with_risk.csv` |
| `OUTPUT_FILE` | `claude_bedrock_theme_frequency_77themes_10test.csv` |
| `MODEL_ID` | `us.anthropic.claude-sonnet-4-6` |
| `MAX_TOKENS` | `20000` |

## Output CSV columns
- `REDCap_ID`, `At_Risk`
- For each of the 77 themes: `<theme>_count` and `<theme>_instances` (JSON list)

Prompt log: `claude_prompt_log_theme_frequency_77.jsonl`.

## Notes
- The console shows only the **non-zero** counts for each participant.
- If many replies fail with `JSON parse failed`, the JSON for 77 themes may be getting cut off.
  Raise `MAX_TOKENS`.
- A row counts as finished only if all 77 counts are filled.
