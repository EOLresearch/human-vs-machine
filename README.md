# Bedrock Journal Analysis

Four Python scripts that read grief (bereavement) journal entries and use Claude on **AWS Bedrock**
to produce structured research data: a suicide-risk severity score, and ratings or counts of
psychological themes.

> **Research use only.** These are screening and annotation aids, not diagnostic tools. Any HIGH
> risk flag must go through the study's safety protocol and be reviewed by qualified clinical staff.

- How to run (saml2aws login → results): **[USAGE.md](USAGE.md)**
- Detailed description of each script: **[docs/](docs/)**

---

## The scripts

| Script | What it produces | Details |
|---|---|---|
| `claude_bedrock_cssrs.py` | Suicide-risk severity 0–6 (C-SSRS), a risk level, and accuracy metrics in MLflow | [docs/cssrs.md](docs/cssrs.md) |
| `claude_bedrock_theme_extraction.py` | A 0–3 rating for each of 19 psychological constructs | [docs/theme_extraction.md](docs/theme_extraction.md) |
| `claude_bedrock_theme_frequency.py` | The same 19 constructs, as a **count** of quoted instances | [docs/theme_frequency.md](docs/theme_frequency.md) |
| `claude_bedrock_theme_frequency_77.py` | A count of quoted instances for each of 77 grief themes | [docs/theme_frequency_77.md](docs/theme_frequency_77.md) |

**Which one should I use?**
- To screen for suicide risk, use **cssrs**.
- To get features for a classifier: **theme_frequency** is more reproducible because the count
  comes from extracted quotes. **theme_extraction** is the older 0–3 rating version.
- For a broad grief-specific profile, use **theme_frequency_77**.

---

## Files

```
├── claude_bedrock_cssrs.py
├── claude_bedrock_theme_extraction.py
├── claude_bedrock_theme_frequency.py
├── claude_bedrock_theme_frequency_77.py
├── docs/                 ← one description file per script
├── requirements.txt
├── README.md             ← this file
├── USAGE.md              ← quick how-to run
└── .gitignore            ← keeps data, logs and secrets out of git
```

**Data is not in this repo.** The input CSVs contain participant journal text and are excluded by
`.gitignore`. Get them from the study's secure storage.

---

## How every script works

```
input CSV ──► one row per participant (REDCap_ID + Entry_masked text)
          ──► skip participants already in the output CSV (resume)
          ──► build prompt = SYSTEM_PROMPT + USER_PROMPT_TEMPLATE + journal text
          ──► bedrock.converse(...)  →  Claude returns JSON
          ──► parse JSON (tolerates ```json fences and extra text)
          ──► append one row to the output CSV  +  log full prompt/response to a .jsonl file
```

| Shared behaviour | Detail |
|---|---|
| Input columns | `REDCap_ID` (or `REDCap ID`, renamed automatically), `Entry_masked` (de-identified text), optional `At_Risk` (0/1 ground truth) |
| AWS auth | `~/.aws/saml2aws_credentials`, profile **`wcm`**, region `us-east-1` |
| Model | `us.anthropic.claude-sonnet-4-6` (set by `MODEL_ID`) |
| Temperature | `0.01`, close to deterministic |
| Text limit | `MAX_JOURNAL_CHARS = 100000`. Longer entries are cut off. The longest observed entry is about 80k chars. |
| Resume | Results are saved after each participant. If you stop with Ctrl+C and re-run, it continues where it left off. Failed rows are retried. |
| Audit log | `claude_prompt_log_*.jsonl` stores the full prompt and raw response for every call. **It contains journal text, so treat it as sensitive.** |
| Settings | CAPITALISED constants at the top of each file. There are no command-line arguments. |

---

## Notes for whoever takes this over

- **Start fresh:** delete or rename the output CSV. Otherwise the script resumes from it.
- **Change the model:** edit `MODEL_ID`. Your AWS role needs access to that model in Bedrock.
- **Change the prompt:** edit `SYSTEM_PROMPT` / `USER_PROMPT_TEMPLATE`. For cssrs, also update
  `PROMPT_VERSION` so MLflow runs can be told apart.
- **Change constructs:** edit the `CONSTRUCTS` dict. Output columns are created from it
  automatically. Do not resume into an old output CSV after changing it.
- **Input file names differ** between scripts. Check `INPUT_FILE` (see the table in USAGE.md).
