import os
import re
import json
import hashlib
import logging
import warnings
import boto3
import mlflow
import pandas as pd
from datetime import datetime
from botocore.exceptions import ClientError
from sklearn.metrics import (
    f1_score, matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, confusion_matrix
)

os.environ["AWS_SHARED_CREDENTIALS_FILE"] = os.path.expanduser("~/.aws/saml2aws_credentials")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
# SAFETY NOTE: this script screens journal text for suicide ideation/behavior
# using the Columbia-Suicide Severity Rating Scale (C-SSRS). It is a research
# screening aid, not a diagnostic instrument or a substitute for clinical
# judgment. Any HIGH risk flag this script produces should be routed through
# your study's actual safety/reporting protocol and reviewed by qualified
# clinical staff — do not rely on this output alone for any real-world risk
# decision.
INPUT_FILE        = "Non_phi_data_with_risk.csv"
OUTPUT_FILE       = "claude_bedrock_results_cssrs_10test.csv"
# 20,000 silently truncated 21.4% of the full dataset (24/112 entries) —
# including 2 of the 3 at-risk cases this script missed in small-scale
# testing. 100,000 covers the observed max (79,938 chars) with headroom,
# and stays well inside the model's context window.
MAX_JOURNAL_CHARS = 100000
AWS_REGION        = "us-east-1"
MODEL_ID          = "us.anthropic.claude-sonnet-4-6"
TEMPERATURE       = 0.01
MAX_TOKENS        = 8000
PROMPT_VERSION    = "v4_cssrs_severity_fewshot"
PILOT_N           = None  # only 10 rows in this test file anyway
PROMPT_LOG_FILE   = "claude_prompt_log_cssrs.jsonl"

# Single 0-6 severity ladder: 1-5 mirror the C-SSRS ideation items, 6 is
# suicidal behavior. Labels are for reporting only — the model outputs Q1-Q6
# reasoning fields plus one "severity" integer.
Q_LABELS = {
    "Q1": "Wish to be dead / passive death wish",
    "Q2": "Nonspecific active suicidal ideation",
    "Q3": "Active ideation with a method",
    "Q4": "Active ideation with some intent to act",
    "Q5": "Active ideation with a specific plan and intent",
    "Q6": "Suicidal behavior",
}

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = (
    "You are a clinical psychologist screening journal text against the "
    "Columbia-Suicide Severity Rating Scale (C-SSRS). Severity levels 1-5 "
    "form an ideation ladder (1 = passive wish to be dead, 5 = active "
    "ideation with a specific plan and intent); level 6 is suicidal "
    "behavior.\n\n"
    "This scale works differently from ordinary symptom screening: do NOT "
    "require a theme to recur across entries before counting it as "
    "evidence for a level. A single, clear, explicit statement is "
    "sufficient on its own — 'I've thought about how I'd do it' stated "
    "once is just as significant as if it were repeated, and should be "
    "scored at the level that statement supports. Do not require distress "
    "language either — a matter-of-fact or calm description of these "
    "thoughts or actions counts just as much as a distressed one.\n\n"
    "When something is clearly and explicitly stated but you are uncertain "
    "whether it meets a given level's threshold, assign that level rather "
    "than the one below it — for a safety screener, under-detection is a "
    "far costlier error than over-detection. Vague or merely possible "
    "hints that are never clearly stated should still get no credit at "
    "that level."
)

# ══════════════════════════════════════════════════════════════════════════════
# FEW-SHOT EXAMPLES
# ══════════════════════════════════════════════════════════════════════════════
FEW_SHOT_EXAMPLES = """Reference examples (for calibration only — these are not part of the text you are evaluating):

[severity 0] "A friend of mine took his own life in prison four years ago... Would someone here be willing to call the prison for me and ask?" — About someone else's death, not the writer's own ideation. No evidence for any item.

[severity 0] "if u dont have a note about a list of things that help/work for you when you feel like crap, make one... i genuinely feel like this might save my life" — Describes a coping tool, not suicidal ideation.

[severity 1] "Why am I alive? ... There's no point to staying alive if ur getting this treatment" — A passive wish not to be alive; no active thought of killing themselves.

[severity 1] "I need reasons to live. I just feel so tired" — A passive wish not to continue living, stated once and clearly; that is sufficient on its own for severity 1.

[severity 2] "I've contemplated this for years... it would be a sin to kill myself" — Explicit, repeated thoughts of killing themselves, no method described.

[severity 2] "I even think of doing it on Christmas day" — Active thought of ending their life on a specific day, no method described.

[severity 3] "I can't stop imagining cutting my own wrists" — A specific method (cutting wrists) is described, with no stated intent to act on it.

[severity 3] "constantly plagued by... imagery of hanging myself out in the backyard" — A specific method and location described as recurring imagery, with no stated intent to act.

[severity 4] "All I want to do is douse myself in water and walk into the huge snowcovered forest... find a nice tree to sit under and wait for hypothermia to set in" — A specific method and immediate setting described as a present desire, without a firmly stated decision or date to carry it out.

[severity 4] "F22 think I'm gonna do it just a matter of when and where" — Clear intent to act, explicitly without a worked-out plan (when/where not yet decided).

[severity 5] "I'm going to hang myself later... with a belt when everyone goes to sleep" — Method, timing, and intent are all specified.

[severity 5] "I've set a date... for Friday. I know how to do it" — A specific date and known method, with intent to carry it out.

[severity 6] "I've chugged an entire bottle of smirnoff and am about to [cut] my wrist to die" — An attempt is already in progress.

[severity 6] "Gonna lie my head down on train-tracks soon. Thanks for everything, sorry. Bye." — A farewell message paired with a stated imminent method; writing a goodbye note is itself a preparatory act."""

# ══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════
USER_PROMPT_TEMPLATE = """Evaluate this text step by step and assign a severity score on a scale of 0–6,
where 0 means the text is not related to suicidal ideation, and 1–6 correspond
to increasing severity based on the Columbia-Suicide Severity Rating Scale (C-SSRS).

C-SSRS severity levels:
1. Wish to be dead: person has wished they were dead or wished they could go to
   sleep and not wake up. (Severity 1)
2. Nonspecific active suicidal thoughts: person has had actual thoughts of
   killing themselves. (Severity 2)
3. Active ideation with methods: person has thought about how they might kill
   themselves. (Severity 3)
4. Active ideation with intent: person has had these thoughts with some intention
   of acting on them. (Severity 4)
5. Active ideation with plan and intent: person has worked out or begun working
   out the details of how to kill themselves and intends to carry it out. (Severity 5)
6. Suicidal behavior: person has done, started to do, or prepared to do something
   to end their life. (Severity 6)

Assign the HIGHEST severity level supported by the text. Base your scoring only on
what is stated or clearly implied — do not infer beyond the text. If a level is not
supported, mark its reasoning "N/A".

Respond ONLY in this JSON format, with no other text:
{
  "Q1": "<reasoning or N/A>",
  "Q2": "<reasoning or N/A>",
  "Q3": "<reasoning or N/A>",
  "Q4": "<reasoning or N/A>",
  "Q5": "<reasoning or N/A>",
  "Q6": "<reasoning or N/A>",
  "severity": <integer 0-6>
}

{FEW_SHOT_EXAMPLES}

Text to evaluate:
\"\"\"
{INSERT_TEXT_HERE}
\"\"\"
"""

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def truncate_journal(text: str) -> str:
    return text[:MAX_JOURNAL_CHARS]

def clean_text(val) -> str | None:
    if val is None:
        return None
    return re.sub(r"[\n\r]+", " ", str(val)).strip()

def prompt_hash(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:8]

def parse_severity(val) -> int | None:
    """Returns an int 0-6, or None if val is missing/unparseable/out of range."""
    try:
        s = int(str(val).strip())
    except (TypeError, ValueError):
        return None
    return s if 0 <= s <= 6 else None

def ask_cssrs_severity(journal_text: str) -> tuple[int | None, dict, str, str]:
    """Returns (severity, reasonings, user_prompt, raw_response)."""
    user_prompt = (
        USER_PROMPT_TEMPLATE
        .replace("{FEW_SHOT_EXAMPLES}", FEW_SHOT_EXAMPLES)
        .replace("{INSERT_TEXT_HERE}", journal_text)
    )

    raw = ""
    try:
        response = bedrock.converse(
            modelId=MODEL_ID,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": TEMPERATURE},
        )
        raw = response["output"]["message"]["content"][0]["text"].strip()
    except ClientError as e:
        logging.error(f"Bedrock error: {e}")
        return None, {q: None for q in Q_LABELS}, user_prompt, ""

    raw_clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        data = json.loads(raw_clean)
    except json.JSONDecodeError:
        logging.error(f"JSON parse failed. Raw:\n{raw[:300]}")
        return None, {q: None for q in Q_LABELS}, user_prompt, raw

    severity = parse_severity(data.get("severity"))
    reasonings = {}
    for q in Q_LABELS:
        val = clean_text(data.get(q))
        reasonings[q] = None if val is None or val.upper() == "N/A" else val

    return severity, reasonings, user_prompt, raw

def severity_to_risk(severity: int | None) -> str:
    """
    Maps the model's 0-6 severity directly to a risk category. Levels 1-5
    follow the ideation ladder; level 6 (behavior) is treated as the most
    severe, per this prompt's own scale. VERIFY these thresholds match your
    study's actual safety protocol before relying on them. A failed/unparsed
    call (severity=None) is reported as "ERROR", not "NONE" — it must not be
    read as a confirmed absence of risk.
    """
    if severity is None:
        return "ERROR"
    if severity >= 4:
        return "HIGH"
    if severity == 3:
        return "MODERATE"
    if severity in (1, 2):
        return "LOW"
    return "NONE"

if __name__ == "__main__":

    # ══════════════════════════════════════════════════════════════════════════════
    # AWS BEDROCK CLIENT
    # ══════════════════════════════════════════════════════════════════════════════
    print(f"Connecting to AWS Bedrock ({AWS_REGION})...")
    session = boto3.Session(profile_name="wcm")
    bedrock = session.client("bedrock-runtime", region_name=AWS_REGION)
    print("Connected.\n")
    # ══════════════════════════════════════════════════════════════════════════════
    # LOAD DATA
    # ══════════════════════════════════════════════════════════════════════════════
    print(f"Loading {INPUT_FILE}...")
    grouped = pd.read_csv(INPUT_FILE)
    if "REDCap ID" in grouped.columns:
        grouped = grouped.rename(columns={"REDCap ID": "REDCap_ID"})
    grouped = grouped.dropna(subset=["REDCap_ID", "Entry_masked"])
    grouped["Entry_masked"] = grouped["Entry_masked"].astype(str).str.strip()
    grouped = grouped[grouped["Entry_masked"].str.len() > 0]
    grouped.rename(columns={"Entry_masked": "Entry"}, inplace=True)

    # ── Resume from checkpoint ───────────────────────────────────────────────────
    already_done = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            done_df      = pd.read_csv(OUTPUT_FILE)
            valid_mask   = pd.to_numeric(done_df["severity"], errors="coerce").between(0, 6)
            valid_df     = done_df[valid_mask]
            already_done = set(valid_df["REDCap_ID"].astype(int).tolist())
            print(f"Resuming — {len(already_done)} successful, {(~valid_mask).sum()} failed (will retry).")
            valid_df.to_csv(OUTPUT_FILE, index=False)
        except Exception:
            pass

    pending = grouped[~grouped["REDCap_ID"].astype(int).isin(already_done)].reset_index(drop=True)
    total   = len(grouped)
    if PILOT_N is not None:
        pending = pending.head(PILOT_N)
        print(f"PILOT MODE — running {len(pending)} of {total} participants (~{100*len(pending)//total}% cost)")
    print(f"Participants total: {total}  |  remaining: {len(pending)}\n")

    # ══════════════════════════════════════════════════════════════════════════════
    # BATCH PROCESSING
    # ══════════════════════════════════════════════════════════════════════════════
    def process_participant(idx: int, row):
        rid  = int(row["REDCap_ID"])
        text = truncate_journal(str(row["Entry"]))
        done = len(already_done) + idx + 1

        print(f"\n[{done}/{total}] REDCap ID {rid}", flush=True)

        severity, reasonings, user_prompt, raw_response = ask_cssrs_severity(text)

        with open(PROMPT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "REDCap_ID":      rid,
                "prompt_version": PROMPT_VERSION,
                "timestamp":      datetime.now().isoformat(),
                "system_prompt":  SYSTEM_PROMPT,
                "user_prompt":    user_prompt,
                "raw_response":   raw_response,
            }, ensure_ascii=False) + "\n")

        risk = severity_to_risk(severity)
        print(f"  severity={severity if severity is not None else '?'}/6  risk={risk}", flush=True)

        if risk == "HIGH":
            logging.warning(f"REDCap ID {rid} flagged HIGH risk — route to study safety protocol for review.")
        elif risk == "ERROR":
            logging.warning(f"REDCap ID {rid} — screening call failed/unparseable, not a confirmed NONE.")

        out = {
            "REDCap_ID":     rid,
            "severity":      severity,
            "risk":          risk,
            "items_present": ", ".join(Q_LABELS[q] for q in Q_LABELS if reasonings.get(q)),
        }
        for q in Q_LABELS:
            out[f"{q}_reason"] = reasonings.get(q)

        return out

    def save_checkpoint(row: dict):
        row_df = pd.DataFrame([row])
        for col in row_df.select_dtypes(include="object").columns:
            row_df[col] = row_df[col].astype(str).str.replace(r"[\n\r]+", " ", regex=True).str.strip()
        write_header = not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0
        row_df.to_csv(OUTPUT_FILE, mode="a", header=write_header, index=False)

    def compute_metrics(pred_col: str, gt_col: str, merged: pd.DataFrame) -> dict:
        y_pred = merged[pred_col].astype(int)
        y_true = merged[gt_col].astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "f1":          f1_score(y_true, y_pred, zero_division=0),
            "mcc":         matthews_corrcoef(y_true, y_pred),
            "kappa":       cohen_kappa_score(y_true, y_pred),
            "auc":         roc_auc_score(y_true, y_pred),
            "n":           len(merged),
        }

    # ══════════════════════════════════════════════════════════════════════════════
    # MLFLOW RUN
    # ══════════════════════════════════════════════════════════════════════════════
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlflow.db')}")
    mlflow.set_experiment("CSSRS_Risk_Screening")
    run_name = f"{PROMPT_VERSION}_{datetime.now().strftime('%m%d_%H%M')}"

    with mlflow.start_run(run_name=run_name):

        mlflow.log_params({
            "model_id":           MODEL_ID,
            "max_journal_chars":  MAX_JOURNAL_CHARS,
            "temperature":        TEMPERATURE,
            "max_tokens":         MAX_TOKENS,
            "prompt_version":     PROMPT_VERSION,
            "system_prompt_hash": prompt_hash(SYSTEM_PROMPT),
            "prompt_template_hash": prompt_hash(USER_PROMPT_TEMPLATE),
            "few_shot_hash":      prompt_hash(FEW_SHOT_EXAMPLES),
            "few_shot":           True,
            "n_participants":     total,
            "input_file":         INPUT_FILE,
        })

        with open("/tmp/cssrs_system_prompt.txt", "w") as f:
            f.write(SYSTEM_PROMPT)
        with open("/tmp/cssrs_prompt_template.txt", "w") as f:
            f.write(USER_PROMPT_TEMPLATE)
        with open("/tmp/cssrs_few_shot_examples.txt", "w") as f:
            f.write(FEW_SHOT_EXAMPLES)
        mlflow.log_artifact("/tmp/cssrs_system_prompt.txt")
        mlflow.log_artifact("/tmp/cssrs_prompt_template.txt")
        mlflow.log_artifact("/tmp/cssrs_few_shot_examples.txt")

        # ── Processing loop ───────────────────────────────────────────────────────
        rows = []
        for idx, row in pending.iterrows():
            try:
                result = process_participant(idx, row)
                rows.append(result)
                save_checkpoint(result)
            except KeyboardInterrupt:
                print(f"\nInterrupted — {len(already_done) + len(rows)} participants saved.")
                raise SystemExit(0)

        print(f"\nDone. Saved '{OUTPUT_FILE}'  ({len(already_done) + len(rows)} participants total)")

        mlflow.log_artifact(OUTPUT_FILE)
        if os.path.exists(PROMPT_LOG_FILE):
            mlflow.log_artifact(PROMPT_LOG_FILE)

        # ── Compute and log metrics against ground truth, if available ────────────
        def print_metrics(label: str, m: dict):
            print(f"\n── {label} (n={m['n']}) ──")
            print(f"  Sensitivity : {m['sensitivity']:.4f}")
            print(f"  Specificity : {m['specificity']:.4f}")
            print(f"  F1          : {m['f1']:.4f}")
            print(f"  MCC         : {m['mcc']:.4f}")
            print(f"  Kappa       : {m['kappa']:.4f}")
            print(f"  AUC         : {m['auc']:.4f}")

        pred = pd.read_csv(OUTPUT_FILE)
        if "At_Risk" in grouped.columns:
            gt = grouped[["REDCap_ID", "At_Risk"]].drop_duplicates(subset="REDCap_ID")
            pred["pred_at_risk"] = (pred["severity"].fillna(0) > 0).astype(int)
            merged = pd.merge(pred[["REDCap_ID", "pred_at_risk"]], gt, on="REDCap_ID")

            m = compute_metrics("pred_at_risk", "At_Risk", merged)
            mlflow.log_metrics({f"atrisk_{k}": round(v, 4) for k, v in m.items() if k != "n"})
            print_metrics("AT-RISK (severity > 0) metrics", m)
        elif "GT_high_risk" in grouped.columns:
            gt = grouped[["REDCap_ID", "GT_high_risk"]].drop_duplicates(subset="REDCap_ID")
            pred["pred_high_risk"] = (pred["risk"] == "HIGH").astype(int)
            merged = pd.merge(pred[["REDCap_ID", "pred_high_risk"]], gt, on="REDCap_ID")

            m = compute_metrics("pred_high_risk", "GT_high_risk", merged)
            mlflow.log_metrics({f"highrisk_{k}": round(v, 4) for k, v in m.items() if k != "n"})
            print_metrics("HIGH-risk flag metrics", m)
        else:
            print("\nSkipping ground-truth metrics — no At_Risk/GT_high_risk column found in input file.")

        print(f"\nMLflow run: {mlflow.active_run().info.run_id}")
        print("View UI: mlflow ui  (then open http://localhost:5000)")
