import os
import re
import json
import logging
import warnings
import boto3
from botocore.config import Config
import pandas as pd
from datetime import datetime
from botocore.exceptions import ClientError

os.environ["AWS_SHARED_CREDENTIALS_FILE"] = os.path.expanduser("~/.aws/saml2aws_credentials")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
# LLM-as-structured-annotator theme extraction: instead of a binary/severity
# risk judgment (see claude_bedrock_cssrs.py), the model rates each of a
# fixed set of theory-driven psychological constructs on a 0-3 anchored
# scale, based on how clearly/strongly each is expressed in the text. The
# resulting rating vector is meant to be fed into a downstream classifier
# (L1 logistic regression), not read as a risk verdict directly -- this
# script only does the extraction step.
#
# SAFETY NOTE: research screening aid, not a diagnostic instrument. See
# claude_bedrock_cssrs.py's safety note -- same caveats apply.
INPUT_FILE        = "Non_phi_data_with_risk.csv"
OUTPUT_FILE       = "claude_bedrock_theme_ratings_10test.csv"
TEXT_COLUMN       = "Entry_masked"
MAX_JOURNAL_CHARS = 100000
AWS_REGION        = "us-east-1"
MODEL_ID          = "us.anthropic.claude-sonnet-4-6"
TEMPERATURE       = 0.01
MAX_TOKENS        = 8000
PROMPT_LOG_FILE   = "claude_prompt_log_theme_extraction.jsonl"

# (construct_name, one-sentence definition) -- same construct set used in
# lexical_features_v2_constructs.py, now rated by LLM judgment instead of
# mechanical phrase matching.
CONSTRUCTS = {
    "negative_emotion": "General negative affect: sadness, grief, anger, guilt, shame, fear, distress.",
    "positive_emotion": "General positive affect: happiness, love, gratitude, hope, comfort, peace, pride.",
    "hopelessness": "Belief that the future is bleak or that things will not improve; giving up on things getting better (Beck's hopelessness construct).",
    "absolutist_thinking": "All-or-nothing language and overgeneralization (always, never, everyone, nothing, completely) applied to one's situation or self-worth.",
    "death_suicide_ideation": "Explicit references to dying, death, suicide, or wanting to not be alive -- either about the self or, if about someone else, framed in a way that reveals the writer's own relationship to the idea.",
    "burdensomeness": "Belief that one is a burden to others, or that others would be better off without them (Joiner's Interpersonal Theory of Suicide).",
    "belongingness_absent": "Lack of connection to others; feeling alone, unseen, or uncared for, or the loss of a sense of belonging (Joiner's Interpersonal Theory of Suicide).",
    "entrapment": "Feeling trapped, stuck, or that there is no way out of one's current situation or pain (O'Connor's Integrated Motivational-Volitional model).",
    "defeat": "A sense of having lost, been beaten down, or exhausted by struggle, distinct from simple sadness (O'Connor's model -- precedes entrapment).",
    "psychache": "Psychological pain described as unbearable, all-consuming, or beyond what can be tolerated (Shneidman's construct).",
    "reunification_longing": "An expressed wish to join, be with, or see again someone who has died.",
    "identity_collapse": "Loss of sense of self; not recognizing who one is anymore, feeling like an empty shell, or merely existing rather than living.",
    "self_blame": "Blaming oneself for negative events, holding oneself responsible in a harsh or global way (my fault, I failed, I caused this).",
    "agency_helplessness": "Feeling that one has no control or influence over one's own life or circumstances.",
    "protective_factors": "Explicit reasons to keep living: responsibilities, relationships, future plans, or things the writer is grateful for or looking forward to.",
    "cognitive_rumination": "Repetitive, intrusive dwelling on distressing thoughts that the writer cannot stop or escape.",
    "social_withdrawal": "Actively pulling away from other people: avoiding contact, isolating oneself, not answering, not wanting to see anyone.",
    "caregiver_role_loss": "Loss of a caregiving role or sense of purpose tied to caring for someone else who has died or is gone.",
    "meaning_loss": "A sense that life, or a specific pursuit, has lost meaning or purpose.",
}

SYSTEM_PROMPT = (
    "You are a clinical psychologist rating journal text against a fixed set "
    "of psychological constructs for a research study. For each construct, "
    "assign a rating on this anchored 0-3 scale:\n"
    "  0 = absent -- no evidence of this construct in the text\n"
    "  1 = hinted -- vague or indirect evidence, not clearly stated\n"
    "  2 = present -- clearly and explicitly expressed at least once\n"
    "  3 = dominant -- a central, recurring theme of the text\n\n"
    "Base every rating only on what is stated or clearly implied in THIS "
    "text -- do not infer beyond it, and do not let one construct's rating "
    "influence another's. A single clear statement is sufficient for a "
    "rating of 2; it does not need to recur. Provide a brief quote or "
    "paraphrase as justification for every rating above 0."
)

USER_PROMPT_TEMPLATE = """Rate this journal entry against each construct below, using the 0-3 scale
from your instructions.

Constructs:
{CONSTRUCT_LIST}

Respond ONLY in this JSON format, with no other text -- one entry per construct,
using the exact construct names given above:
{
  "<construct_name>": {"rating": <0-3>, "evidence": "<brief quote/paraphrase, or empty string if rating is 0>"},
  ...
}

Text to evaluate:
\"\"\"
{INSERT_TEXT_HERE}
\"\"\"
"""


def build_construct_list_text() -> str:
    return "\n".join(f"- {name}: {definition}" for name, definition in CONSTRUCTS.items())


def truncate_journal(text: str) -> str:
    return text[:MAX_JOURNAL_CHARS]


def extract_json_block(raw: str):
    raw_clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw_clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_clean, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def rate_themes(journal_text: str) -> tuple[dict | None, str, str]:
    """Returns (ratings_dict, user_prompt, raw_response). ratings_dict maps
    construct_name -> (rating:int, evidence:str)."""
    user_prompt = (
        USER_PROMPT_TEMPLATE
        .replace("{CONSTRUCT_LIST}", build_construct_list_text())
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
        return None, user_prompt, ""

    data = extract_json_block(raw)
    if data is None:
        logging.error(f"JSON parse failed. Raw:\n{raw[:300]}")
        return None, user_prompt, raw

    ratings = {}
    for name in CONSTRUCTS:
        entry = data.get(name)
        if isinstance(entry, dict):
            try:
                rating = int(entry.get("rating"))
                rating = rating if 0 <= rating <= 3 else None
            except (TypeError, ValueError):
                rating = None
            evidence = entry.get("evidence") or ""
        else:
            rating, evidence = None, ""
        ratings[name] = (rating, evidence)

    return ratings, user_prompt, raw


if __name__ == "__main__":
    print(f"Connecting to AWS Bedrock ({AWS_REGION})...")
    session = boto3.Session(profile_name="wcm")
    boto_config = Config(read_timeout=300, connect_timeout=30, retries={"max_attempts": 3})
    bedrock = session.client("bedrock-runtime", region_name=AWS_REGION, config=boto_config)
    print("Connected.\n")

    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    if "REDCap ID" in df.columns:
        df = df.rename(columns={"REDCap ID": "REDCap_ID"})
    df = df.dropna(subset=["REDCap_ID", TEXT_COLUMN])
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df = df[df[TEXT_COLUMN].str.len() > 0].reset_index(drop=True)

    already_done = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            done_df = pd.read_csv(OUTPUT_FILE)
            rating_cols = [f"{c}_rating" for c in CONSTRUCTS]
            valid_mask = done_df[rating_cols].notna().all(axis=1)
            valid_df = done_df[valid_mask]
            already_done = set(valid_df["REDCap_ID"].astype(int).tolist())
            print(f"Resuming — {len(already_done)} already done.")
            valid_df.to_csv(OUTPUT_FILE, index=False)
        except Exception:
            pass

    total = len(df)
    pending = df[~df["REDCap_ID"].astype(int).isin(already_done)].reset_index(drop=True)
    print(f"Total: {total}  |  remaining: {len(pending)}\n")

    def save_checkpoint(row: dict):
        row_df = pd.DataFrame([row])
        for col in row_df.select_dtypes(include="object").columns:
            row_df[col] = row_df[col].astype(str).str.replace(r"[\n\r]+", " ", regex=True).str.strip()
        write_header = not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0
        row_df.to_csv(OUTPUT_FILE, mode="a", header=write_header, index=False)

    for idx, row in pending.iterrows():
        rid = int(row["REDCap_ID"])
        text = truncate_journal(str(row[TEXT_COLUMN]))
        done = len(already_done) + idx + 1
        print(f"[{done}/{total}] REDCap ID {rid}", flush=True)

        ratings, user_prompt, raw_response = rate_themes(text)

        with open(PROMPT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "REDCap_ID": rid,
                "timestamp": datetime.now().isoformat(),
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "raw_response": raw_response,
            }, ensure_ascii=False) + "\n")

        out = {"REDCap_ID": rid, "At_Risk": row.get("At_Risk")}
        if ratings:
            for name in CONSTRUCTS:
                rating, evidence = ratings[name]
                out[f"{name}_rating"] = rating
                out[f"{name}_evidence"] = evidence
            print(f"  ratings: {[(n, ratings[n][0]) for n in CONSTRUCTS]}", flush=True)
        else:
            for name in CONSTRUCTS:
                out[f"{name}_rating"] = None
                out[f"{name}_evidence"] = None
            print("  FAILED", flush=True)

        save_checkpoint(out)

    print(f"\nDone. Saved to {OUTPUT_FILE}")
