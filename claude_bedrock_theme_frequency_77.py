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
# 77-theme frequency extraction (expanded bereavement/grief taxonomy) -- 10-person
# pilot run, same instance-counting methodology as claude_bedrock_theme_frequency.py
# (count = len(quoted/paraphrased instances), not a self-reported number).
# ══════════════════════════════════════════════════════════════════════════════
INPUT_FILE        = "Non_phi_data_with_risk.csv"
OUTPUT_FILE       = "claude_bedrock_theme_frequency_77themes_10test.csv"
TEXT_COLUMN       = "Entry_masked"
MAX_JOURNAL_CHARS = 100000
AWS_REGION        = "us-east-1"
MODEL_ID          = "us.anthropic.claude-sonnet-4-6"
TEMPERATURE       = 0.01
MAX_TOKENS        = 20000
PROMPT_LOG_FILE   = "claude_prompt_log_theme_frequency_77.jsonl"

CONSTRUCTS = {
    "prolonged_grief_yearning": "Persistent, intense yearning or longing for the deceased that continues without abating over time.",
    "nonacceptance_of_death": "Difficulty accepting the reality or finality of the death; disbelief that the person is really gone.",
    "continuing_bonds": "Maintaining an ongoing sense of connection, relationship, or communication with the deceased (talking to them, feeling their presence).",
    "linking_objects": "Attachment to specific physical objects, possessions, or places that are kept because they belonged to or connect to the deceased.",
    "reminder_triggered_grief_reactivity": "Intense grief reactions triggered by reminders, places, sounds, dates, or objects associated with the deceased.",
    "rumination_counterfactual_thinking": "Repetitive dwelling on 'what if' or 'if only' scenarios about how things could have gone differently before the death.",
    "guilt_self_blame": "Feeling guilty or blaming oneself for the death, for not doing enough, or for things said/unsaid.",
    "ambivalent_attachment_to_deceased": "Mixed or conflicting feelings toward the deceased -- love combined with resentment, anger, or unresolved conflict.",
    "idealization_of_deceased": "Portraying the deceased in an overly positive, idealized light, minimizing flaws or negative aspects.",
    "anger_perceived_betrayal": "Anger at the deceased, at others, or at the situation; feeling betrayed or abandoned by the death.",
    "grievance_medical_providers": "Anger, blame, or grievance directed at doctors, hospitals, or the end-of-life care the deceased received.",
    "identity_disruption_role_loss": "Loss of sense of self or life role tied to the relationship with the deceased (e.g., no longer a spouse, caregiver, child).",
    "identification_with_deceased": "Taking on traits, habits, roles, or characteristics of the deceased as a way of maintaining connection.",
    "loneliness_social_isolation": "Feeling lonely, isolated, or cut off from others since the loss.",
    "loss_of_confidant_secure_base": "Loss of the person who was one's primary source of emotional security, support, or someone to confide in.",
    "search_for_substitute_attachment": "Seeking a new person or relationship to fill the emotional role the deceased used to occupy.",
    "dependency_enmeshment_with_deceased": "Descriptions suggesting an unusually dependent or enmeshed relationship with the deceased prior to death.",
    "depressive_symptomatology": "Sadness, low mood, loss of interest or pleasure, low energy, or other depressive symptoms.",
    "anxiety_worry": "Anxiety, nervousness, worry, or fear about the future or about specific concerns.",
    "sleep_disturbance_dreams": "Trouble sleeping, insomnia, or dreams/nightmares about the deceased.",
    "somatic_distress": "Physical/bodily symptoms of distress (headaches, chest tightness, fatigue, appetite changes) linked to grief.",
    "emotional_numbing_detachment": "Feeling emotionally numb, detached, or disconnected from one's feelings or from others.",
    "affective_lability_crying": "Sudden or frequent crying episodes, or rapid shifts in emotional state.",
    "intrusive_reexperiencing_death_scene": "Intrusive, unwanted memories or images of the illness, dying process, or death scene.",
    "distress_over_deceaseds_suffering": "Distress specifically about the pain or suffering the deceased experienced before dying.",
    "religious_spiritual_coping": "Using religious or spiritual beliefs/practices as a source of comfort or coping with the loss.",
    "existential_spiritual_struggle": "Questioning the meaning of life, death, or one's faith as a result of the loss; existential crisis.",
    "institutional_religious_disillusionment": "Disillusionment, anger, or loss of trust in organized religion or religious institutions because of the death.",
    "meaning_making_benefit_finding": "Efforts to find meaning, growth, or some positive outcome from the loss experience.",
    "legacy_memorialization_generativity": "Actions or thoughts about preserving the deceased's legacy, memorializing them, or carrying their impact forward.",
    "reminiscence_shared_memories": "Recounting or narrating specific shared memories or stories about the deceased.",
    "gratitude_positive_affect": "Expressions of gratitude, appreciation, or positive emotion, including about the relationship with the deceased.",
    "avoidance_distraction_coping": "Avoiding reminders of the loss, or using distraction/busyness to cope rather than confronting grief directly.",
    "desired_detachment_from_deceased": "An expressed wish to feel less attached, to move on, or to reduce the emotional hold of the deceased.",
    "substance_use_recovery_survivor": "The survivor's own use of alcohol/drugs, or their recovery process, as a coping response to the loss.",
    "addiction_substance_related_death": "The deceased's death being related to addiction or substance use.",
    "anniversary_holiday_exacerbation": "Worsened grief around anniversaries, holidays, or other significant recurring dates.",
    "milestone_related_grief": "Grief triggered by life milestones (graduations, weddings, births) the deceased is not present for.",
    "loss_of_shared_future_plans": "Grief over plans, trips, or a shared future that had been planned with the deceased and will no longer happen.",
    "grief_for_deceaseds_unlived_future": "Grief specifically about what the deceased themselves will never get to experience or do.",
    "unfinished_business_unexpressed_emotion": "A sense of unfinished business, or things left unsaid/undone with the deceased.",
    "unanswered_questions_about_deceased": "Unanswered questions about the deceased's life, thoughts, or inner world that can no longer be resolved.",
    "value_ideological_discordance": "A sense of value or ideological conflict/disagreement with the deceased.",
    "passive_death_wishes_si": "Passive wishes to not be alive, to not wake up, or explicit suicidal ideation.",
    "personal_mortality_salience": "Increased awareness of, or preoccupation with, one's own mortality as a result of the death.",
    "memory_loss_fear_of_forgetting": "Fear of forgetting the deceased, their voice, face, or memories of them; distress over fading memories.",
    "seeking_deceaseds_approval": "Seeking or imagining the deceased's approval, pride, or validation for one's actions.",
    "family_conflict_estrangement": "Conflict, tension, or estrangement within the family, including conflict arising from or around the death.",
    "emotional_inhibition_family_of_origin": "Description of one's family of origin as emotionally inhibited, restrictive, or unsupportive of expressing feelings.",
    "history_of_interpersonal_abuse": "References to a history of interpersonal abuse (physical, emotional, sexual) in the writer's life.",
    "current_intimate_relationship_strain": "Strain, conflict, or difficulty in a current romantic/intimate relationship.",
    "perceived_social_support": "Feeling supported by friends, family, or community; presence of a support network.",
    "perceived_invalidation_disenfranchised_grief": "Feeling that one's grief is not recognized, validated, or supported by others (disenfranchised grief).",
    "concern_for_other_survivors": "Concern or worry about how other surviving family members or friends are coping.",
    "caregiver_burden_role_reversal": "Burden from having been a caregiver, or a reversal of caregiving roles, prior to or at the time of death.",
    "parental_role_investment": "Investment in, or reflections on, one's role as a parent (either as caregiver to children or in relation to one's own parent).",
    "companion_animal_attachment": "Attachment to, comfort from, or grief involving a pet/companion animal.",
    "yearning_for_physical_affection_touch": "Longing for physical touch, affection, or closeness that was present in the relationship with the deceased.",
    "food_cooking_as_relational_ritual": "References to food, cooking, or meals as a ritual or way of maintaining connection to the deceased or others.",
    "occupational_academic_stress": "Stress related to work, career, or school/academic demands.",
    "perfectionism_contingent_self_worth": "Perfectionistic tendencies or self-worth that seems contingent on achievement or meeting high standards.",
    "shame_self_devaluation": "Feelings of shame, worthlessness, or harshly devaluing oneself.",
    "financial_strain": "Financial stress, hardship, or worry about money.",
    "estate_settlement_household_burden": "Burden of settling the deceased's estate, finances, or managing household tasks left behind.",
    "relocation_ambivalence_home_environment": "Ambivalence about relocating, moving, or feelings about the home environment tied to the deceased.",
    "geographic_separation_from_family": "Distress or difficulty related to living far from family members.",
    "personal_illness_health_burden": "The writer's own physical illness or health management burden.",
    "self_neglect_decline_in_functioning": "Neglecting self-care, or a decline in daily functioning (hygiene, eating, responsibilities).",
    "restoration_oriented_coping": "Active, forward-looking coping: goal-setting, striving, or engaging with life tasks and restoration.",
    "nature_solitude_physical_activity_regulation": "Using nature, solitude, or physical activity/exercise to regulate emotions or cope.",
    "help_seeking_treatment_engagement": "Seeking or engaging with therapy, counseling, support groups, or other professional help.",
    "cumulative_secondary_losses": "Experiencing multiple or cumulative losses (other deaths, secondary losses like loss of routine, identity, financial security).",
    "perceived_untimeliness_insufficient_time": "A sense that the death was untimely, premature, or that not enough time was had with the deceased.",
    "acceptance_of_timely_death_relief": "Acceptance of the death as timely, or expressions of relief (e.g., after prolonged suffering).",
    "pandemic_related_stressors": "Stressors specifically related to the COVID-19 pandemic (isolation, restrictions, illness).",
    "sociopolitical_demoralization": "Demoralization, distress, or disillusionment related to broader sociopolitical events or climate.",
    "reflection_on_journaling_task": "Meta-commentary reflecting on the act of journaling itself or the writing task/prompt.",
}

SYSTEM_PROMPT = (
    "You are a clinical psychologist extracting evidence of bereavement and grief-related "
    "psychological constructs from journal text for a research study. For each construct, "
    "find EVERY distinct place in the text where it is expressed, and list each one as a "
    "separate short quote or close paraphrase.\n\n"
    "Rules for counting instances:\n"
    "- Each instance must be a SEPARATE expression of the construct, not the same statement "
    "described twice. If a single sentence expresses the construct once, that is one instance, "
    "no matter how emotionally strong the sentence is -- do not list the same passage more than "
    "once and do not split one statement into several instances to inflate the count.\n"
    "- Only list instances that are clearly and explicitly expressed in THIS text -- do not "
    "infer beyond what is stated, and do not let evidence for one construct count toward a "
    "different construct.\n"
    "- If a construct does not appear anywhere in the text, its list must be empty.\n"
    "- Be exhaustive: re-read the full text for each construct rather than stopping after "
    "finding one or two instances.\n"
    "- There are 77 constructs. Most will have empty or near-empty lists for any given entry -- "
    "that is expected and correct. Do not force matches."
)

USER_PROMPT_TEMPLATE = """Extract every instance of each construct below from this journal entry,
following the rules in your instructions.

Constructs:
{CONSTRUCT_LIST}

Respond ONLY in this JSON format, with no other text -- one entry per construct,
using the exact construct names given above. "instances" must be a JSON array
of short quotes/paraphrases, one per distinct occurrence (empty array if none):
{
  "<construct_name>": {"instances": ["<quote or paraphrase 1>", "<quote or paraphrase 2>", ...]},
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
    raw_clean = raw.strip()
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", raw_clean, flags=re.DOTALL)
    for block in reversed(fenced_blocks):
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_clean, flags=re.MULTILINE).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def rate_themes(journal_text: str) -> tuple[dict | None, str, str]:
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
            instances = entry.get("instances")
            instances = instances if isinstance(instances, list) else []
            instances = [str(x) for x in instances if isinstance(x, str) and x.strip()]
            count = len(instances)
        else:
            count, instances = None, []
        ratings[name] = (count, instances)

    return ratings, user_prompt, raw


if __name__ == "__main__":
    print(f"{len(CONSTRUCTS)} constructs loaded.")
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
            rating_cols = [f"{c}_count" for c in CONSTRUCTS]
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
                count, instances = ratings[name]
                out[f"{name}_count"] = count
                out[f"{name}_instances"] = json.dumps(instances, ensure_ascii=False)
            nonzero = [(n, ratings[n][0]) for n in CONSTRUCTS if ratings[n][0]]
            print(f"  nonzero counts: {nonzero}", flush=True)
        else:
            for name in CONSTRUCTS:
                out[f"{name}_count"] = None
                out[f"{name}_instances"] = None
            print("  FAILED", flush=True)

        save_checkpoint(out)

    print(f"\nDone. Saved to {OUTPUT_FILE}")
