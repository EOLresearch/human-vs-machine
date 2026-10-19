# How to Run (Quick Guide)

From login to results in 6 steps. Commands are for macOS/Linux Terminal.

---

## 1. One-time setup

```bash
brew install saml2aws awscli      # login tool + AWS CLI
python3 --version                 # needs Python 3.10 or newer
```

Configure saml2aws once. Your IT/cloud team gives you the SSO (identity provider) details:

```bash
saml2aws configure \
  --idp-provider=<your IdP, e.g. AzureAD / Okta / ADFS> \
  --url=<your SSO URL> \
  --username=<your institutional email> \
  --profile=wcm \
  --region=us-east-1
```

The scripts expect the profile **`wcm`** and the credentials file
**`~/.aws/saml2aws_credentials`**. To make saml2aws write there, add this line under your account
in `~/.saml2aws`:

```
aws_credentials_file = ~/.aws/saml2aws_credentials
```

## 2. Log in to AWS (every session, tokens expire after a few hours)

```bash
saml2aws login --profile=wcm
```

Enter your password and approve MFA. Check that it worked:

```bash
AWS_SHARED_CREDENTIALS_FILE=~/.aws/saml2aws_credentials aws sts get-caller-identity --profile wcm
```

If you see your account/role, you're in.

## 3. Set up Python (one time)

```bash
cd Bedrock-Journal-Analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(Next time, just `cd` into the folder and run `source .venv/bin/activate`.)

## 4. Add the data file

Copy the CSV from secure study storage into this folder. Its name must match `INPUT_FILE` in the
script:

| Script | Expects |
|---|---|
| `claude_bedrock_cssrs.py` | `Non_phi_data_with_risk.csv` |
| `claude_bedrock_theme_extraction.py` | `Non_phi_data_with_risk.csv` |
| `claude_bedrock_theme_frequency.py` | `non_phi_risk_assessment.csv` |
| `claude_bedrock_theme_frequency_77.py` | `Non_phi_data_with_risk.csv` |

Required columns: `REDCap_ID` and `Entry_masked`. `At_Risk` is optional.

## 5. (Optional) Edit settings

Open the script and change the constants at the top: `INPUT_FILE`, `OUTPUT_FILE`, `MODEL_ID`,
`MAX_TOKENS` (and `PILOT_N` in cssrs to run only the first N participants).

## 6. Run

```bash
python claude_bedrock_cssrs.py                 # or any other script name
```

You'll see progress like `[3/10] REDCap ID 204` and a result line for each participant.

**Results:**
- `OUTPUT_FILE` (CSV): one row per participant, saved as it goes
- `claude_prompt_log_*.jsonl`: full prompt and response for every call
- cssrs only: run `mlflow ui` and open http://localhost:5000 to see runs and metrics

**Stopped halfway?** Run the same command again. Finished participants are skipped.
**Want a clean re-run?** Delete or rename the output CSV first.

---

## Common errors

| Error | Fix |
|---|---|
| `ExpiredToken` / `UnrecognizedClientException` | Run `saml2aws login --profile=wcm` again |
| `ProfileNotFound: wcm` | Re-run `saml2aws configure --profile=wcm` |
| `AccessDeniedException` | Your role can't use that model. Request access in Bedrock → Model access, or change `MODEL_ID` |
| `FileNotFoundError: ...csv` | The CSV isn't in this folder, or its name doesn't match `INPUT_FILE` |
| `JSON parse failed` / `FAILED` | The reply was cut off or malformed. Raise `MAX_TOKENS` and re-run (it resumes) |
| `ReadTimeout` | Very long entry. Re-run and it will retry that participant |
