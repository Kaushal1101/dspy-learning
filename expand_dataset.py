"""
Expand data/dc_dataset.xlsx to ~660 examples (20 per sub-domain).

Generator : gpt-4o-mini  @ temperature=1.0  (batch of 5 per call)
Supervisor: gpt-4o-mini  every SUPERVISOR_INTERVAL new rows
             — rates variety 1-10 and flags near-duplicates / gaps

Run from the repo root:
    python expand_dataset.py
"""

import json, sys, time, textwrap
from pathlib import Path
from collections import defaultdict

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
client = OpenAI()

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH        = Path("data/dc_dataset.xlsx")
OUTPUT_PATH         = Path("data/dc_dataset.xlsx")   # overwrite in place
TARGET_PER_SUBDOMAIN = 20
GEN_MODEL           = "openai/gpt-4o-mini"
SUPER_MODEL         = "openai/gpt-4o"                # stronger for supervision
TEMPERATURE         = 1.0
BATCH_SIZE          = 5                               # utterances per API call
SUPERVISOR_INTERVAL = 50                              # supervisor check cadence

# ── Taxonomy (domain → sub-domain → description) ─────────────────────────────
TAXONOMY = {
    "AppointmentManagement": {
        "View":           "checking or viewing existing appointment details, dates, times, doctor name",
        "Schedule":       "booking or scheduling a NEW appointment, or asking for available slots",
        "Reschedule":     "rescheduling or changing the date/time of an EXISTING appointment",
        "Cancel":         "cancelling an existing appointment",
        "Register":       "pre-registration before a visit, getting a queue number, or checking queue status",
        "GeneralInquiry": "general appointment admin questions: preparation, referrals, subsidy eligibility, switching doctor, estimated waiting time",
    },
    "PaymentBillingSupport": {
        "ViewOutstandingBalance": "checking how much is currently owed on a medical bill",
        "RequestInvoice":         "requesting, downloading, or viewing a medical invoice or official bill document",
        "MakePayment":            "making or completing a payment for a medical bill",
        "GeneralInquiry":         "general billing questions: payment methods, itemized charges, what a charge is for, billing disputes, Medisave/insurance use, processing timelines",
    },
    "MedicationPrescriptionAssistance": {
        "RefillPrescription":  "requesting a REFILL of an existing prescription, or checking medication balance / stock",
        "RenewPrescription":   "RENEWING or extending a prescription that has expired or is about to expire, getting doctor re-approval to continue medication",
        "ViewRefillHistory":   "viewing the history or status of previously SUBMITTED medication refill requests (not the same as prescription records)",
        "GeneralInquiry":      "general questions about medication refills: how to edit or cancel a submitted request, delivery options, collection points, cross-institution collection, processing timelines, proxy collection",
    },
    "HealthRecordsAccess": {
        "AccessImmunizationRecords":       "accessing vaccination records from NIR/NEHR (flu, tetanus, Hep B, measles, chickenpox/varicella, HPV, other NCIS/NAIS vaccines)",
        "AccessHealthScreeningRecords":    "accessing Healthier SG screening results: cardiovascular, cervical, colorectal cancer, breast cancer screening",
        "AccessDischargeSummary":          "accessing records from a hospital stay, day surgery, or A&E visit: admission details, diagnosis, procedures performed, prescriptions given",
        "AccessDrugAllergyRecords":        "accessing drug allergy records, Adverse Drug Reactions (ADR), or G6PD deficiency status — NOT food allergies",
        "AccessLabReports":                "accessing general lab test results: full blood count, lipid profile, liver/kidney function, urinalysis, glucose, HbA1c",
        "AccessRadiologyReports":          "accessing imaging results: X-ray, MRI, CT scan, PET scan, mammogram, ultrasound",
        "AccessHistologyCytologyReports":  "accessing biopsy results, Pap smear results, or Fine Needle Aspiration Cytology (FNAC)",
        "AccessGeneticReports":            "accessing genetic test results specifically for Familial Hypercholesterolaemia (FH)",
        "GeneralInquiry":                  "general health records questions: why records haven't appeared yet, processing delays, requesting urgent access, updating drug allergy or personal health details",
    },
    "GeneralHealthInquiry": {
        "MedicationRelated":                "informational questions about medications, drug side effects, dosage, drug interactions, storage — NOT requesting a refill/renewal",
        "MentalHealthSupport":              "mental health, emotional wellbeing, stress, anxiety, depression, grief, loneliness, counseling options — informational, not transactional",
        "GeneralHealthcareServicesRelated": "informational questions about how HealthHub services work, eligibility, Healthier SG enrolment, referral pathways, what a service is for — NOT performing a transaction",
        "GeneralHealthRelated":             "general health questions: symptoms, conditions, diet, exercise, wellness, preventive health — not covered by any other sub-domain",
        "HealthcareInstitutionInformation": "questions about hospital or clinic locations, opening hours, contact numbers, which hospital offers a service, comparing healthcare providers",
        "Emergency":                        "urgent or potentially life-threatening symptoms requiring IMMEDIATE medical attention: chest pain, difficulty breathing, suspected stroke or heart attack, severe bleeding, loss of consciousness — also borderline side-effect vs emergency cases",
    },
}

VARIETY_GUIDANCE = """\
Generate utterances that are varied along these dimensions:
- Phrasing style: formal ("I would like to..."), casual ("Can I..."), terse ("Check my appt"),
  Singlish-inflected ("My appointment still there or not?"), third-person ("My mum needs to...")
- Specificity: vague ("I need help with my records") vs specific ("I need my MRI scan from SGH last month")
- Scenario diversity: use different conditions, medications, departments, family members, time contexts
- Sentence length: mix short questions with longer multi-clause sentences
- Do NOT start multiple utterances the same way. No two utterances should be near-paraphrases of each other.
- Every utterance must be unambiguously self-contained — no pronouns referencing prior context ("Can you do that?")
"""


def build_generation_prompt(domain: str, sub_domain: str, description: str,
                             existing: list[str], n: int) -> str:
    existing_block = "\n".join(f"  - {u}" for u in existing[:15]) if existing else "  (none yet)"
    return textwrap.dedent(f"""\
        You are building a labelled dataset for a healthcare chatbot domain classifier.

        Generate exactly {n} diverse, realistic user utterances that belong to:
          Domain    : {domain}
          Sub-domain: {sub_domain}
          Meaning   : {description}

        {VARIETY_GUIDANCE}

        Existing utterances for this sub-domain (DO NOT repeat or closely paraphrase these):
        {existing_block}

        Return ONLY a JSON array of {n} strings. No commentary, no keys — just the array.
        Example format: ["utterance one", "utterance two", ...]
    """)


def generate_batch(domain: str, sub_domain: str, description: str,
                   existing: list[str], n: int) -> list[str]:
    prompt = build_generation_prompt(domain, sub_domain, description, existing, n)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=GEN_MODEL.replace("openai/", ""),
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=1200,
            )
            text = resp.choices[0].message.content.strip()
            # strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            utterances = json.loads(text)
            if isinstance(utterances, list):
                return [str(u).strip() for u in utterances if str(u).strip()]
        except Exception as e:
            print(f"    [gen error attempt {attempt+1}] {e}")
            time.sleep(2 ** attempt)
    return []


def supervisor_check(new_rows: list[dict], all_generated_so_far: list[dict]) -> None:
    sample_text = "\n".join(
        f"  [{r['domain']}/{r['sub_domain']}] {r['utterance']}" for r in new_rows
    )
    total_so_far = len(all_generated_so_far)
    dist = defaultdict(int)
    for r in all_generated_so_far:
        dist[f"{r['domain']}/{r['sub_domain']}"] += 1
    dist_text = "\n".join(f"  {k}: {v}" for k, v in sorted(dist.items()))

    prompt = textwrap.dedent(f"""\
        You are supervising the quality and variety of a synthetic dataset being generated
        for a healthcare chatbot domain classifier.

        Here are the last {len(new_rows)} utterances just generated:
        {sample_text}

        Current sub-domain distribution ({total_so_far} total generated so far):
        {dist_text}

        Please evaluate:
        1. Variety score (1-10): Are the utterances genuinely diverse in phrasing, style, and scenario?
        2. Near-duplicates: List any utterances that are too similar to each other (direct them by prefix).
        3. Gaps: Which sub-domains or scenarios seem under-represented or too narrow in coverage?
        4. One-line recommendation for the next batch.

        Be concise. Format your response as:
        Variety: <score>/10
        Near-duplicates: <list or "none">
        Gaps: <list or "none">
        Recommendation: <one sentence>
    """)

    try:
        resp = client.chat.completions.create(
            model=SUPER_MODEL.replace("openai/", ""),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
        )
        print("\n" + "─" * 60)
        print(f"SUPERVISOR CHECK (after {total_so_far} generated rows)")
        print("─" * 60)
        print(resp.choices[0].message.content.strip())
        print("─" * 60 + "\n")
    except Exception as e:
        print(f"  [supervisor error] {e}")


def main():
    # ── Load existing data ────────────────────────────────────────────────────
    df = pd.read_excel(DATASET_PATH)
    df = df[df["ir_utterance_count"] > 0].reset_index(drop=True)
    # Drop Issues sub-domain — not in the classifier taxonomy
    issues_mask = df["sub_domain"] == "Issues"
    if issues_mask.any():
        print(f"Dropping {issues_mask.sum()} existing 'Issues' rows (not in taxonomy).")
        df = df[~issues_mask].reset_index(drop=True)
    print(f"Loaded {len(df)} existing rows across "
          f"{df.groupby(['domain','sub_domain']).ngroups} sub-domains.\n")

    # existing utterances per sub-domain (for dedup)
    existing_map: dict[tuple, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        existing_map[(row["domain"], row["sub_domain"])].append(row["utterance"])

    # ── Plan generation ───────────────────────────────────────────────────────
    plan: list[tuple[str, str, int]] = []   # (domain, sub_domain, n_to_generate)
    for domain, sub_domains in TAXONOMY.items():
        for sub_domain in sub_domains:
            have = len(existing_map.get((domain, sub_domain), []))
            need = max(0, TARGET_PER_SUBDOMAIN - have)
            if need > 0:
                plan.append((domain, sub_domain, need))

    total_to_generate = sum(n for _, _, n in plan)
    print(f"Plan: generate {total_to_generate} new utterances across {len(plan)} sub-domains.")
    print(f"Target total: {len(df) + total_to_generate} rows.\n")

    # ── Generate ──────────────────────────────────────────────────────────────
    new_rows: list[dict] = []
    supervisor_buffer: list[dict] = []

    for domain, sub_domain, need in plan:
        description = TAXONOMY[domain][sub_domain]
        existing = existing_map[(domain, sub_domain)].copy()
        generated_for_this: list[str] = []
        remaining = need

        print(f"  {domain}/{sub_domain}: need {need} "
              f"(have {len(existing)}) ...", end=" ", flush=True)

        while remaining > 0:
            batch_n = min(BATCH_SIZE, remaining)
            batch = generate_batch(
                domain, sub_domain, description,
                existing + generated_for_this, batch_n
            )
            for utt in batch:
                # simple dedup: skip if too similar to any existing
                lower = utt.lower()
                if any(lower == e.lower() for e in existing + generated_for_this):
                    continue
                generated_for_this.append(utt)
                row = dict(
                    utterance=utt,
                    domain=domain,
                    sub_domain=sub_domain,
                    ir_utterance_count=1,
                    alternative_domain=None,
                    alternative_sub_domain=None,
                )
                new_rows.append(row)
                supervisor_buffer.append(row)
                remaining -= 1
                if remaining <= 0:
                    break

            # supervisor check
            if len(supervisor_buffer) >= SUPERVISOR_INTERVAL:
                supervisor_check(supervisor_buffer, new_rows)
                supervisor_buffer = []

            time.sleep(0.3)   # gentle rate-limit back-off

        print(f"done ({len(generated_for_this)} added)")

    # final supervisor check if there's a leftover buffer
    if supervisor_buffer:
        supervisor_check(supervisor_buffer, new_rows)

    # ── Save ──────────────────────────────────────────────────────────────────
    new_df = pd.DataFrame(new_rows)

    # align columns with original
    for col in df.columns:
        if col not in new_df.columns:
            new_df[col] = None
    new_df = new_df[df.columns]

    combined = pd.concat([df, new_df], ignore_index=True)
    combined.to_excel(OUTPUT_PATH, index=False)

    print(f"\nDone. {len(new_rows)} rows added → {len(combined)} total rows saved to {OUTPUT_PATH}")
    print("\nFinal sub-domain distribution:")
    dist = combined.groupby(["domain", "sub_domain"]).size().reset_index(name="count")
    for _, row in dist.iterrows():
        print(f"  {row['domain']}/{row['sub_domain']}: {row['count']}")


if __name__ == "__main__":
    main()
