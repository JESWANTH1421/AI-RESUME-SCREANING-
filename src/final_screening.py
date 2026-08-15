"""
============================================================
TALENTLENS
NLP-ONLY RECRUITMENT SYSTEM
STAGE 6 - FINAL SCREENING
TOP-K + FRAUD DETECTION INTEGRATION
============================================================

Purpose:
    Combines:
        1. Stage 4 Top-K shortlist
        2. Stage 5 fraud detection

    A candidate can only receive a final shortlist decision
    if the resume passes the fraud check.

Decision logic:

    HIGH fraud
        -> FRAUD FLAGGED

    MEDIUM fraud
        -> MANUAL REVIEW

    LOW fraud
        -> Keep Stage 4 decision

This is deterministic and does not use GenAI.
============================================================
"""

import os
import sys
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "DATA")

SHORTLIST_DIR = os.path.join(
    DATA_DIR,
    "shortlist_results"
)

FRAUD_DIR = os.path.join(
    DATA_DIR,
    "fraud_detection"
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "final_screening"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_header(title):
    print("\n")
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_line():
    print("-" * 60)


# ============================================================
# FIND FILES
# ============================================================

def find_shortlist_file():
    """
    Finds the Stage 4 Top-K shortlist CSV.
    """

    if not os.path.exists(SHORTLIST_DIR):
        return None

    files = [
        f for f in os.listdir(SHORTLIST_DIR)
        if f.lower().endswith(".csv")
    ]

    if not files:
        return None

    # Prefer top_k_shortlist file
    preferred = [
        f for f in files
        if "top_k_shortlist" in f.lower()
    ]

    if preferred:
        return os.path.join(SHORTLIST_DIR, preferred[0])

    return os.path.join(SHORTLIST_DIR, files[0])


def find_fraud_file():
    """
    Finds Stage 5 fraud detection report.
    """

    path = os.path.join(
        FRAUD_DIR,
        "resume_fraud_report.csv"
    )

    if os.path.exists(path):
        return path

    return None


# ============================================================
# LOAD STAGE 4 RESULTS
# ============================================================

def load_shortlist():

    print_header("LOADING STAGE 4 TOP-K RESULTS")

    path = find_shortlist_file()

    if path is None:
        print("ERROR: Stage 4 shortlist file not found.")
        print()
        print("Expected location:")
        print(SHORTLIST_DIR)
        sys.exit(1)

    print(f"Shortlist file found:")
    print(path)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"ERROR loading shortlist: {e}")
        sys.exit(1)

    print(f"\nCandidates loaded: {len(df)}")

    print("\nColumns detected:")

    for column in df.columns:
        print(f" - {column}")

    return df


# ============================================================
# LOAD STAGE 5 FRAUD RESULTS
# ============================================================

def load_fraud_report():

    print_header("LOADING STAGE 5 FRAUD RESULTS")

    path = find_fraud_file()

    if path is None:
        print("ERROR: Fraud detection report not found.")
        print()
        print("Expected location:")
        print(os.path.join(
            FRAUD_DIR,
            "resume_fraud_report.csv"
        ))
        sys.exit(1)

    print("Fraud report found:")
    print(path)

    try:
        fraud_df = pd.read_csv(path)
    except Exception as e:
        print(f"ERROR loading fraud report: {e}")
        sys.exit(1)

    print(f"\nResumes in fraud report: {len(fraud_df)}")

    print("\nColumns detected:")

    for column in fraud_df.columns:
        print(f" - {column}")

    return fraud_df


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

def normalize_columns(df):

    df = df.copy()

    df.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in df.columns
    ]

    return df


# ============================================================
# IDENTIFY RESUME COLUMN
# ============================================================

def find_resume_column(df):

    possible_columns = [
        "filename",
        "resume",
        "resume_filename",
        "resume_file",
        "file_name"
    ]

    for col in possible_columns:
        if col in df.columns:
            return col

    return None


# ============================================================
# IDENTIFY FRAUD RISK COLUMN
# ============================================================

def find_risk_column(df):

    possible_columns = [
        "risk",
        "risk_level",
        "fraud_risk",
        "risk_category"
    ]

    for col in possible_columns:
        if col in df.columns:
            return col

    return None


# ============================================================
# IDENTIFY FRAUD SCORE COLUMN
# ============================================================

def find_fraud_score_column(df):

    possible_columns = [
        "fraud_score",
        "fraud_probability",
        "risk_score",
        "score"
    ]

    for col in possible_columns:
        if col in df.columns:
            return col

    return None


# ============================================================
# CLEAN RISK VALUE
# ============================================================

def clean_risk(value):

    if pd.isna(value):
        return "LOW"

    value = str(value).strip().upper()

    if "HIGH" in value:
        return "HIGH"

    if "MEDIUM" in value:
        return "MEDIUM"

    if "LOW" in value:
        return "LOW"

    return "LOW"


# ============================================================
# COMBINE STAGE 4 + STAGE 5
# ============================================================

def integrate_results(shortlist_df, fraud_df):

    print_header("INTEGRATING TOP-K + FRAUD DETECTION")

    shortlist_df = normalize_columns(shortlist_df)
    fraud_df = normalize_columns(fraud_df)

    shortlist_resume_col = find_resume_column(shortlist_df)
    fraud_resume_col = find_resume_column(fraud_df)

    if shortlist_resume_col is None:
        print("ERROR: Could not identify resume filename column")
        sys.exit(1)

    if fraud_resume_col is None:
        print("ERROR: Could not identify resume filename column")
        sys.exit(1)

    risk_col = find_risk_column(fraud_df)

    if risk_col is None:
        print("ERROR: Could not identify fraud risk column")
        sys.exit(1)

    fraud_score_col = find_fraud_score_column(fraud_df)

    print(f"Stage 4 resume column : {shortlist_resume_col}")
    print(f"Stage 5 resume column : {fraud_resume_col}")
    print(f"Fraud risk column     : {risk_col}")

    if fraud_score_col:
        print(f"Fraud score column    : {fraud_score_col}")

    # --------------------------------------------------------
    # Create lookup table
    # --------------------------------------------------------

    fraud_lookup = fraud_df[
        [fraud_resume_col, risk_col]
        + ([fraud_score_col] if fraud_score_col else [])
    ].copy()

    fraud_lookup = fraud_lookup.rename(
        columns={
            fraud_resume_col: "filename",
            risk_col: "fraud_risk"
        }
    )

    if fraud_score_col:
        fraud_lookup = fraud_lookup.rename(
            columns={
                fraud_score_col: "fraud_score"
            }
        )

    # Normalize filenames
    fraud_lookup["filename"] = (
        fraud_lookup["filename"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    shortlist_df["filename"] = (
        shortlist_df[shortlist_resume_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    final_df = shortlist_df.merge(
        fraud_lookup,
        on="filename",
        how="left"
    )

    # Missing fraud information = LOW
    final_df["fraud_risk"] = (
        final_df["fraud_risk"]
        .fillna("LOW")
        .apply(clean_risk)
    )

    if "fraud_score" not in final_df.columns:
        final_df["fraud_score"] = 0.0

    final_df["fraud_score"] = pd.to_numeric(
        final_df["fraud_score"],
        errors="coerce"
    ).fillna(0.0)

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    def final_decision(row):

        fraud_risk = row["fraud_risk"]

        if fraud_risk == "HIGH":
            return "FRAUD FLAGGED"

        if fraud_risk == "MEDIUM":
            return "MANUAL REVIEW"

        # LOW fraud risk:
        # Preserve Stage 4 decision

        if "decision" in row.index:
            return str(row["decision"]).upper()

        return "CONSIDER"

    final_df["final_decision"] = final_df.apply(
        final_decision,
        axis=1
    )

    # --------------------------------------------------------
    # Fraud explanation
    # --------------------------------------------------------

    def fraud_explanation(row):

        risk = row["fraud_risk"]

        if risk == "HIGH":
            return (
                "Resume contains strong indicators of hidden or "
                "suspicious text. Candidate is blocked from automatic "
                "shortlisting and requires fraud review."
            )

        if risk == "MEDIUM":
            return (
                "Resume contains some suspicious formatting or hidden "
                "text indicators. Candidate requires manual verification "
                "before proceeding."
            )

        return (
            "No significant hidden-text fraud indicators were detected. "
            "Candidate proceeds using the NLP matching decision."
        )

    final_df["fraud_explanation"] = final_df.apply(
        fraud_explanation,
        axis=1
    )

    return final_df


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(df):

    print_header("FINAL SCREENING RESULTS")

    for _, row in df.iterrows():

        print_line()

        filename = row.get(
            "filename",
            "Unknown"
        )

        print(f"Resume       : {filename}")

        if "category" in row:
            print(f"Category     : {row['category']}")

        if "rank" in row:
            print(f"Rank         : {row['rank']}")

        if "final_score" in row:

            try:
                score = float(row["final_score"])

                # Scores from Stage 4 are stored as decimal values
                if score <= 1:
                    score_display = score * 100
                else:
                    score_display = score

                print(
                    f"Match Score  : {score_display:.2f}%"
                )

            except Exception:
                print(
                    f"Match Score  : {row['final_score']}"
                )

        if "decision" in row:
            print(
                f"Original Decision : "
                f"{str(row['decision']).upper()}"
            )

        print(
            f"Fraud Risk   : "
            f"{row['fraud_risk']}"
        )

        try:
            fraud_score = float(row["fraud_score"])

            if fraud_score <= 1:
                fraud_score *= 100

            print(
                f"Fraud Score  : "
                f"{fraud_score:.2f}%"
            )

        except Exception:
            pass

        print(
            f"FINAL DECISION : "
            f"{row['final_decision']}"
        )

        print("\nFraud Explanation:")
        print(row["fraud_explanation"])

    print_line()


# ============================================================
# SUMMARY
# ============================================================

def display_summary(df):

    print_header("STAGE 6 FINAL SUMMARY")

    total = len(df)

    fraud_flagged = (
        df["final_decision"]
        == "FRAUD FLAGGED"
    ).sum()

    manual_review = (
        df["final_decision"]
        == "MANUAL REVIEW"
    ).sum()

    strong_shortlist = (
        df["final_decision"]
        == "STRONG SHORTLIST"
    ).sum()

    shortlist = (
        df["final_decision"]
        == "SHORTLIST"
    ).sum()

    consider = (
        df["final_decision"]
        == "CONSIDER"
    ).sum()

    reject = (
        df["final_decision"]
        == "REJECT"
    ).sum()

    print(f"Top-K candidates       : {total}")
    print(f"Fraud flagged          : {fraud_flagged}")
    print(f"Manual review          : {manual_review}")
    print(f"Strong shortlist       : {strong_shortlist}")
    print(f"Shortlist              : {shortlist}")
    print(f"Consider               : {consider}")
    print(f"Reject                 : {reject}")

    print()

    if fraud_flagged > 0:

        print(
            "IMPORTANT:"
        )

        print(
            f"{fraud_flagged} candidate(s) were removed "
            "from automatic consideration because of "
            "high fraud risk."
        )

    else:

        print(
            "✓ No high-risk resumes were present in Top-K."
        )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(df):

    print_header("SAVING FINAL SCREENING RESULTS")

    output_path = os.path.join(
        OUTPUT_DIR,
        "final_screening_results.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("Final screening file saved to:")
    print(output_path)

    # --------------------------------------------------------
    # Save clean final shortlist
    # --------------------------------------------------------

    clean_df = df[
        ~df["final_decision"].isin(
            [
                "FRAUD FLAGGED",
                "MANUAL REVIEW",
                "REJECT"
            ]
        )
    ].copy()

    clean_path = os.path.join(
        OUTPUT_DIR,
        "final_clean_shortlist.csv"
    )

    clean_df.to_csv(
        clean_path,
        index=False
    )

    print("\nClean shortlist saved to:")
    print(clean_path)

    # --------------------------------------------------------
    # Save fraud flagged candidates
    # --------------------------------------------------------

    flagged_df = df[
        df["final_decision"]
        == "FRAUD FLAGGED"
    ].copy()

    flagged_path = os.path.join(
        OUTPUT_DIR,
        "final_fraud_flagged.csv"
    )

    flagged_df.to_csv(
        flagged_path,
        index=False
    )

    print("\nFraud flagged candidates saved to:")
    print(flagged_path)

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "TALENTLENS\n"
        "NLP-ONLY RECRUITMENT SYSTEM\n"
        "STAGE 6 - FINAL SCREENING"
    )

    print(
        "\nCombining:"
    )

    print(
        "Stage 4 → Top-K candidate ranking"
    )

    print(
        "Stage 5 → Hidden-text fraud detection"
    )

    print(
        "Stage 6 → Final trustworthy screening"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    shortlist_df = load_shortlist()

    fraud_df = load_fraud_report()

    # --------------------------------------------------------
    # Integrate
    # --------------------------------------------------------

    final_df = integrate_results(
        shortlist_df,
        fraud_df
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_results(final_df)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    display_summary(final_df)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = save_results(
        final_df
    )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print_header(
        "STAGE 6 COMPLETED SUCCESSFULLY"
    )

    print(
        "Final NLP recruitment pipeline now supports:"
    )

    print("✓ TF-IDF similarity")
    print("✓ Cosine similarity")
    print("✓ Skill matching")
    print("✓ Experience matching")
    print("✓ Qualification matching")
    print("✓ Candidate ranking")
    print("✓ Top-K shortlisting")
    print("✓ Score-based candidate decisions")
    print("✓ Deterministic explanations")
    print("✓ Hidden-text fraud detection")
    print("✓ Fraud-aware final screening")
    print("✓ Automatic fraud blocking")

    print("\nGenerated:")
    print(f"✓ {output_path}")

    print("\nPipeline:")
    print(
        "JD + Resumes"
        " → NLP Matching"
        " → Ranking"
        " → Top-K"
        " → Fraud Detection"
        " → Final Screening"
    )

    print(
        "\nNext stage:"
    )

    print(
        "Build the final TalentLens interface/dashboard."
    )


if __name__ == "__main__":
    main()