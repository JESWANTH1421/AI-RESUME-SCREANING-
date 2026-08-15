# ============================================================
# TALENTLENS
# NLP-ONLY RECRUITMENT SYSTEM
# STAGE 4 - TOP-K SHORTLIST + CANDIDATE EXPLANATION
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path.cwd()

MATCHING_DIR = BASE_DIR / "DATA" / "matching_results"
SHORTLIST_DIR = BASE_DIR / "DATA" / "shortlist_results"

SHORTLIST_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DISPLAY
# ============================================================

def print_header(title):
    print("\n")
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# FIND LATEST MATCHING FILE
# ============================================================

def find_matching_file():

    files = list(MATCHING_DIR.glob("*_ranking.csv"))

    if not files:
        print("\nERROR: No Stage 3 ranking file found.")
        print(f"Expected folder:\n{MATCHING_DIR}")
        return None

    # Most recently modified ranking file
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return files[0]


# ============================================================
# LOAD DATA
# ============================================================

def load_matching_results():

    print_header("LOADING STAGE 3 MATCHING RESULTS")

    ranking_file = find_matching_file()

    if ranking_file is None:
        return None, None

    print("\nRanking file found:")
    print(ranking_file)

    try:
        df = pd.read_csv(ranking_file)
    except Exception as e:
        print(f"\nERROR loading ranking file: {e}")
        return None, None

    print(f"\nCandidates loaded: {len(df)}")

    print("\nColumns detected:")

    for column in df.columns:
        print(f" - {column}")

    return df, ranking_file


# ============================================================
# NORMALIZE SCORES
# ============================================================

def normalize_score(value):
    """
    Converts a score into a standard 0-1 range.

    Stage 3 normally stores scores as:
        0.6238 -> 62.38%

    But this function also safely handles:
        62.38 -> 62.38%

    This prevents the Stage 4 percentage bug.
    """

    try:

        if pd.isna(value):
            return 0.0

        value = float(value)

        # If already represented as percentage
        if value > 1:
            value = value / 100.0

        # Keep between 0 and 1
        value = max(0.0, min(1.0, value))

        return value

    except Exception:
        return 0.0


# ============================================================
# NORMALIZE DATAFRAME SCORES
# ============================================================

def normalize_dataframe_scores(df):

    score_columns = [
        "final_score",
        "tfidf_score",
        "skill_score",
        "experience_score",
        "qualification_score"
    ]

    for column in score_columns:

        if column in df.columns:

            df[column] = df[column].apply(
                normalize_score
            )

    return df


# ============================================================
# SAFE TEXT
# ============================================================

def safe_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


# ============================================================
# SAFE SKILL LIST
# ============================================================

def parse_skills(value):

    text = safe_text(value)

    if not text:
        return []

    # Handle common separators
    text = text.replace(";", ",")

    skills = []

    for skill in text.split(","):

        skill = skill.strip().lower()

        if skill and skill not in skills:
            skills.append(skill)

    return skills


# ============================================================
# DECISION LOGIC
# ============================================================

def determine_decision(score):

    score = normalize_score(score)

    percentage = score * 100

    if percentage >= 70:
        return "STRONG SHORTLIST"

    elif percentage >= 55:
        return "SHORTLIST"

    elif percentage >= 40:
        return "CONSIDER"

    else:
        return "REJECT"


# ============================================================
# SCORE DESCRIPTION
# ============================================================

def describe_score(score):

    score = normalize_score(score)

    percentage = score * 100

    if percentage >= 70:
        return "strong"

    elif percentage >= 55:
        return "good"

    elif percentage >= 40:
        return "moderate"

    else:
        return "weak"


# ============================================================
# EXPERIENCE DESCRIPTION
# ============================================================

def describe_experience(score):

    score = normalize_score(score)

    percentage = score * 100

    if percentage >= 0.99 * 100:
        return "fully satisfies the experience requirement"

    elif percentage >= 0.75 * 100:
        return "mostly satisfies the experience requirement"

    elif percentage >= 0.50 * 100:
        return "partially satisfies the experience requirement"

    else:
        return "does not strongly satisfy the experience requirement"


# ============================================================
# QUALIFICATION DESCRIPTION
# ============================================================

def describe_qualification(score):

    score = normalize_score(score)

    percentage = score * 100

    if percentage >= 0.99 * 100:
        return "matches the required qualification"

    elif percentage >= 0.50 * 100:
        return "partially matches the required qualification"

    else:
        return "does not strongly match the required qualification"


# ============================================================
# BUILD EXPLANATION
# ============================================================

def build_explanation(row):

    final_score = normalize_score(
        row.get("final_score", 0)
    )

    tfidf_score = normalize_score(
        row.get("tfidf_score", 0)
    )

    skill_score = normalize_score(
        row.get("skill_score", 0)
    )

    experience_score = normalize_score(
        row.get("experience_score", 0)
    )

    qualification_score = normalize_score(
        row.get("qualification_score", 0)
    )

    matched_skills = parse_skills(
        row.get("matched_skills", "")
    )

    missing_skills = parse_skills(
        row.get("missing_skills", "")
    )

    matched_count = int(
        row.get("matched_skill_count", len(matched_skills))
        if not pd.isna(row.get("matched_skill_count", np.nan))
        else len(matched_skills)
    )

    required_count = int(
        row.get("required_skill_count", 0)
        if not pd.isna(row.get("required_skill_count", np.nan))
        else 0
    )

    decision = determine_decision(final_score)

    # --------------------------------------------------------
    # Overall assessment
    # --------------------------------------------------------

    overall_strength = describe_score(final_score)

    if decision == "STRONG SHORTLIST":

        explanation = (
            "The candidate is a strong match for the selected "
            "job description."
        )

    elif decision == "SHORTLIST":

        explanation = (
            "The candidate is a good match for the selected "
            "job description and should be considered for "
            "shortlisting."
        )

    elif decision == "CONSIDER":

        explanation = (
            "The candidate shows a moderate match with the "
            "selected job description and may require further "
            "review."
        )

    else:

        explanation = (
            "The candidate shows a relatively weak overall "
            "match with the selected job description."
        )

    # --------------------------------------------------------
    # Skill explanation
    # --------------------------------------------------------

    if required_count > 0:

        explanation += (
            f" The candidate matches {matched_count} of "
            f"{required_count} identified required skills."
        )

    elif matched_skills:

        explanation += (
            " The candidate has several relevant skills "
            "identified from the resume."
        )

    if matched_skills:

        explanation += (
            " Matched skills include: "
            + ", ".join(matched_skills)
            + "."
        )

    if missing_skills:

        explanation += (
            " Missing or unmatched skills include: "
            + ", ".join(missing_skills)
            + "."
        )

    # --------------------------------------------------------
    # Experience explanation
    # --------------------------------------------------------

    experience_description = describe_experience(
        experience_score
    )

    explanation += (
        " The candidate "
        + experience_description
        + "."
    )

    # --------------------------------------------------------
    # Qualification explanation
    # --------------------------------------------------------

    qualification_description = describe_qualification(
        qualification_score
    )

    explanation += (
        " The educational qualification "
        + qualification_description
        + "."
    )

    # --------------------------------------------------------
    # TF-IDF explanation
    # --------------------------------------------------------

    tfidf_percentage = tfidf_score * 100

    if tfidf_percentage >= 20:

        explanation += (
            " The resume has strong textual similarity "
            "with the job description."
        )

    elif tfidf_percentage >= 10:

        explanation += (
            " The resume has moderate textual similarity "
            "with the job description."
        )

    elif tfidf_percentage >= 5:

        explanation += (
            " The resume has some textual similarity "
            "with the job description."
        )

    else:

        explanation += (
            " Textual similarity with the job description "
            "is relatively low."
        )

    return explanation


# ============================================================
# CREATE SHORTLIST
# ============================================================

def create_shortlist(df, top_k):

    print_header("CREATING TOP-K SHORTLIST")

    total_candidates = len(df)

    print(f"\nAvailable candidates: {total_candidates}")
    print(f"Requested Top-K: {top_k}")

    if total_candidates == 0:

        print("\nERROR: No candidates available.")

        return None

    # Make sure top_k is valid
    top_k = max(1, min(top_k, total_candidates))

    # --------------------------------------------------------
    # Normalize scores before sorting
    # --------------------------------------------------------

    df = normalize_dataframe_scores(df)

    # --------------------------------------------------------
    # Sort by actual numerical final score
    # --------------------------------------------------------

    df = df.sort_values(
        by="final_score",
        ascending=False
    ).copy()

    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------

    df = df.reset_index(drop=True)

    # --------------------------------------------------------
    # Remove old rank if present
    #
    # This fixes:
    # ValueError: cannot insert rank, already exists
    # --------------------------------------------------------

    if "rank" in df.columns:

        df = df.drop(
            columns=["rank"]
        )

    # --------------------------------------------------------
    # Create new rank
    # --------------------------------------------------------

    df.insert(
        0,
        "rank",
        range(1, len(df) + 1)
    )

    # --------------------------------------------------------
    # Select Top-K
    # --------------------------------------------------------

    shortlist = df.head(top_k).copy()

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    shortlist["decision"] = shortlist[
        "final_score"
    ].apply(
        determine_decision
    )

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    shortlist["explanation"] = shortlist.apply(
        build_explanation,
        axis=1
    )

    print(
        f"\nTop {len(shortlist)} candidates selected."
    )

    return shortlist


# ============================================================
# DISPLAY CANDIDATE
# ============================================================

def display_candidate(row):

    print("\n" + "-" * 60)

    print(f"Rank: {int(row['rank'])}")

    print(
        f"Resume: {safe_text(row.get('filename', 'Unknown'))}"
    )

    print(
        f"Category: {safe_text(row.get('category', 'Unknown'))}"
    )

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    final_score = normalize_score(
        row.get("final_score", 0)
    ) * 100

    tfidf_score = normalize_score(
        row.get("tfidf_score", 0)
    ) * 100

    skill_score = normalize_score(
        row.get("skill_score", 0)
    ) * 100

    experience_score = normalize_score(
        row.get("experience_score", 0)
    ) * 100

    qualification_score = normalize_score(
        row.get("qualification_score", 0)
    ) * 100

    print(
        f"Final Score: {final_score:.2f}%"
    )

    print(
        f"Decision: {row['decision']}"
    )

    print(
        f"TF-IDF: {tfidf_score:.2f}%"
    )

    print(
        f"Skill Match: {skill_score:.2f}%"
    )

    print(
        f"Experience Match: {experience_score:.2f}%"
    )

    print(
        f"Qualification Match: "
        f"{qualification_score:.2f}%"
    )

    # --------------------------------------------------------
    # Skills
    # --------------------------------------------------------

    matched_skills = parse_skills(
        row.get("matched_skills", "")
    )

    missing_skills = parse_skills(
        row.get("missing_skills", "")
    )

    print(
        "Matched Skills: "
        + (
            ", ".join(matched_skills)
            if matched_skills
            else "None"
        )
    )

    print(
        "Missing Skills: "
        + (
            ", ".join(missing_skills)
            if missing_skills
            else "None"
        )
    )

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    print("\nEXPLANATION:")

    print(row["explanation"])


# ============================================================
# DISPLAY SUMMARY
# ============================================================

def display_summary(shortlist):

    print_header("STAGE 4 SUMMARY")

    total = len(shortlist)

    strong = (
        shortlist["decision"]
        == "STRONG SHORTLIST"
    ).sum()

    shortlisted = (
        shortlist["decision"]
        == "SHORTLIST"
    ).sum()

    consider = (
        shortlist["decision"]
        == "CONSIDER"
    ).sum()

    reject = (
        shortlist["decision"]
        == "REJECT"
    ).sum()

    average_score = (
        shortlist["final_score"]
        .mean()
        * 100
    )

    best_score = (
        shortlist["final_score"]
        .max()
        * 100
    )

    print(f"\nTop-K candidates: {total}")

    print(
        f"Strong shortlist: {strong}"
    )

    print(
        f"Shortlist: {shortlisted}"
    )

    print(
        f"Consider: {consider}"
    )

    print(
        f"Reject: {reject}"
    )

    print(
        f"\nAverage Top-K score: "
        f"{average_score:.2f}%"
    )

    print(
        f"Best candidate score: "
        f"{best_score:.2f}%"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_shortlist(shortlist, ranking_file):

    print_header("SAVING SHORTLIST RESULTS")

    # --------------------------------------------------------
    # Generate filename
    # --------------------------------------------------------

    original_name = ranking_file.stem

    output_name = (
        original_name
        .replace("_ranking", "")
        + "_top_k_shortlist.csv"
    )

    output_file = (
        SHORTLIST_DIR / output_name
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    try:

        shortlist.to_csv(
            output_file,
            index=False,
            encoding="utf-8"
        )

    except Exception as e:

        print(
            f"\nERROR saving shortlist: {e}"
        )

        return None

    print(
        "\nShortlist saved to:"
    )

    print(output_file)

    return output_file


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print(
        "TALENTLENS"
    )
    print(
        "NLP-ONLY RECRUITMENT SYSTEM"
    )
    print(
        "STAGE 4 - TOP-K SHORTLIST + EXPLANATION"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Load Stage 3
    # --------------------------------------------------------

    df, ranking_file = (
        load_matching_results()
    )

    if df is None:

        return

    # --------------------------------------------------------
    # Ask Top-K
    # --------------------------------------------------------

    print_header(
        "CREATING TOP-K SHORTLIST"
    )

    default_top_k = 10

    print(
        f"\nAvailable candidates: "
        f"{len(df)}"
    )

    print(
        f"Default Top-K: "
        f"{default_top_k}"
    )

    user_input = input(
        f"\nEnter Top-K value "
        f"(press Enter for {default_top_k}): "
    ).strip()

    if not user_input:

        top_k = default_top_k

    else:

        try:

            top_k = int(user_input)

        except ValueError:

            print(
                "\nInvalid Top-K value."
            )

            print(
                f"Using default: "
                f"{default_top_k}"
            )

            top_k = default_top_k

    # --------------------------------------------------------
    # Create shortlist
    # --------------------------------------------------------

    shortlist = create_shortlist(
        df,
        top_k
    )

    if shortlist is None:

        return

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print_header(
        "TOP-K SHORTLIST RESULTS"
    )

    for _, row in shortlist.iterrows():

        display_candidate(row)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = save_shortlist(
        shortlist,
        ranking_file
    )

    if output_file is None:

        return

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    display_summary(
        shortlist
    )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print_header(
        "STAGE 4 COMPLETED SUCCESSFULLY"
    )

    print("\nGenerated file:")

    print(
        f"✓ {output_file}"
    )

    print("\nNLP pipeline now supports:")

    print(
        "✓ TF-IDF similarity"
    )

    print(
        "✓ Cosine similarity"
    )

    print(
        "✓ Skill matching"
    )

    print(
        "✓ Experience matching"
    )

    print(
        "✓ Qualification matching"
    )

    print(
        "✓ Candidate ranking"
    )

    print(
        "✓ Top-K shortlisting"
    )

    print(
        "✓ Score-based candidate decisions"
    )

    print(
        "✓ Deterministic candidate explanation"
    )

    print("\nDecision thresholds:")

    print(
        "✓ 70%+  → STRONG SHORTLIST"
    )

    print(
        "✓ 55–69% → SHORTLIST"
    )

    print(
        "✓ 40–54% → CONSIDER"
    )

    print(
        "✓ <40%  → REJECT"
    )

    print("\nNext stage:")

    print(
        "White-space / hidden-text resume fraud detection"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()