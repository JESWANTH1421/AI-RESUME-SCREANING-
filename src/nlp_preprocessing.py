import re
from pathlib import Path

import pandas as pd


# ==========================================================
# TALENTLENS
# NLP-ONLY RECRUITMENT SYSTEM
#
# STAGE 2:
# FAST NLP PREPROCESSING + JD FILTERING
# ==========================================================


BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "DATA"

RESUME_FILE = DATA_DIR / "resume_text.csv"
JD_FILE = DATA_DIR / "job_descriptions_processed.csv"

RESUME_OUTPUT = DATA_DIR / "resume_text_processed.csv"
JD_OUTPUT = DATA_DIR / "relevant_job_descriptions.csv"


# ==========================================================
# RELEVANT JOB TITLE / ROLE KEYWORDS
# ==========================================================

JOB_KEYWORDS = [
    "software",
    "developer",
    "programmer",
    "data",
    "machine learning",
    "artificial intelligence",
    "ai",
    "network",
    "database",
    "systems",
    "system",
    "cyber",
    "security",
    "cloud",
    "devops",
    "qa",
    "quality assurance",
    "tester",
    "testing",
    "web",
    "frontend",
    "backend",
    "full stack",
    "ui",
    "ux",
    "technical",
    "technology",
    "information technology",
    "it ",
    "engineer",
    "engineering",
    "automation",
    "embedded",
    "electronics",
    "electrical",
    "mechanical",
    "aerospace",
    "civil",
    "architect",
    "analyst",
    "scientist",
    "administrator",
    "database",
]


# ==========================================================
# SKILLS WE WANT TO IDENTIFY
# ==========================================================

SKILLS = [
    "python",
    "java",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "html",
    "css",
    "react",
    "angular",
    "vue",
    "node.js",
    "node",
    "sql",
    "mysql",
    "postgresql",
    "oracle",
    "mongodb",
    "database",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data science",
    "data analyst",
    "data analysis",
    "natural language processing",
    "nlp",
    "computer vision",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "aws",
    "azure",
    "gcp",
    "google cloud",
    "docker",
    "kubernetes",
    "jenkins",
    "devops",
    "git",
    "github",
    "networking",
    "network security",
    "cybersecurity",
    "firewall",
    "routing",
    "tcp/ip",
    "selenium",
    "automation testing",
    "software testing",
    "quality assurance",
    "matlab",
    "labview",
    "autocad",
    "solidworks",
    "cad",
]


# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove email
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Normalize separators
    text = text.replace("/", " ")
    text = text.replace("|", " ")
    text = text.replace(",", " ")
    text = text.replace(";", " ")

    # Remove unusual characters
    text = re.sub(
        r"[^a-z0-9+#.\s-]",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================================
# FAST SKILL EXTRACTION
# ==========================================================

def extract_skills(text):

    if not text:
        return ""

    text = text.lower()

    found = []

    for skill in SKILLS:

        skill_lower = skill.lower()

        if skill_lower in text:

            found.append(skill)

    return ", ".join(
        sorted(set(found))
    )


# ==========================================================
# PROCESS RESUMES
# ==========================================================

def process_resumes():

    print("\n")
    print("=" * 60)
    print("RESUME NLP PREPROCESSING")
    print("=" * 60)

    if not RESUME_FILE.exists():

        print(
            f"ERROR: {RESUME_FILE} not found."
        )

        return None

    df = pd.read_csv(
        RESUME_FILE
    )

    print(
        f"Resumes loaded: {len(df)}"
    )

    # Clean resume text
    print(
        "\nCleaning resume text..."
    )

    df["clean_text"] = (
        df["resume_text"]
        .fillna("")
        .map(clean_text)
    )

    # Skill extraction on only 238 resumes
    print(
        "Extracting resume skills..."
    )

    df["extracted_skills"] = (
        df["clean_text"]
        .map(extract_skills)
    )

    df.to_csv(
        RESUME_OUTPUT,
        index=False,
        encoding="utf-8"
    )

    print(
        f"\nSaved: {RESUME_OUTPUT}"
    )

    return df


# ==========================================================
# FILTER JOB DESCRIPTIONS
# ==========================================================

def filter_job_descriptions():

    print("\n")
    print("=" * 60)
    print("FAST JOB DESCRIPTION FILTERING")
    print("=" * 60)

    if not JD_FILE.exists():

        print(
            f"ERROR: {JD_FILE} not found."
        )

        return None

    columns = [
        "Job Id",
        "Experience",
        "Qualifications",
        "Job Title",
        "Role",
        "Job Description",
        "skills",
        "Responsibilities",
    ]

    print(
        "\nLoading required JD columns..."
    )

    df = pd.read_csv(
        JD_FILE,
        usecols=columns,
        low_memory=False
    )

    print(
        f"Total JDs: {len(df):,}"
    )

    # ------------------------------------------------------
    # Combine ONLY Job Title and Role first.
    # This is much faster than processing full JD text.
    # ------------------------------------------------------

    print(
        "\nFiltering by Job Title and Role..."
    )

    title_role = (
        df["Job Title"]
        .fillna("")
        .astype(str)
        .str.lower()
        + " "
        + df["Role"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    pattern = "|".join(
        re.escape(keyword.lower())
        for keyword in JOB_KEYWORDS
    )

    mask = title_role.str.contains(
        pattern,
        regex=True,
        na=False
    )

    relevant_df = df.loc[
        mask
    ].copy()

    print(
        f"\nRelevant JDs: {len(relevant_df):,}"
    )

    print(
        f"Removed JDs: "
        f"{len(df) - len(relevant_df):,}"
    )

    # ------------------------------------------------------
    # Now combine the smaller dataset
    # ------------------------------------------------------

    print(
        "\nCreating NLP text for relevant JDs..."
    )

    relevant_df["jd_text"] = (

        relevant_df[
            [
                "Job Title",
                "Role",
                "Job Description",
                "skills",
                "Responsibilities",
                "Qualifications",
            ]
        ]
        .fillna("")
        .astype(str)
        .agg(
            " ".join,
            axis=1
        )
    )

    # ------------------------------------------------------
    # Clean text
    # ------------------------------------------------------

    print(
        "Cleaning relevant JD text..."
    )

    relevant_df["clean_text"] = (
        relevant_df["jd_text"]
        .map(clean_text)
    )

    # ------------------------------------------------------
    # Skill extraction
    # ------------------------------------------------------

    print(
        "Extracting skills from relevant JDs..."
    )

    relevant_df["extracted_skills"] = (
        relevant_df["clean_text"]
        .map(extract_skills)
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    print(
        "\nSaving filtered JD dataset..."
    )

    relevant_df.to_csv(
        JD_OUTPUT,
        index=False,
        encoding="utf-8"
    )

    print(
        f"\nSaved: {JD_OUTPUT}"
    )

    return relevant_df


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

def show_results(
    resume_df,
    jd_df
):

    print("\n")
    print("=" * 60)
    print("STAGE 2 RESULTS")
    print("=" * 60)

    # ------------------------------------------------------
    # Resume examples
    # ------------------------------------------------------

    if resume_df is not None:

        print("\nRESUME SKILL EXAMPLES")

        for i in range(
            min(5, len(resume_df))
        ):

            print("\n-------------------------")

            print(
                "Resume:",
                resume_df.iloc[i]["filename"]
            )

            print(
                "Skills:",
                resume_df.iloc[i][
                    "extracted_skills"
                ]
            )

    # ------------------------------------------------------
    # JD examples
    # ------------------------------------------------------

    if jd_df is not None:

        print("\n")
        print("=" * 60)
        print("TOP RELEVANT JOB TITLES")
        print("=" * 60)

        print(
            jd_df[
                "Job Title"
            ]
            .value_counts()
            .head(20)
        )

        print("\nJD SKILL EXAMPLES")

        for i in range(
            min(5, len(jd_df))
        ):

            print("\n-------------------------")

            print(
                "Job:",
                jd_df.iloc[i][
                    "Job Title"
                ]
            )

            print(
                "Role:",
                jd_df.iloc[i][
                    "Role"
                ]
            )

            print(
                "Skills:",
                jd_df.iloc[i][
                    "extracted_skills"
                ]
            )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("\n")
    print("=" * 60)
    print("TALENTLENS")
    print("NLP-ONLY RECRUITMENT SYSTEM")
    print("STAGE 2 - FAST NLP PREPROCESSING")
    print("=" * 60)

    # ------------------------------------------------------
    # Process resumes
    # ------------------------------------------------------

    resume_df = process_resumes()

    # ------------------------------------------------------
    # Filter JDs
    # ------------------------------------------------------

    jd_df = filter_job_descriptions()

    # ------------------------------------------------------
    # Show results
    # ------------------------------------------------------

    show_results(
        resume_df,
        jd_df
    )

    # ------------------------------------------------------
    # Final
    # ------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("STAGE 2 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated files:")

    print(
        f"✓ {RESUME_OUTPUT}"
    )

    print(
        f"✓ {JD_OUTPUT}"
    )

    print("\nNext stage:")

    print(
        "TF-IDF + cosine similarity + skill matching"
    )


if __name__ == "__main__":

    main()