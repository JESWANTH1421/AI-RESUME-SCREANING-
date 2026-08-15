import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# TALENTLENS
# NLP-ONLY RECRUITMENT SYSTEM
# STAGE 3 - CORRECTED NLP MATCHING ENGINE
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DATA"

RESUME_FILE = DATA_DIR / "resume_text_processed.csv"
JD_FILE = DATA_DIR / "relevant_job_descriptions.csv"

RESULT_DIR = DATA_DIR / "matching_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# CONFIGURATION
# ==========================================================

TOP_K = 10

# Final score weights
TFIDF_WEIGHT = 0.30
SKILL_WEIGHT = 0.40
EXPERIENCE_WEIGHT = 0.20
QUALIFICATION_WEIGHT = 0.10


# ==========================================================
# SKILL NORMALIZATION
# ==========================================================

SKILL_ALIASES = {
    "python": ["python"],
    "java": ["java"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "csharp"],
    "html": ["html"],
    "css": ["css"],
    "react": ["react", "reactjs"],
    "angular": ["angular"],
    "vue": ["vue", "vuejs"],
    "node": ["node", "node.js", "nodejs"],
    "node.js": ["node.js", "nodejs"],
    "sql": ["sql"],
    "mysql": ["mysql"],
    "postgresql": ["postgresql", "postgres"],
    "oracle": ["oracle"],
    "mongodb": ["mongodb", "mongo"],
    "database": ["database", "databases"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "git": ["git"],
    "github": ["github"],
    "jenkins": ["jenkins"],
    "linux": ["linux"],
    "machine learning": ["machine learning"],
    "deep learning": ["deep learning"],
    "nlp": ["nlp", "natural language processing"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "spark": ["spark", "apache spark"],
    "hadoop": ["hadoop"],
    "tableau": ["tableau"],
    "power bi": ["power bi", "powerbi"],
    "excel": ["excel", "microsoft excel"],
    "selenium": ["selenium"],
    "rest api": ["rest api", "restful api", "rest"],
    "flask": ["flask"],
    "django": ["django"],
    "spring": ["spring", "spring boot"],
    "labview": ["labview"],
    "cad": ["cad"],
    "autocad": ["autocad"],
    "solidworks": ["solidworks"],
    "matlab": ["matlab"],
    "quality assurance": [
        "quality assurance",
        "qa",
        "quality testing"
    ],
}


def normalize_text(text):
    """
    Basic NLP normalization.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Normalize common symbols
    text = text.replace("node.js", "nodejs")
    text = text.replace("c sharp", "c#")
    text = text.replace("c plus plus", "c++")

    # Keep useful technical symbols
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================================
# SKILL EXTRACTION
# ==========================================================

def extract_skills(text):
    """
    Extract normalized skills from text.
    """

    text = normalize_text(text)

    found_skills = set()

    for canonical_skill, aliases in SKILL_ALIASES.items():

        for alias in aliases:

            alias_normalized = normalize_text(alias)

            if not alias_normalized:
                continue

            # Word-boundary matching for safer extraction
            pattern = r"(?<![a-z0-9])" + re.escape(alias_normalized) + r"(?![a-z0-9])"

            if re.search(pattern, text):
                found_skills.add(canonical_skill)
                break

    return sorted(found_skills)


# ==========================================================
# EXPERIENCE PARSING
# ==========================================================

def parse_experience(experience_text):
    """
    Converts experience strings such as:
    '2 to 14 Years'
    into minimum and maximum experience.
    """

    if pd.isna(experience_text):
        return None, None

    text = str(experience_text).lower()

    numbers = re.findall(r"\d+(?:\.\d+)?", text)

    if not numbers:
        return None, None

    numbers = [float(x) for x in numbers]

    if len(numbers) >= 2:
        return numbers[0], numbers[1]

    return numbers[0], numbers[0]


def experience_match(candidate_text, jd_experience):
    """
    Estimates how well a candidate's experience matches the JD.

    Because resume datasets do not always contain a clean
    'years of experience' field, this function uses
    detected year ranges where available.
    """

    jd_min, jd_max = parse_experience(jd_experience)

    if jd_min is None:
        return 0.5

    candidate_years = extract_candidate_experience(candidate_text)

    if candidate_years is None:
        # Unknown experience → neutral score
        return 0.5

    if candidate_years >= jd_min:
        return 1.0

    if candidate_years >= jd_min * 0.75:
        return 0.8

    if candidate_years >= jd_min * 0.5:
        return 0.5

    return 0.2


def extract_candidate_experience(text):
    """
    Attempts to estimate total years of experience from resume text.

    Examples:
    '2015 to 2020'
    '2018 - 2024'
    '5 years experience'
    """

    if not text:
        return None

    text = str(text).lower()

    # Explicit "X years" patterns
    explicit_matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?experience",
        text
    )

    if explicit_matches:
        values = [float(x) for x in explicit_matches]
        return max(values)

    # Detect employment year ranges
    year_ranges = re.findall(
        r"(20\d{2})\s*(?:to|-|–)\s*(20\d{2}|current|present)",
        text
    )

    durations = []

    current_year = 2026

    for start, end in year_ranges:

        try:
            start_year = int(start)

            if end in ["current", "present"]:
                end_year = current_year
            else:
                end_year = int(end)

            if end_year >= start_year:
                durations.append(end_year - start_year)

        except Exception:
            continue

    if durations:
        # This is only an approximation.
        # Avoid summing overlapping employment periods.
        return max(durations)

    return None


# ==========================================================
# QUALIFICATION MATCHING
# ==========================================================

QUALIFICATION_GROUPS = {
    "phd": {
        "phd",
        "doctorate",
        "doctoral",
    },

    "masters": {
        "m.tech",
        "mtech",
        "mca",
        "mba",
        "m.com",
        "mcom",
        "master",
        "masters",
        "master's",
    },

    "bachelors": {
        "b.tech",
        "btech",
        "bca",
        "bba",
        "b.com",
        "bcom",
        "ba",
        "bachelor",
        "bachelors",
        "bachelor's",
    },

    "diploma": {
        "diploma",
        "associate",
        "associates",
    },
}


def normalize_qualification(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = text.replace("’", "'")

    return text.strip()


def get_qualification_level(qualification):
    """
    Maps qualification into a broader education level.
    """

    qualification = normalize_qualification(qualification)

    if not qualification:
        return None

    if any(x in qualification for x in QUALIFICATION_GROUPS["phd"]):
        return "phd"

    if any(x in qualification for x in QUALIFICATION_GROUPS["masters"]):
        return "masters"

    if any(x in qualification for x in QUALIFICATION_GROUPS["bachelors"]):
        return "bachelors"

    if any(x in qualification for x in QUALIFICATION_GROUPS["diploma"]):
        return "diploma"

    return "other"


def qualification_match(candidate_text, jd_qualification):
    """
    Qualification matching based on broad education levels.

    Scoring:
    1.0 = exact / equivalent
    0.8 = candidate higher than required
    0.5 = partially related
    0.0 = clearly unrelated
    """

    jd_level = get_qualification_level(jd_qualification)

    if jd_level is None:
        return 0.5

    candidate_text = normalize_text(candidate_text)

    candidate_levels = []

    # Detect qualifications in resume
    for level, keywords in QUALIFICATION_GROUPS.items():

        for keyword in keywords:

            keyword_normalized = normalize_text(keyword)

            if keyword_normalized in candidate_text:
                candidate_levels.append(level)
                break

    if not candidate_levels:
        return 0.0

    level_rank = {
        "diploma": 1,
        "bachelors": 2,
        "masters": 3,
        "phd": 4,
        "other": 0,
    }

    required_rank = level_rank.get(jd_level, 0)

    best_score = 0.0

    for candidate_level in candidate_levels:

        candidate_rank = level_rank.get(candidate_level, 0)

        if candidate_level == jd_level:
            score = 1.0

        elif candidate_rank > required_rank:
            score = 0.8

        elif candidate_rank == required_rank - 1:
            score = 0.5

        else:
            score = 0.0

        best_score = max(best_score, score)

    return best_score


# ==========================================================
# SKILL MATCHING
# ==========================================================

def calculate_skill_match(resume_skills, jd_skills):

    resume_set = set(resume_skills)
    jd_set = set(jd_skills)

    if not jd_set:
        return 0.5, [], []

    matched = sorted(resume_set.intersection(jd_set))
    missing = sorted(jd_set - resume_set)

    score = len(matched) / len(jd_set)

    return score, matched, missing


# ==========================================================
# LOAD DATA
# ==========================================================

def load_data():

    print("\n")
    print("=" * 60)
    print("LOADING NLP DATA")
    print("=" * 60)

    if not RESUME_FILE.exists():
        print("\nERROR: Resume processed file not found:")
        print(RESUME_FILE)
        return None, None

    if not JD_FILE.exists():
        print("\nERROR: Relevant JD file not found:")
        print(JD_FILE)
        return None, None

    print("\nLoading resumes...")

    resume_df = pd.read_csv(
        RESUME_FILE,
        low_memory=False
    )

    print(f"Resumes: {len(resume_df):,}")

    print("\nLoading relevant job descriptions...")

    jd_df = pd.read_csv(
        JD_FILE,
        low_memory=False
    )

    print(f"Relevant JDs: {len(jd_df):,}")

    return resume_df, jd_df


# ==========================================================
# JOB TITLE SELECTION
# ==========================================================

def select_job_title(jd_df):

    print("\n")
    print("=" * 60)
    print("JOB TITLE SELECTION")
    print("=" * 60)

    title_counts = (
        jd_df["Job Title"]
        .fillna("")
        .value_counts()
        .head(30)
    )

    print("\nTop available job titles:\n")

    titles = list(title_counts.index)

    for i, title in enumerate(titles, 1):
        print(
            f"{i:2d}. {title} "
            f"({title_counts[title]:,})"
        )

    while True:

        user_input = input(
            "\nEnter job title or number: "
        ).strip()

        if user_input.isdigit():

            index = int(user_input)

            if 1 <= index <= len(titles):
                selected_title = titles[index - 1]
                break

        else:

            matching_titles = [
                title
                for title in jd_df["Job Title"].dropna().unique()
                if user_input.lower() in str(title).lower()
            ]

            if matching_titles:

                if len(matching_titles) == 1:
                    selected_title = matching_titles[0]
                    break

                print("\nMatching titles:")

                for i, title in enumerate(
                    matching_titles[:20],
                    1
                ):
                    print(f"{i}. {title}")

                try:
                    choice = int(
                        input(
                            "\nSelect title number: "
                        )
                    )

                    if 1 <= choice <= min(
                        20,
                        len(matching_titles)
                    ):
                        selected_title = matching_titles[
                            choice - 1
                        ]
                        break

                except ValueError:
                    pass

        print(
            "\nInvalid selection. "
            "Please try again."
        )

    return selected_title


# ==========================================================
# ROLE / JD SELECTION
# ==========================================================

def select_exact_jd(jd_df, selected_title):

    title_df = jd_df[
        jd_df["Job Title"].astype(str).str.lower()
        == selected_title.lower()
    ].copy()

    print("\n")
    print("=" * 60)
    print("EXACT JOB DESCRIPTION SELECTION")
    print("=" * 60)

    print(
        f"\nJob Title: {selected_title}"
    )

    print(
        f"Matching JDs found: {len(title_df):,}"
    )

    # Remove duplicate JDs where possible
    columns_for_duplicate_check = [
        "Job Title",
        "Role",
        "Experience",
        "Qualifications",
        "skills",
        "Job Description",
    ]

    available_columns = [
        c for c in columns_for_duplicate_check
        if c in title_df.columns
    ]

    title_df = title_df.drop_duplicates(
        subset=available_columns
    ).reset_index(drop=True)

    print(
        f"Unique JDs available: {len(title_df):,}"
    )

    # Show roles first
    role_counts = (
        title_df["Role"]
        .fillna("Unknown")
        .value_counts()
        .head(20)
    )

    print("\nAvailable roles:\n")

    roles = list(role_counts.index)

    for i, role in enumerate(roles, 1):
        print(
            f"{i:2d}. {role} "
            f"({role_counts[role]:,})"
        )

    while True:

        user_input = input(
            "\nEnter role or number: "
        ).strip()

        selected_role = None

        if user_input.isdigit():

            index = int(user_input)

            if 1 <= index <= len(roles):
                selected_role = roles[index - 1]

        else:

            matching_roles = [
                role
                for role in roles
                if user_input.lower() in str(role).lower()
            ]

            if len(matching_roles) == 1:
                selected_role = matching_roles[0]

            elif len(matching_roles) > 1:

                print("\nMatching roles:")

                for i, role in enumerate(
                    matching_roles,
                    1
                ):
                    print(f"{i}. {role}")

                try:
                    choice = int(
                        input("\nSelect role number: ")
                    )

                    if 1 <= choice <= len(matching_roles):
                        selected_role = matching_roles[
                            choice - 1
                        ]

                except ValueError:
                    pass

        if selected_role:
            break

        print(
            "\nInvalid selection. "
            "Please try again."
        )

    role_df = title_df[
        title_df["Role"].fillna("Unknown")
        .astype(str)
        .str.lower()
        == selected_role.lower()
    ].copy()

    print(
        f"\nJDs found for role "
        f"'{selected_role}': {len(role_df):,}"
    )

    # Show a small number of unique JD choices
    preview_columns = [
        "Experience",
        "Qualifications",
        "skills",
        "Job Description",
    ]

    preview_columns = [
        c for c in preview_columns
        if c in role_df.columns
    ]

    display_df = role_df[
        preview_columns
    ].drop_duplicates().head(10).reset_index()

    if len(display_df) == 0:
        print("\nNo detailed JD found.")
        return None

    print("\nAvailable JD variants:\n")

    for i in range(len(display_df)):

        row = display_df.iloc[i]

        print("-" * 60)
        print(f"JD OPTION {i + 1}")
        print(
            f"Experience: "
            f"{row.get('Experience', 'N/A')}"
        )
        print(
            f"Qualification: "
            f"{row.get('Qualifications', 'N/A')}"
        )
        print(
            f"Skills: "
            f"{str(row.get('skills', ''))[:250]}"
        )

        description = str(
            row.get("Job Description", "")
        )

        print(
            f"Description: "
            f"{description[:300]}"
        )

    while True:

        try:

            choice = int(
                input(
                    f"\nSelect JD option "
                    f"(1-{len(display_df)}): "
                )
            )

            if 1 <= choice <= len(display_df):
                break

        except ValueError:
            pass

        print("Invalid selection.")

    selected_preview = display_df.iloc[
        choice - 1
    ]

    # Match the selected variant back to the actual row
    selected_row = None

    for _, row in role_df.iterrows():

        same = True

        for column in preview_columns:

            a = str(row.get(column, ""))
            b = str(selected_preview.get(column, ""))

            if a != b:
                same = False
                break

        if same:
            selected_row = row
            break

    if selected_row is None:
        selected_row = role_df.iloc[
            selected_preview["index"]
        ]

    return selected_row


# ==========================================================
# BUILD JD TEXT
# ==========================================================

def build_jd_text(jd_row):

    fields = [
        "Job Title",
        "Role",
        "Experience",
        "Qualifications",
        "skills",
        "Responsibilities",
        "Job Description",
    ]

    parts = []

    for field in fields:

        if field in jd_row.index:

            value = jd_row[field]

            if not pd.isna(value):
                parts.append(str(value))

    return " ".join(parts)


# ==========================================================
# PREPARE RESUMES
# ==========================================================

def prepare_resumes(resume_df):

    print("\n")
    print("=" * 60)
    print("PREPARING RESUME DATA")
    print("=" * 60)

    resume_df = resume_df.copy()

    # Ensure text column exists
    if "resume_text_clean" in resume_df.columns:

        resume_df["nlp_text"] = (
            resume_df["resume_text_clean"]
            .fillna("")
            .astype(str)
        )

    elif "resume_text" in resume_df.columns:

        resume_df["nlp_text"] = (
            resume_df["resume_text"]
            .fillna("")
            .astype(str)
        )

    elif "cleaned_text" in resume_df.columns:

        resume_df["nlp_text"] = (
            resume_df["cleaned_text"]
            .fillna("")
            .astype(str)
        )

    else:

        raise ValueError(
            "No resume text column found."
        )

    print("Extracting resume skills...")

    if "skills" not in resume_df.columns:

        resume_df["skills"] = resume_df[
            "nlp_text"
        ].apply(extract_skills)

    else:

        # Existing skills may be strings such as:
        # "python, java, sql"
        resume_df["skills"] = resume_df[
            "skills"
        ].apply(
            lambda x: (
                extract_skills(x)
                if not isinstance(x, list)
                else x
            )
        )

    return resume_df


# ==========================================================
# MATCHING ENGINE
# ==========================================================

def perform_matching(resume_df, jd_row):

    print("\n")
    print("=" * 60)
    print("NLP CANDIDATE MATCHING")
    print("=" * 60)

    jd_text = build_jd_text(jd_row)

    jd_skills = extract_skills(
        " ".join(
            [
                str(jd_row.get("skills", "")),
                str(jd_row.get("Job Description", "")),
                str(jd_row.get("Responsibilities", "")),
            ]
        )
    )

    print("\nJD skills:")
    print(
        ", ".join(jd_skills)
        if jd_skills
        else "No recognized skills"
    )

    # ------------------------------------------------------
    # TF-IDF
    # ------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("BUILDING TF-IDF MODEL")
    print("=" * 60)

    documents = (
        [jd_text]
        + resume_df["nlp_text"].tolist()
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(
        documents
    )

    jd_vector = tfidf_matrix[0]

    resume_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(
        jd_vector,
        resume_vectors
    ).flatten()

    print("TF-IDF similarity calculated.")

    # ------------------------------------------------------
    # Candidate scoring
    # ------------------------------------------------------

    results = []

    jd_experience = jd_row.get(
        "Experience",
        ""
    )

    jd_qualification = jd_row.get(
        "Qualifications",
        ""
    )

    for index, row in resume_df.iterrows():

        resume_text = row["nlp_text"]

        resume_skills = row["skills"]

        # Skill matching
        skill_score, matched, missing = (
            calculate_skill_match(
                resume_skills,
                jd_skills
            )
        )

        # Experience matching
        exp_score = experience_match(
            resume_text,
            jd_experience
        )

        # Qualification matching
        qualification_score = (
            qualification_match(
                resume_text,
                jd_qualification
            )
        )

        # TF-IDF
        tfidf_score = similarities[index]

        # Final score
        final_score = (
            TFIDF_WEIGHT * tfidf_score
            +
            SKILL_WEIGHT * skill_score
            +
            EXPERIENCE_WEIGHT * exp_score
            +
            QUALIFICATION_WEIGHT * qualification_score
        )

        results.append(
            {
                "resume_id": row.get(
                    "resume_id",
                    index + 1
                ),

                "filename": row.get(
                    "filename",
                    ""
                ),

                "category": row.get(
                    "category",
                    ""
                ),

                "final_score": final_score,

                "tfidf_score": tfidf_score,

                "skill_score": skill_score,

                "experience_score": exp_score,

                "qualification_score": (
                    qualification_score
                ),

                "matched_skills": ", ".join(
                    matched
                ),

                "missing_skills": ", ".join(
                    missing
                ),

                "matched_skill_count": len(
                    matched
                ),

                "required_skill_count": len(
                    jd_skills
                ),
            }
        )

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        "final_score",
        ascending=False
    ).reset_index(drop=True)

    result_df["rank"] = (
        result_df.index + 1
    )

    return result_df, jd_skills


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

def display_results(
    result_df,
    jd_row,
    jd_skills
):

    print("\n")
    print("=" * 60)
    print("TOP CANDIDATES")
    print("=" * 60)

    print(
        f"\nJob: "
        f"{jd_row.get('Job Title', '')}"
    )

    print(
        f"Role: "
        f"{jd_row.get('Role', '')}"
    )

    print(
        f"Experience: "
        f"{jd_row.get('Experience', '')}"
    )

    print(
        f"Qualification: "
        f"{jd_row.get('Qualifications', '')}"
    )

    print("\n")

    top_results = result_df.head(TOP_K)

    for _, candidate in top_results.iterrows():

        print("-" * 60)

        print(
            f"Rank: {int(candidate['rank'])}"
        )

        print(
            f"Resume: "
            f"{candidate['filename']}"
        )

        print(
            f"Category: "
            f"{candidate['category']}"
        )

        print(
            f"Final Score: "
            f"{candidate['final_score'] * 100:.2f}%"
        )

        print(
            f"TF-IDF: "
            f"{candidate['tfidf_score'] * 100:.2f}%"
        )

        print(
            f"Skill Match: "
            f"{candidate['skill_score'] * 100:.2f}%"
        )

        print(
            f"Experience: "
            f"{candidate['experience_score'] * 100:.2f}%"
        )

        print(
            f"Qualification: "
            f"{candidate['qualification_score'] * 100:.2f}%"
        )

        matched = candidate[
            "matched_skills"
        ]

        missing = candidate[
            "missing_skills"
        ]

        print(
            f"Matched Skills: "
            f"{matched if matched else 'None'}"
        )

        print(
            f"Missing Skills: "
            f"{missing if missing else 'None'}"
        )


# ==========================================================
# SAVE RESULTS
# ==========================================================

def save_results(
    result_df,
    jd_row
):

    job_title = str(
        jd_row.get(
            "Job Title",
            "job"
        )
    )

    role = str(
        jd_row.get(
            "Role",
            "role"
        )
    )

    filename = (
        job_title
        + "_"
        + role
    )

    filename = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        filename
    ).lower()

    output_file = (
        RESULT_DIR
        / f"{filename}_ranking.csv"
    )

    result_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    print("\n")
    print("=" * 60)
    print("RESULTS SAVED")
    print("=" * 60)

    print(
        f"\nOutput file:\n{output_file}"
    )

    return output_file


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("\n")
    print("=" * 60)
    print("TALENTLENS")
    print("NLP-ONLY RECRUITMENT SYSTEM")
    print("STAGE 3 - CORRECTED NLP MATCHING ENGINE")
    print("=" * 60)

    # ------------------------------------------------------
    # Load data
    # ------------------------------------------------------

    resume_df, jd_df = load_data()

    if resume_df is None:
        return

    if jd_df is None:
        return

    # ------------------------------------------------------
    # Prepare resumes
    # ------------------------------------------------------

    resume_df = prepare_resumes(
        resume_df
    )

    # ------------------------------------------------------
    # Select job title
    # ------------------------------------------------------

    selected_title = select_job_title(
        jd_df
    )

    # ------------------------------------------------------
    # Select exact JD
    # ------------------------------------------------------

    jd_row = select_exact_jd(
        jd_df,
        selected_title
    )

    if jd_row is None:
        print(
            "\nCould not select a JD."
        )
        return

    # ------------------------------------------------------
    # Show selected JD
    # ------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("SELECTED JOB DESCRIPTION")
    print("=" * 60)

    print(
        f"\nJob Title: "
        f"{jd_row.get('Job Title', '')}"
    )

    print(
        f"Role: "
        f"{jd_row.get('Role', '')}"
    )

    print(
        f"Experience: "
        f"{jd_row.get('Experience', '')}"
    )

    print(
        f"Qualification: "
        f"{jd_row.get('Qualifications', '')}"
    )

    print(
        f"Skills: "
        f"{jd_row.get('skills', '')}"
    )

    # ------------------------------------------------------
    # Matching
    # ------------------------------------------------------

    result_df, jd_skills = perform_matching(
        resume_df,
        jd_row
    )

    # ------------------------------------------------------
    # Display
    # ------------------------------------------------------

    display_results(
        result_df,
        jd_row,
        jd_skills
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    save_results(
        result_df,
        jd_row
    )

    # ------------------------------------------------------
    # Complete
    # ------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("STAGE 3 CORRECTION COMPLETED")
    print("=" * 60)

    print(
        "\nNLP pipeline supports:"
    )

    print("✓ Exact Job Title selection")
    print("✓ Exact Role selection")
    print("✓ Exact JD variant selection")
    print("✓ TF-IDF")
    print("✓ Cosine similarity")
    print("✓ Skill matching")
    print("✓ Experience matching")
    print("✓ Improved qualification matching")
    print("✓ Candidate ranking")

    print("\nNext stage:")
    print("Top-K shortlist + candidate explanation")


if __name__ == "__main__":
    main()