import io
import re
import json
from pathlib import Path

import pandas as pd
import streamlit as st  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# TALENTLENS - UPDATED STREAMLIT APP
# ============================================================
# Features:
# 1. Upload multiple PDF resumes
# 2. Enter JD as text OR upload a JD text/PDF file
# 3. Live NLP screening
# 4. Matched and missing skills are explicitly displayed
# 5. Individual candidate score is shown as both 0.xx and xx.xx%
# 6. Hidden-text / white-text fraud detection
# 7. Top-K ranking
# 8. Working What-If Candidate Simulator
# 9. Existing Stage-6 results can also be viewed
#
# Run:
#     streamlit run src/app.py
#
# Required:
#     pip install streamlit pandas scikit-learn pymupdf
# ============================================================

try:
    import pymupdf as fitz
except ImportError:
    import fitz


# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="TalentLens",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "DATA"
RESUME_DIR = DATA_DIR / "resumes"
FINAL_DIR = DATA_DIR / "final_screening"
FRAUD_DIR = DATA_DIR / "fraud_detection"
SHORTLIST_DIR = DATA_DIR / "shortlist_results"

FINAL_RESULTS = FINAL_DIR / "final_screening_results.csv"
STAGE4_RESULTS = SHORTLIST_DIR / "software_engineer_frontend_developer_top_k_shortlist.csv"
FRAUD_RESULTS = FRAUD_DIR / "resume_fraud_report.csv"


# ------------------------------------------------------------
# SKILL VOCABULARY
# ------------------------------------------------------------
# This is deliberately a broad vocabulary. The system only
# uses skills that are actually present in the JD.
SKILLS = [
    # Programming
    "python", "java", "c", "c++", "c#", "javascript", "typescript",
    "r", "go", "golang", "kotlin", "swift", "scala", "ruby", "php",

    # Web
    "html", "css", "react", "reactjs", "angular", "vue", "vue.js",
    "node", "node.js", "express", "next.js", "bootstrap", "tailwind",
    "jquery", "rest api", "rest", "graphql",

    # Data / ML / AI
    "machine learning", "deep learning", "artificial intelligence",
    "natural language processing", "nlp", "computer vision",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
    "pandas", "numpy", "opencv", "transformers", "hugging face",
    "llm", "large language model", "generative ai", "genai",
    "rag", "retrieval augmented generation", "bert", "gpt",
    "xgboost", "lightgbm", "catboost", "cnn", "rnn", "lstm",
    "spacy", "nltk", "whisper",

    # Data
    "sql", "mysql", "postgresql", "postgres", "oracle", "mongodb",
    "nosql", "redis", "sqlite", "pl/sql", "plsql",
    "data analysis", "data analytics", "data science",
    "statistics", "power bi", "tableau", "excel",

    # Cloud / DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "jenkins", "git", "github", "gitlab", "ci/cd", "cicd",
    "terraform", "linux", "bash",

    # Software engineering
    "oop", "object oriented programming", "data structures",
    "algorithms", "dsa", "system design", "microservices",
    ".net", ".net core", "spring", "spring boot",
    "flask", "django", "fastapi", "api development",
    "unit testing", "testing",

    # Other common recruitment skills
    "agile", "scrum", "jira", "communication", "problem solving",
]


# ------------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------------
def normalize(text):
    text = str(text or "").lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_skill(skill):
    return normalize(skill).strip()


# ------------------------------------------------------------
# PDF EXTRACTION
# ------------------------------------------------------------
def extract_pdf_text(pdf_bytes):
    """Extract visible text from a PDF."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
        return "\n".join(pages).strip()
    except Exception as exc:
        return f"[PDF extraction error: {exc}]"


def extract_pdf_fraud(pdf_bytes):
    """
    Detect likely hidden/white/tiny text.

    This is a heuristic fraud detector, not a legal/authenticity verifier.
    """
    result = {
        "white_text_chars": 0,
        "tiny_text_chars": 0,
        "transparent_text_chars": 0,
        "suspicious_text_chars": 0,
        "hidden_keywords": [],
        "fraud_score": 0.0,
        "risk": "LOW",
        "reasons": [],
    }

    suspicious_parts = []

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page in doc:
            data = page.get_text("dict")

            for block in data.get("blocks", []):
                if block.get("type") != 0:
                    continue

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = str(span.get("text", "")).strip()
                        if not text:
                            continue

                        chars = len(text)
                        size = float(span.get("size", 10) or 10)
                        color = int(span.get("color", 0) or 0)

                        # PyMuPDF color integer is usually RGB packed as 0xRRGGBB.
                        r = (color >> 16) & 255
                        g = (color >> 8) & 255
                        b = color & 255

                        is_white = r >= 245 and g >= 245 and b >= 245
                        is_tiny = size <= 4.0

                        if is_white:
                            result["white_text_chars"] += chars
                            suspicious_parts.append(text)

                        if is_tiny:
                            result["tiny_text_chars"] += chars
                            suspicious_parts.append(text)

                        # Some PDFs encode transparent/invisible text
                        # through opacity/rendering tricks that are not
                        # always exposed uniformly by PyMuPDF. We therefore
                        # use white/tiny + text ratio as the main signal.
                        if is_white or is_tiny:
                            result["suspicious_text_chars"] += chars

        doc.close()

        suspicious_text = normalize(" ".join(suspicious_parts))

        keyword_list = [
            s for s in SKILLS
            if re.search(r"(?<!\w)" + re.escape(clean_skill(s)) + r"(?!\w)", suspicious_text)
        ]

        result["hidden_keywords"] = sorted(set(keyword_list))

        # Score calculation
        score = 0.0
        if result["white_text_chars"] > 0:
            score += 0.45
        if result["tiny_text_chars"] > 0:
            score += 0.20

        # Suspicious text containing many technical recruitment keywords
        if result["hidden_keywords"]:
            score += 0.30

        total_suspicious = result["suspicious_text_chars"]

        if total_suspicious >= 500:
            score += 0.15
        elif total_suspicious >= 200:
            score += 0.10
        elif total_suspicious >= 50:
            score += 0.05

        result["fraud_score"] = min(score, 1.0)

        reasons = []
        if result["white_text_chars"] > 0:
            reasons.append("white/near-white text detected")
        if result["tiny_text_chars"] > 0:
            reasons.append("tiny text detected")
        if result["hidden_keywords"]:
            reasons.append("technical/recruitment keywords found in suspicious text")
        if total_suspicious >= 200:
            reasons.append("high suspicious-text volume")

        result["reasons"] = reasons

        if result["fraud_score"] >= 0.70:
            result["risk"] = "HIGH"
        elif result["fraud_score"] >= 0.35:
            result["risk"] = "MEDIUM"
        else:
            result["risk"] = "LOW"

    except Exception as exc:
        result["risk"] = "LOW"
        result["reasons"] = [f"Fraud scan error: {exc}"]

    return result


# ------------------------------------------------------------
# SKILL EXTRACTION
# ------------------------------------------------------------
def skill_in_text(skill, text):
    """
    Robust matching for skills.
    Handles punctuation such as C++, C#, .NET, Node.js, etc.
    """
    t = normalize(text)
    s = clean_skill(skill)

    # Special cases
    if s in {"c++", "c#", ".net", "node.js", "next.js", "vue.js", "pl/sql"}:
        return s in t

    # Multi-word phrases and normal words
    pattern = r"(?<![a-z0-9])" + re.escape(s) + r"(?![a-z0-9])"
    return re.search(pattern, t) is not None


def extract_skills(text):
    found = []
    for skill in SKILLS:
        if skill_in_text(skill, text):
            found.append(skill)
    return sorted(set(found))


# ------------------------------------------------------------
# EXPERIENCE EXTRACTION
# ------------------------------------------------------------
def extract_years_experience(text):
    t = normalize(text)

    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)",
        r"experience\s*(?:of)?\s*(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
    ]

    values = []
    for pattern in patterns:
        for match in re.findall(pattern, t):
            try:
                values.append(float(match))
            except ValueError:
                pass

    return max(values) if values else 0.0


def extract_required_experience(jd_text):
    return extract_years_experience(jd_text)


# ------------------------------------------------------------
# QUALIFICATION EXTRACTION
# ------------------------------------------------------------
DEGREES = [
    "phd", "doctorate", "m.tech", "mtech", "m.e", "me",
    "mba", "mca", "msc", "m.sc", "master",
    "b.tech", "btech", "b.e", "be", "bca", "bsc", "b.sc",
    "bachelor", "degree", "diploma",
]


def extract_qualifications(text):
    t = normalize(text)
    return [d for d in DEGREES if d in t]


def qualification_match(resume_text, jd_text):
    jd_q = set(extract_qualifications(jd_text))
    resume_q = set(extract_qualifications(resume_text))

    if not jd_q:
        return 1.0, [], []

    matched = sorted(jd_q.intersection(resume_q))
    missing = sorted(jd_q.difference(resume_q))

    return len(matched) / len(jd_q), matched, missing


# ------------------------------------------------------------
# TF-IDF
# ------------------------------------------------------------
def tfidf_score(jd_text, resume_text):
    jd = normalize(jd_text)
    resume = normalize(resume_text)

    if not jd or not resume:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000,
        )
        matrix = vectorizer.fit_transform([jd, resume])
        score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return float(score)
    except Exception:
        return 0.0


# ------------------------------------------------------------
# SCORING
# ------------------------------------------------------------
def calculate_candidate(jd_text, resume_text, filename, pdf_bytes):
    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)

    matched = sorted(set(jd_skills).intersection(resume_skills))
    missing = sorted(set(jd_skills).difference(resume_skills))

    if jd_skills:
        skill_match = len(matched) / len(jd_skills)
    else:
        skill_match = 0.0

    required_exp = extract_required_experience(jd_text)
    candidate_exp = extract_years_experience(resume_text)

    if required_exp <= 0:
        experience_match = 1.0
    else:
        experience_match = min(candidate_exp / required_exp, 1.0)

    qual_score, qual_matched, qual_missing = qualification_match(
        resume_text, jd_text
    )

    text_score = tfidf_score(jd_text, resume_text)

    # Same scoring philosophy used by the NLP screening stage:
    # 25% textual similarity
    # 35% skills
    # 20% experience
    # 20% qualification
    final_score = (
        0.25 * text_score
        + 0.35 * skill_match
        + 0.20 * experience_match
        + 0.20 * qual_score
    )

    fraud = extract_pdf_fraud(pdf_bytes)

    if fraud["risk"] == "HIGH":
        decision = "FRAUD BLOCKED"
    elif final_score >= 0.75:
        decision = "STRONG SHORTLIST"
    elif final_score >= 0.60:
        decision = "SHORTLIST"
    elif final_score >= 0.45:
        decision = "CONSIDER"
    else:
        decision = "REJECT"

    return {
        "filename": filename,
        "score": float(final_score),
        "score_percent": float(final_score * 100),
        "tfidf_score": text_score,
        "skill_match": skill_match,
        "experience_match": experience_match,
        "qualification_match": qual_score,
        "matched_skills": matched,
        "missing_skills": missing,
        "jd_skills": jd_skills,
        "resume_skills": resume_skills,
        "candidate_experience": candidate_exp,
        "required_experience": required_exp,
        "qualification_matched": qual_matched,
        "qualification_missing": qual_missing,
        "decision": decision,
        "fraud": fraud,
        "resume_text": resume_text,
    }


# ------------------------------------------------------------
# WHAT-IF SIMULATOR
# ------------------------------------------------------------
def simulate_candidate(candidate, added_skills, simulated_experience):
    """
    Simulate adding missing skills and changing experience.

    IMPORTANT:
    We calculate the score from the same scoring formula as the
    real screening stage, so the What-If result actually changes.
    """
    jd_skills = set(candidate["jd_skills"])
    current_matched = set(candidate["matched_skills"])

    added = {
        clean_skill(x)
        for x in added_skills
        if clean_skill(x)
    }

    simulated_matched = current_matched.union(added)
    simulated_matched = simulated_matched.intersection(jd_skills)

    if jd_skills:
        sim_skill = len(simulated_matched) / len(jd_skills)
    else:
        sim_skill = 0.0

    required_exp = candidate["required_experience"]

    if required_exp <= 0:
        sim_exp = 1.0
    else:
        sim_exp = min(max(float(simulated_experience), 0.0) / required_exp, 1.0)

    sim_qual = candidate["qualification_match"]
    sim_tfidf = candidate["tfidf_score"]

    sim_score = (
        0.25 * sim_tfidf
        + 0.35 * sim_skill
        + 0.20 * sim_exp
        + 0.20 * sim_qual
    )

    if candidate["fraud"]["risk"] == "HIGH":
        decision = "FRAUD BLOCKED"
    elif sim_score >= 0.75:
        decision = "STRONG SHORTLIST"
    elif sim_score >= 0.60:
        decision = "SHORTLIST"
    elif sim_score >= 0.45:
        decision = "CONSIDER"
    else:
        decision = "REJECT"

    return {
        "score": sim_score,
        "skill_match": sim_skill,
        "experience_match": sim_exp,
        "decision": decision,
        "matched_skills": sorted(simulated_matched),
        "missing_skills": sorted(jd_skills.difference(simulated_matched)),
    }


# ------------------------------------------------------------
# EXISTING CSV LOADING
# ------------------------------------------------------------
def parse_list_value(value):
    """Convert CSV list-like values to a Python list."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "[]"}:
        return []

    # Try JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    text = text.strip("[](){}")
    text = text.replace("'", "").replace('"', "")

    parts = re.split(r"[,;|]", text)
    return [p.strip() for p in parts if p.strip()]


def load_existing_results():
    if not FINAL_RESULTS.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(FINAL_RESULTS)

        # Normalize common column names
        rename_map = {}
        for col in df.columns:
            low = col.lower().strip()

            if low in {"resume", "resume_file", "file"}:
                rename_map[col] = "filename"
            elif low in {"final_score", "match_score", "score"}:
                rename_map[col] = "score"
            elif low == "risk":
                rename_map[col] = "risk"

        df = df.rename(columns=rename_map)

        if "score" in df.columns:
            df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)

            # If CSV stores percentage 81.17, convert to 0.8117.
            if df["score"].max() > 1:
                df["score"] = df["score"] / 100.0

        return df
    except Exception:
        return pd.DataFrame()


# ------------------------------------------------------------
# UI HELPERS
# ------------------------------------------------------------
def score_text(value):
    value = float(value)
    return f"{value:.3f}  |  {value * 100:.2f}%"


def show_skill_list(title, skills, empty_message):
    st.markdown(f"### {title}")

    if skills:
        cols = st.columns(min(4, max(1, len(skills))))
        for i, skill in enumerate(skills):
            cols[i % len(cols)].success(f"✓ {skill}")
    else:
        st.info(empty_message)


def decision_badge(decision):
    if decision == "STRONG SHORTLIST":
        st.success(decision)
    elif decision == "SHORTLIST":
        st.success(decision)
    elif decision == "CONSIDER":
        st.warning(decision)
    elif decision == "FRAUD BLOCKED":
        st.error(decision)
    else:
        st.error(decision)


# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
if "live_results" not in st.session_state:
    st.session_state.live_results = []

if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.title("🎯 TalentLens")
    st.caption("NLP Recruitment Intelligence")

    st.markdown("---")
    st.markdown("### Pipeline")
    st.markdown(
        """
        1. JD + Resume Upload
        2. PDF Text Extraction
        3. Skill Extraction
        4. TF-IDF Similarity
        5. Experience Matching
        6. Qualification Matching
        7. Fraud Detection
        8. Ranking / Top-K
        9. What-If Simulation
        """
    )

    st.markdown("---")
    st.info(
        "The app does not need the resumes to be preloaded. "
        "Upload the resumes and paste the JD, then click Run Screening."
    )


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.title("🎯 TalentLens")
st.subheader("AI-Powered Resume Screening & Recruitment Intelligence")

st.markdown(
    "Upload resumes, provide the job description, and run the complete "
    "NLP screening pipeline."
)


# ------------------------------------------------------------
# JD + RESUME INPUT
# ------------------------------------------------------------
st.header("1. Job Description")

jd_tab1, jd_tab2 = st.tabs(["📝 Enter JD as Text", "📄 Upload JD File"])

with jd_tab1:
    jd_text = st.text_area(
        "Paste the Job Description here",
        value=st.session_state.jd_text,
        height=240,
        placeholder=(
            "Example:\n"
            "We are looking for a Python developer with 2 years of experience. "
            "Required skills: Python, SQL, Machine Learning, Docker, AWS. "
            "Bachelor degree required."
        ),
        key="jd_text_area",
    )

with jd_tab2:
    jd_file = st.file_uploader(
        "Upload JD PDF or TXT",
        type=["pdf", "txt"],
        key="jd_file",
    )

    if jd_file is not None:
        if jd_file.name.lower().endswith(".pdf"):
            jd_text_uploaded = extract_pdf_text(jd_file.getvalue())
        else:
            jd_text_uploaded = jd_file.getvalue().decode(
                "utf-8", errors="ignore"
            )

        st.text_area(
            "Extracted JD",
            value=jd_text_uploaded,
            height=220,
            key="uploaded_jd_preview",
        )

        if st.button("Use Uploaded JD"):
            st.session_state.jd_text = jd_text_uploaded
            st.rerun()


# Use typed JD first. If empty, use stored/uploaded JD.
if jd_text.strip():
    active_jd = jd_text.strip()
else:
    active_jd = st.session_state.jd_text.strip()

st.header("2. Upload Resumes")

uploaded_resumes = st.file_uploader(
    "Upload one or more resume PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    key="resume_uploader",
)

if uploaded_resumes:
    st.success(f"{len(uploaded_resumes)} resume(s) uploaded.")

    st.session_state.uploaded_names = [
        f.name for f in uploaded_resumes
    ]

    with st.expander("Uploaded resumes", expanded=False):
        for f in uploaded_resumes:
            st.write(f"📄 {f.name}")


# ------------------------------------------------------------
# RUN SCREENING
# ------------------------------------------------------------
st.header("3. Run Screening")

run_col1, run_col2 = st.columns([1, 1])

with run_col1:
    run_screening = st.button(
        "🚀 Run Resume Screening",
        type="primary",
        use_container_width=True,
    )

with run_col2:
    clear_results = st.button(
        "🗑️ Clear Current Results",
        use_container_width=True,
    )

if clear_results:
    st.session_state.live_results = []
    st.rerun()


if run_screening:
    if not active_jd:
        st.error("Please enter or upload a Job Description first.")
    elif not uploaded_resumes:
        st.error("Please upload at least one resume PDF.")
    else:
        st.session_state.jd_text = active_jd

        progress = st.progress(0)
        status = st.empty()

        results = []

        for i, resume_file in enumerate(uploaded_resumes):
            status.write(
                f"Processing {i + 1}/{len(uploaded_resumes)}: "
                f"{resume_file.name}"
            )

            pdf_bytes = resume_file.getvalue()
            resume_text = extract_pdf_text(pdf_bytes)

            candidate = calculate_candidate(
                active_jd,
                resume_text,
                resume_file.name,
                pdf_bytes,
            )

            results.append(candidate)
            progress.progress((i + 1) / len(uploaded_resumes))

        results.sort(key=lambda x: x["score"], reverse=True)

        for rank, candidate in enumerate(results, start=1):
            candidate["rank"] = rank

        st.session_state.live_results = results

        status.success(
            f"Completed screening for {len(results)} resume(s)."
        )

        st.rerun()


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------
results = st.session_state.live_results

if results:
    st.markdown("---")
    st.header("4. Screening Results")

    strong_count = sum(
        x["decision"] == "STRONG SHORTLIST" for x in results
    )
    shortlist_count = sum(
        x["decision"] == "SHORTLIST" for x in results
    )
    consider_count = sum(
        x["decision"] == "CONSIDER" for x in results
    )
    reject_count = sum(
        x["decision"] == "REJECT" for x in results
    )
    fraud_count = sum(
        x["fraud"]["risk"] == "HIGH" for x in results
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Resumes", len(results))
    c2.metric("Strong Shortlist", strong_count)
    c3.metric("Shortlist", shortlist_count)
    c4.metric("Consider", consider_count)
    c5.metric("Fraud Blocked", fraud_count)

    # Results table
    table_rows = []

    for x in results:
        table_rows.append(
            {
                "Rank": x["rank"],
                "Resume": x["filename"],
                "Score": f"{x['score']:.3f}",
                "Score %": f"{x['score'] * 100:.2f}%",
                "Skills": f"{x['skill_match'] * 100:.2f}%",
                "Experience": f"{x['experience_match'] * 100:.2f}%",
                "Qualification": f"{x['qualification_match'] * 100:.2f}%",
                "Fraud": x["fraud"]["risk"],
                "Decision": x["decision"],
            }
        )

    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.header("5. Candidate Analysis")

    candidate_labels = [
        f"#{x['rank']} — {x['filename']} — {x['score'] * 100:.2f}%"
        for x in results
    ]

    selected_label = st.selectbox(
        "Select a candidate",
        candidate_labels,
        key="candidate_selector",
    )

    selected_index = candidate_labels.index(selected_label)
    candidate = results[selected_index]

    st.subheader(candidate["filename"])

    # Main score
    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Overall Match",
        f"{candidate['score'] * 100:.2f}%",
        help=f"Normalized score: {candidate['score']:.3f}",
    )

    m2.metric(
        "Skill Match",
        f"{candidate['skill_match'] * 100:.2f}%",
    )

    fraud_delta = candidate["fraud"]["fraud_score"] * 100
    m3.metric(
        "Fraud Risk",
        f"{fraud_delta:.2f}%",
    )

    m4, m5, m6 = st.columns(3)

    m4.metric(
        "TF-IDF Similarity",
        f"{candidate['tfidf_score'] * 100:.2f}%",
    )

    m5.metric(
        "Experience Match",
        f"{candidate['experience_match'] * 100:.2f}%",
    )

    m6.metric(
        "Qualification Match",
        f"{candidate['qualification_match'] * 100:.2f}%",
    )

    st.caption(
        f"Candidate score: **{candidate['score']:.3f}** "
        f"(**{candidate['score'] * 100:.2f}%**)"
    )

    decision_badge(candidate["decision"])

    # --------------------------------------------------------
    # MATCHED / MISSING SKILLS
    # --------------------------------------------------------
    st.markdown("---")

    skill_col1, skill_col2 = st.columns(2)

    with skill_col1:
        st.markdown("### ✅ Matched Skills")

        if candidate["matched_skills"]:
            st.success(
                f"{len(candidate['matched_skills'])} of "
                f"{len(candidate['jd_skills'])} JD skills matched."
            )

            for skill in candidate["matched_skills"]:
                st.markdown(f"**✓ {skill}**")
        else:
            st.warning(
                "No matched skills were detected from the JD vocabulary."
            )

    with skill_col2:
        st.markdown("### ❌ Missing Skills")

        if candidate["missing_skills"]:
            st.error(
                f"{len(candidate['missing_skills'])} JD skill(s) missing."
            )

            for skill in candidate["missing_skills"]:
                st.markdown(f"**✗ {skill}**")
        else:
            st.success("No required skills are missing.")

    # Explicit skill diagnostic
    with st.expander("🔎 Skill Matching Diagnostic", expanded=False):
        st.write("**JD skills detected:**")
        st.write(candidate["jd_skills"] or "None")

        st.write("**Resume skills detected:**")
        st.write(candidate["resume_skills"] or "None")

        st.write("**Matched:**")
        st.write(candidate["matched_skills"] or "None")

        st.write("**Missing:**")
        st.write(candidate["missing_skills"] or "None")

        if not candidate["jd_skills"]:
            st.warning(
                "The JD did not contain any skills from the current "
                "skill vocabulary. Add recognizable technical skills "
                "such as Python, SQL, AWS, React, Docker, etc."
            )

    # --------------------------------------------------------
    # WHY THIS SCORE
    # --------------------------------------------------------
    st.markdown("---")
    st.markdown("### 💡 Why this score?")

    explanation = (
        f"The candidate has a final NLP score of "
        f"{candidate['score'] * 100:.2f}%. "
        f"TF-IDF similarity contributes "
        f"{candidate['tfidf_score'] * 100:.2f}%, "
        f"skill matching is {candidate['skill_match'] * 100:.2f}%, "
        f"experience matching is {candidate['experience_match'] * 100:.2f}%, "
        f"and qualification matching is "
        f"{candidate['qualification_match'] * 100:.2f}%."
    )

    st.info(explanation)

    # --------------------------------------------------------
    # EXPERIENCE DETAILS
    # --------------------------------------------------------
    st.markdown("### 🧑‍💼 Experience")

    e1, e2 = st.columns(2)
    e1.metric(
        "Candidate Experience",
        f"{candidate['candidate_experience']:.1f} years",
    )
    e2.metric(
        "Required Experience",
        f"{candidate['required_experience']:.1f} years",
    )

    # --------------------------------------------------------
    # FRAUD DETAILS
    # --------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🛡️ Fraud Detection")

    fraud = candidate["fraud"]

    f1, f2, f3 = st.columns(3)

    f1.metric(
        "Risk",
        fraud["risk"],
    )

    f2.metric(
        "Fraud Score",
        f"{fraud['fraud_score'] * 100:.2f}%",
    )

    f3.metric(
        "Suspicious Characters",
        fraud["suspicious_text_chars"],
    )

    if fraud["risk"] == "HIGH":
        st.error(
            "This resume is blocked because hidden-text indicators "
            "were detected."
        )
    elif fraud["risk"] == "MEDIUM":
        st.warning(
            "This resume requires manual review because suspicious "
            "text indicators were detected."
        )
    else:
        st.success("No significant hidden-text fraud indicators detected.")

    if fraud["reasons"]:
        st.write("**Detection reasons:**")
        for reason in fraud["reasons"]:
            st.write(f"- {reason}")

    if fraud["hidden_keywords"]:
        st.write("**Keywords found in suspicious text:**")
        st.write(", ".join(fraud["hidden_keywords"]))

    # --------------------------------------------------------
    # WHAT-IF SIMULATOR
    # --------------------------------------------------------
    st.markdown("---")
    st.header("🔬 What-If Candidate Simulator")

    st.write(
        "Simulate how the candidate's score changes if they gain "
        "missing skills or additional experience."
    )

    missing_skills = candidate["missing_skills"]

    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        st.markdown("#### Add Missing Skills")

        if missing_skills:
            selected_missing = st.multiselect(
                "Select skills to add",
                options=missing_skills,
                default=[],
                key=f"whatif_missing_{candidate['filename']}",
            )
        else:
            st.success(
                "The candidate already matches all detected JD skills."
            )
            selected_missing = []

        # This solves the previous bug where the What-If box
        # had "No options to select".
        custom_skill_text = st.text_input(
            "Or add skills manually",
            placeholder="Example: AWS, Docker, Kubernetes",
            key=f"whatif_custom_{candidate['filename']}",
        )

    with sim_col2:
        st.markdown("#### Simulated Experience")

        required_exp = candidate["required_experience"]
        current_exp = candidate["candidate_experience"]

        max_exp = max(10.0, required_exp + 5.0, current_exp + 5.0)

        simulated_exp = st.number_input(
            "Years of experience",
            min_value=0.0,
            max_value=float(max_exp),
            value=float(current_exp),
            step=0.5,
            key=f"whatif_exp_{candidate['filename']}",
        )

    custom_skills = []
    if custom_skill_text.strip():
        custom_skills = [
            x.strip()
            for x in re.split(r"[,;]", custom_skill_text)
            if x.strip()
        ]

    all_added_skills = selected_missing + custom_skills

    simulation = simulate_candidate(
        candidate,
        all_added_skills,
        simulated_exp,
    )

    st.markdown("### Simulation Result")

    s1, s2, s3 = st.columns(3)

    s1.metric(
        "Current Score",
        f"{candidate['score'] * 100:.2f}%",
        help=f"{candidate['score']:.3f}",
    )

    delta = simulation["score"] - candidate["score"]

    s2.metric(
        "What-If Score",
        f"{simulation['score'] * 100:.2f}%",
        delta=f"{delta * 100:+.2f}%",
    )

    s3.metric(
        "What-If Decision",
        simulation["decision"],
    )

    st.caption(
        f"Current score = **{candidate['score']:.3f}** | "
        f"What-If score = **{simulation['score']:.3f}**"
    )

    sim_m1, sim_m2 = st.columns(2)

    sim_m1.metric(
        "Simulated Skill Match",
        f"{simulation['skill_match'] * 100:.2f}%",
    )

    sim_m2.metric(
        "Simulated Experience Match",
        f"{simulation['experience_match'] * 100:.2f}%",
    )

    st.markdown("#### Simulated Skill State")

    sc1, sc2 = st.columns(2)

    with sc1:
        st.write("**Matched after simulation**")
        if simulation["matched_skills"]:
            for skill in simulation["matched_skills"]:
                st.success(f"✓ {skill}")
        else:
            st.info("No matched skills.")

    with sc2:
        st.write("**Still missing after simulation**")
        if simulation["missing_skills"]:
            for skill in simulation["missing_skills"]:
                st.error(f"✗ {skill}")
        else:
            st.success("All detected JD skills are matched.")

    if abs(delta) < 0.0001:
        st.info(
            "The score did not change. To increase it, add a skill that "
            "is currently missing or increase experience when the JD "
            "requires more experience."
        )
    elif delta > 0:
        st.success(
            f"The candidate improves by {delta * 100:.2f} percentage points."
        )
    else:
        st.warning(
            f"The candidate decreases by {abs(delta) * 100:.2f} percentage points."
        )

    # --------------------------------------------------------
    # DOWNLOAD LIVE RESULTS
    # --------------------------------------------------------
    st.markdown("---")
    st.header("6. Export Results")

    export_rows = []

    for x in results:
        export_rows.append(
            {
                "rank": x["rank"],
                "filename": x["filename"],
                "score": round(x["score"], 4),
                "score_percent": round(x["score"] * 100, 2),
                "tfidf_score": round(x["tfidf_score"], 4),
                "skill_match": round(x["skill_match"], 4),
                "experience_match": round(x["experience_match"], 4),
                "qualification_match": round(x["qualification_match"], 4),
                "matched_skills": ", ".join(x["matched_skills"]),
                "missing_skills": ", ".join(x["missing_skills"]),
                "fraud_score": round(x["fraud"]["fraud_score"], 4),
                "fraud_risk": x["fraud"]["risk"],
                "decision": x["decision"],
            }
        )

    export_df = pd.DataFrame(export_rows)

    st.download_button(
        "⬇️ Download Screening CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="talentlens_live_screening.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ------------------------------------------------------------
# EXISTING STAGE 6 RESULTS
# ------------------------------------------------------------
st.markdown("---")
with st.expander("📂 View Previous Stage-6 Results", expanded=False):
    existing_df = load_existing_results()

    if existing_df.empty:
        st.info(
            "No existing Stage-6 result file was found. "
            "Run the pipeline or use the live upload screening above."
        )
    else:
        st.dataframe(
            existing_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            f"Loaded from: {FINAL_RESULTS}"
        )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.caption(
    "TalentLens | NLP-only recruitment screening | "
    "TF-IDF + skill matching + experience + qualification + fraud detection + What-If"
)