import re
import math
from pathlib import Path

import fitz
import pandas as pd


# ==========================================================
# TALENTLENS
# NLP-ONLY RECRUITMENT SYSTEM
# STAGE 5 - WHITE-SPACE / HIDDEN-TEXT FRAUD DETECTION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "DATA"
RESUME_DIR = DATA_DIR / "resumes"

OUTPUT_DIR = DATA_DIR / "fraud_detection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "resume_fraud_report.csv"
CLEAN_RESUME_FILE = OUTPUT_DIR / "clean_resumes.csv"
FLAGGED_RESUME_FILE = OUTPUT_DIR / "flagged_resumes.csv"


# ==========================================================
# SETTINGS
# ==========================================================

# Text with very small font is suspicious.
SMALL_FONT_THRESHOLD = 4.0

# Near-white text threshold.
# RGB values close to 255 are considered white.
WHITE_COLOR_THRESHOLD = 245

# Extremely low alpha is suspicious.
LOW_ALPHA_THRESHOLD = 0.15

# If suspicious text is a significant percentage of all text,
# increase fraud risk.
SUSPICIOUS_TEXT_RATIO_HIGH = 0.20

# Keyword repetition threshold.
KEYWORD_REPETITION_THRESHOLD = 8

# Common recruitment / technical keywords.
COMMON_SKILLS = {
    "python", "java", "javascript", "typescript", "c", "c++", "c#",
    "html", "css", "react", "angular", "vue", "node", "node.js",
    "sql", "mysql", "postgresql", "mongodb", "oracle",
    "aws", "azure", "gcp", "docker", "kubernetes",
    "git", "github", "jenkins",
    "machine learning", "deep learning", "artificial intelligence",
    "nlp", "natural language processing",
    "tensorflow", "pytorch", "scikit-learn",
    "pandas", "numpy",
    "power bi", "tableau",
    "excel", "data analysis", "data science",
    "flask", "django", "spring", "spring boot",
    "rest api", "api",
    "linux", "unix",
    "devops", "ci/cd",
    "selenium", "testing", "quality assurance",
    "cloud", "database",
    "communication", "leadership",
}


# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================

def normalize_text(text):
    """Normalize extracted text."""

    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def rgb_from_int(color):
    """
    Convert PyMuPDF integer color representation
    to RGB values.
    """

    try:
        color = int(color)

        r = (color >> 16) & 255
        g = (color >> 8) & 255
        b = color & 255

        return r, g, b

    except Exception:
        return 0, 0, 0


def is_white_or_near_white(color):
    """Check whether text color is white or nearly white."""

    r, g, b = rgb_from_int(color)

    return (
        r >= WHITE_COLOR_THRESHOLD
        and g >= WHITE_COLOR_THRESHOLD
        and b >= WHITE_COLOR_THRESHOLD
    )


def extract_keywords(text):
    """Find known technical/recruitment keywords."""

    normalized = normalize_text(text)

    found = []

    for skill in COMMON_SKILLS:

        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, normalized):
            found.append(skill)

    return sorted(set(found))


def count_keyword_occurrences(text):
    """Count repeated skills."""

    normalized = normalize_text(text)

    counts = {}

    for skill in COMMON_SKILLS:

        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        count = len(re.findall(pattern, normalized))

        if count >= KEYWORD_REPETITION_THRESHOLD:
            counts[skill] = count

    return counts


# ==========================================================
# PDF ANALYSIS
# ==========================================================

def analyze_pdf(pdf_path):

    result = {
        "filename": pdf_path.name,
        "pages": 0,
        "total_text_blocks": 0,
        "visible_text_chars": 0,
        "suspicious_text_chars": 0,
        "white_text_chars": 0,
        "tiny_text_chars": 0,
        "transparent_text_chars": 0,
        "suspicious_blocks": 0,
        "white_text_blocks": 0,
        "tiny_text_blocks": 0,
        "transparent_text_blocks": 0,
        "hidden_keywords": "",
        "repeated_keywords": "",
        "suspicious_text_ratio": 0.0,
        "fraud_score": 0.0,
        "risk": "LOW",
        "reasons": "",
        "error": "",
    }

    try:

        doc = fitz.open(pdf_path)

        result["pages"] = len(doc)

        all_visible_text = []
        suspicious_text = []

        for page in doc:

            blocks = page.get_text("dict").get("blocks", [])

            for block in blocks:

                if "lines" not in block:
                    continue

                for line in block["lines"]:

                    for span in line["spans"]:

                        text = span.get("text", "")

                        if not text.strip():
                            continue

                        result["total_text_blocks"] += 1

                        text_length = len(text)

                        result["visible_text_chars"] += text_length

                        size = float(span.get("size", 0))

                        color = span.get("color", 0)

                        alpha = span.get("alpha", 255)

                        suspicious = False

                        # --------------------------------------------------
                        # WHITE / NEAR-WHITE TEXT
                        # --------------------------------------------------

                        if is_white_or_near_white(color):

                            result["white_text_chars"] += text_length
                            result["white_text_blocks"] += 1

                            suspicious = True

                            suspicious_text.append(text)

                        # --------------------------------------------------
                        # VERY SMALL TEXT
                        # --------------------------------------------------

                        if size > 0 and size <= SMALL_FONT_THRESHOLD:

                            result["tiny_text_chars"] += text_length
                            result["tiny_text_blocks"] += 1

                            suspicious = True

                            suspicious_text.append(text)

                        # --------------------------------------------------
                        # TRANSPARENT TEXT
                        # --------------------------------------------------

                        if alpha is not None:

                            try:
                                alpha_value = float(alpha)

                                # PyMuPDF can expose alpha as 0-255
                                # or normalized values depending on version.

                                if alpha_value <= 1:
                                    transparent = (
                                        alpha_value <= LOW_ALPHA_THRESHOLD
                                    )
                                else:
                                    transparent = (
                                        alpha_value <= 40
                                    )

                                if transparent:

                                    result[
                                        "transparent_text_chars"
                                    ] += text_length

                                    result[
                                        "transparent_text_blocks"
                                    ] += 1

                                    suspicious = True

                                    suspicious_text.append(text)

                            except Exception:
                                pass

                        if suspicious:

                            result["suspicious_blocks"] += 1

                        else:

                            all_visible_text.append(text)

        doc.close()

        suspicious_combined = " ".join(suspicious_text)

        result["suspicious_text_chars"] = len(
            suspicious_combined
        )

        # ------------------------------------------------------
        # SUSPICIOUS TEXT RATIO
        # ------------------------------------------------------

        if result["visible_text_chars"] > 0:

            result["suspicious_text_ratio"] = (
                result["suspicious_text_chars"]
                / result["visible_text_chars"]
            )

        # ------------------------------------------------------
        # HIDDEN KEYWORDS
        # ------------------------------------------------------

        result["hidden_keywords"] = ", ".join(
            extract_keywords(suspicious_combined)
        )

        # ------------------------------------------------------
        # REPEATED KEYWORDS
        # ------------------------------------------------------

        repeated = count_keyword_occurrences(
            suspicious_combined
        )

        result["repeated_keywords"] = ", ".join(
            f"{k} ({v})"
            for k, v in repeated.items()
        )

        # ======================================================
        # FRAUD SCORE
        # ======================================================

        score = 0
        reasons = []

        # White text
        if result["white_text_chars"] > 0:

            score += 40

            reasons.append(
                "white/near-white text detected"
            )

        # Tiny text
        if result["tiny_text_chars"] > 0:

            score += 20

            reasons.append(
                "very small text detected"
            )

        # Transparent text
        if result["transparent_text_chars"] > 0:

            score += 25

            reasons.append(
                "transparent text detected"
            )

        # Large suspicious ratio
        if (
            result["suspicious_text_ratio"]
            >= SUSPICIOUS_TEXT_RATIO_HIGH
        ):

            score += 15

            reasons.append(
                "high suspicious-text ratio"
            )

        # Hidden technical keywords
        if result["hidden_keywords"]:

            score += 20

            reasons.append(
                "technical/recruitment keywords found in suspicious text"
            )

        # Repeated keywords
        if result["repeated_keywords"]:

            score += 15

            reasons.append(
                "repeated technical keywords detected"
            )

        # Cap score
        score = min(score, 100)

        result["fraud_score"] = score

        # ======================================================
        # RISK LEVEL
        # ======================================================

        if score >= 70:

            result["risk"] = "HIGH"

        elif score >= 35:

            result["risk"] = "MEDIUM"

        else:

            result["risk"] = "LOW"

        result["reasons"] = "; ".join(reasons)

        return result

    except Exception as e:

        result["error"] = str(e)

        return result


# ==========================================================
# PROCESS ALL RESUMES
# ==========================================================

def process_all_resumes():

    print("\n" + "=" * 60)
    print("RESUME FRAUD DETECTION")
    print("=" * 60)

    pdf_files = sorted(
        RESUME_DIR.rglob("*.pdf")
    )

    print(
        f"\nPDF resumes found: {len(pdf_files)}"
    )

    if not pdf_files:

        print(
            "\nERROR: No PDF resumes found."
        )

        print(
            f"Expected location:\n{RESUME_DIR}"
        )

        return None

    results = []

    print("\nScanning resumes...\n")

    for i, pdf in enumerate(pdf_files, 1):

        print(
            f"Processing {i}/{len(pdf_files)}: "
            f"{pdf.name}"
        )

        result = analyze_pdf(pdf)

        results.append(result)

    df = pd.DataFrame(results)

    return df


# ==========================================================
# SAVE RESULTS
# ==========================================================

def save_results(df):

    # ------------------------------------------------------
    # Full report
    # ------------------------------------------------------

    df.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8"
    )

    # ------------------------------------------------------
    # Clean resumes
    # ------------------------------------------------------

    clean_df = df[
        df["risk"] == "LOW"
    ].copy()

    clean_df.to_csv(
        CLEAN_RESUME_FILE,
        index=False,
        encoding="utf-8"
    )

    # ------------------------------------------------------
    # Flagged resumes
    # ------------------------------------------------------

    flagged_df = df[
        df["risk"].isin(
            ["MEDIUM", "HIGH"]
        )
    ].copy()

    flagged_df.to_csv(
        FLAGGED_RESUME_FILE,
        index=False,
        encoding="utf-8"
    )

    return clean_df, flagged_df


# ==========================================================
# DISPLAY SUMMARY
# ==========================================================

def display_summary(df):

    print("\n")
    print("=" * 60)
    print("FRAUD DETECTION SUMMARY")
    print("=" * 60)

    total = len(df)

    low = len(
        df[df["risk"] == "LOW"]
    )

    medium = len(
        df[df["risk"] == "MEDIUM"]
    )

    high = len(
        df[df["risk"] == "HIGH"]
    )

    print(
        f"\nTotal resumes scanned : {total}"
    )

    print(
        f"LOW risk              : {low}"
    )

    print(
        f"MEDIUM risk           : {medium}"
    )

    print(
        f"HIGH risk             : {high}"
    )

    print(
        f"Flagged resumes       : {medium + high}"
    )

    # ------------------------------------------------------
    # Show flagged resumes
    # ------------------------------------------------------

    flagged = df[
        df["risk"].isin(
            ["MEDIUM", "HIGH"]
        )
    ].copy()

    if len(flagged) > 0:

        print("\n")
        print("=" * 60)
        print("FLAGGED RESUMES")
        print("=" * 60)

        for _, row in flagged.iterrows():

            print("\n" + "-" * 60)

            print(
                f"Resume       : {row['filename']}"
            )

            print(
                f"Risk         : {row['risk']}"
            )

            print(
                f"Fraud Score  : {row['fraud_score']:.2f}%"
            )

            print(
                f"White Text   : {row['white_text_chars']} chars"
            )

            print(
                f"Tiny Text    : {row['tiny_text_chars']} chars"
            )

            print(
                f"Transparent  : {row['transparent_text_chars']} chars"
            )

            print(
                f"Hidden Skills: "
                f"{row['hidden_keywords'] or 'None'}"
            )

            print(
                f"Reasons      : "
                f"{row['reasons'] or 'None'}"
            )

    else:

        print("\n")
        print(
            "✓ No suspicious resumes detected."
        )

    # ------------------------------------------------------
    # Top suspicious resumes
    # ------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("TOP FRAUD RISK RESUMES")
    print("=" * 60)

    top = df.sort_values(
        "fraud_score",
        ascending=False
    ).head(10)

    for _, row in top.iterrows():

        print(
            f"{row['filename']:<20} "
            f"{row['fraud_score']:>6.2f}% "
            f"{row['risk']}"
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("\n")
    print("=" * 60)
    print("TALENTLENS")
    print("NLP-ONLY RECRUITMENT SYSTEM")
    print("STAGE 5 - WHITE-SPACE / HIDDEN-TEXT FRAUD DETECTION")
    print("=" * 60)

    print(
        f"\nResume directory:\n{RESUME_DIR}"
    )

    if not RESUME_DIR.exists():

        print(
            "\nERROR: Resume directory does not exist."
        )

        return

    df = process_all_resumes()

    if df is None:

        return

    clean_df, flagged_df = save_results(df)

    display_summary(df)

    # ======================================================
    # FINAL OUTPUT
    # ======================================================

    print("\n")
    print("=" * 60)
    print("STAGE 5 COMPLETED")
    print("=" * 60)

    print("\nGenerated files:")

    print(
        f"✓ {REPORT_FILE}"
    )

    print(
        f"✓ {CLEAN_RESUME_FILE}"
    )

    print(
        f"✓ {FLAGGED_RESUME_FILE}"
    )

    print("\nNext stage:")

    print(
        "Integrate fraud detection with Top-K screening"
    )


if __name__ == "__main__":
    main()