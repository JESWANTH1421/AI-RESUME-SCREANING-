import zipfile
from pathlib import Path

import pymupdf
import pandas as pd


# ==========================================================
# TALENTLENS NLP-ONLY
# STAGE 1: DATASET PROCESSING
# ==========================================================

BASE_DIR = Path.cwd()

# Your actual DATA folder
DATA_DIR = BASE_DIR / "DATA"

# Folders where extracted datasets will be stored
RESUME_DIR = DATA_DIR / "resumes"
JD_DIR = DATA_DIR / "job_descriptions"

# Your actual ZIP file names
RESUME_ZIP = DATA_DIR / "Resume_Dataset (1).zip"
JD_ZIP = DATA_DIR / "archive.zip"

# Output files
RESUME_OUTPUT = DATA_DIR / "resume_text.csv"
JD_OUTPUT = DATA_DIR / "job_descriptions_processed.csv"


# ==========================================================
# CREATE REQUIRED FOLDERS
# ==========================================================

def create_folders():

    RESUME_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    JD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ==========================================================
# ZIP EXTRACTION
# ==========================================================

def extract_zip(zip_file, destination):

    if not zip_file.exists():

        print("\nERROR: ZIP file not found:")
        print(zip_file)

        return False

    print("\n" + "-" * 60)
    print(f"Extracting: {zip_file.name}")
    print(f"Destination: {destination}")
    print("-" * 60)

    try:

        with zipfile.ZipFile(zip_file, "r") as zip_ref:

            zip_ref.extractall(destination)

        print("Extraction completed successfully.")

        return True

    except zipfile.BadZipFile:

        print("ERROR: This is not a valid ZIP file.")

        return False

    except Exception as e:

        print(f"ERROR during extraction: {e}")

        return False


# ==========================================================
# PDF TEXT EXTRACTION
# ==========================================================

def extract_pdf_text(pdf_path):

    text = ""

    try:

        with pymupdf.open(pdf_path) as pdf:

            for page in pdf:

                text += page.get_text()
                text += "\n"

        return text.strip()

    except Exception as e:

        print(
            f"ERROR reading {pdf_path.name}: {e}"
        )

        return ""


# ==========================================================
# FIND RESUME PDFs
# ==========================================================

def find_resume_pdfs():

    pdf_files = list(
        RESUME_DIR.rglob("*.pdf")
    )

    return pdf_files


# ==========================================================
# PROCESS RESUMES
# ==========================================================

def process_resumes():

    print("\n")
    print("=" * 60)
    print("RESUME PROCESSING")
    print("=" * 60)

    pdf_files = find_resume_pdfs()

    print(
        f"\nPDF resumes found: {len(pdf_files)}"
    )

    if not pdf_files:

        print(
            "\nERROR: No PDF resumes were found."
        )

        print(
            f"Check this folder:\n{RESUME_DIR}"
        )

        return None

    records = []

    for i, pdf in enumerate(
        pdf_files,
        start=1
    ):

        print(
            f"Processing {i}/{len(pdf_files)}: "
            f"{pdf.name}"
        )

        text = extract_pdf_text(pdf)

        records.append({

            "resume_id": i,

            "filename": pdf.name,

            "category": pdf.parent.name,

            "resume_text": text,

            "text_length": len(text)

        })

    df = pd.DataFrame(records)

    # Save extracted resume text
    df.to_csv(
        RESUME_OUTPUT,
        index=False,
        encoding="utf-8"
    )

    empty_resumes = (
        df["text_length"] == 0
    ).sum()

    print("\n")
    print("=" * 60)
    print("RESUME PROCESSING COMPLETED")
    print("=" * 60)

    print(
        f"Total resumes: {len(df)}"
    )

    print(
        f"Empty resumes: {empty_resumes}"
    )

    print(
        f"Output file: {RESUME_OUTPUT}"
    )

    return df


# ==========================================================
# FIND CSV FILES
# ==========================================================

def find_csv_files():

    return list(
        JD_DIR.rglob("*.csv")
    )


# ==========================================================
# PROCESS JOB DESCRIPTIONS
# ==========================================================

def process_job_descriptions():

    print("\n")
    print("=" * 60)
    print("JOB DESCRIPTION PROCESSING")
    print("=" * 60)

    csv_files = find_csv_files()

    print(
        f"\nCSV files found: {len(csv_files)}"
    )

    if not csv_files:

        print(
            "\nERROR: No CSV file was found."
        )

        print(
            f"Check this folder:\n{JD_DIR}"
        )

        return None

    # Display all CSV files found
    print("\nCSV files:")

    for csv_file in csv_files:

        print(
            f" - {csv_file}"
        )

    # Use the largest CSV file.
    # This prevents accidentally selecting a small metadata CSV.
    csv_file = max(
        csv_files,
        key=lambda file: file.stat().st_size
    )

    print("\n")
    print(
        f"Selected JD file: {csv_file.name}"
    )

    print(
        "Loading job-description dataset..."
    )

    try:

        df = pd.read_csv(
            csv_file,
            low_memory=False
        )

    except Exception as e:

        print(
            f"\nERROR reading CSV: {e}"
        )

        return None

    print("\n")
    print("=" * 60)
    print("JOB DESCRIPTION DATASET LOADED")
    print("=" * 60)

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # ------------------------------------------------------
    # Columns
    # ------------------------------------------------------

    print("\nColumns:")

    for column in df.columns:

        print(
            f" - {column}"
        )

    # ------------------------------------------------------
    # Missing values
    # ------------------------------------------------------

    print("\nTop missing-value columns:")

    missing_values = (
        df.isnull()
        .sum()
        .sort_values(
            ascending=False
        )
        .head(20)
    )

    print(
        missing_values
    )

    # ------------------------------------------------------
    # Save processed copy
    # ------------------------------------------------------

    print(
        "\nSaving JD dataset..."
    )

    df.to_csv(
        JD_OUTPUT,
        index=False,
        encoding="utf-8"
    )

    print(
        f"Saved to: {JD_OUTPUT}"
    )

    return df


# ==========================================================
# RESUME STATISTICS
# ==========================================================

def display_resume_statistics(
    resume_df
):

    if resume_df is None:

        return

    print("\n")
    print("=" * 60)
    print("RESUME DATASET STATISTICS")
    print("=" * 60)

    print("\nDataset shape:")

    print(
        resume_df.shape
    )

    print("\nResume categories:")

    print(
        resume_df[
            "category"
        ].value_counts(
            dropna=False
        )
    )

    print("\nText length statistics:")

    print(
        resume_df[
            "text_length"
        ].describe()
    )

    # ------------------------------------------------------
    # First resume preview
    # ------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FIRST RESUME PREVIEW")
    print("=" * 60)

    first_resume = resume_df.iloc[0]

    print(
        f"\nFilename: "
        f"{first_resume['filename']}"
    )

    print(
        f"Category: "
        f"{first_resume['category']}"
    )

    print("\nExtracted text:")

    print("-" * 60)

    print(
        str(
            first_resume[
                "resume_text"
            ]
        )[:3000]
    )

    print("-" * 60)


# ==========================================================
# JOB DESCRIPTION PREVIEW
# ==========================================================

def display_jd_statistics(
    jd_df
):

    if jd_df is None:

        return

    print("\n")
    print("=" * 60)
    print("JOB DESCRIPTION PREVIEW")
    print("=" * 60)

    print(
        jd_df.head(3).to_string()
    )


# ==========================================================
# MAIN
# ==========================================================

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
        "STAGE 1 - DATA PROCESSING"
    )

    print("=" * 60)

    print(
        f"\nProject folder:\n{BASE_DIR}"
    )

    print(
        f"\nDATA folder:\n{DATA_DIR}"
    )

    # ------------------------------------------------------
    # Create folders
    # ------------------------------------------------------

    create_folders()

    # ------------------------------------------------------
    # Check resume ZIP
    # ------------------------------------------------------

    print("\n")
    print(
        "[1/4] Checking resume dataset..."
    )

    if not RESUME_ZIP.exists():

        print(
            "\nERROR: Resume dataset not found."
        )

        print(
            f"Expected:\n{RESUME_ZIP}"
        )

        return

    print(
        f"Found: {RESUME_ZIP.name}"
    )

    # ------------------------------------------------------
    # Extract resume dataset
    # ------------------------------------------------------

    print("\n")
    print(
        "[2/4] Extracting resume dataset..."
    )

    resume_success = extract_zip(
        RESUME_ZIP,
        RESUME_DIR
    )

    if not resume_success:

        return

    # ------------------------------------------------------
    # Check JD ZIP
    # ------------------------------------------------------

    print("\n")
    print(
        "[3/4] Checking job-description dataset..."
    )

    if not JD_ZIP.exists():

        print(
            "\nERROR: Job-description dataset not found."
        )

        print(
            f"Expected:\n{JD_ZIP}"
        )

        return

    print(
        f"Found: {JD_ZIP.name}"
    )

    # ------------------------------------------------------
    # Extract JD dataset
    # ------------------------------------------------------

    print("\n")
    print(
        "[4/4] Extracting job-description dataset..."
    )

    jd_success = extract_zip(
        JD_ZIP,
        JD_DIR
    )

    if not jd_success:

        return

    # ------------------------------------------------------
    # Process resumes
    # ------------------------------------------------------

    resume_df = process_resumes()

    # ------------------------------------------------------
    # Process job descriptions
    # ------------------------------------------------------

    jd_df = process_job_descriptions()

    # ------------------------------------------------------
    # Display statistics
    # ------------------------------------------------------

    display_resume_statistics(
        resume_df
    )

    display_jd_statistics(
        jd_df
    )

    # ------------------------------------------------------
    # Final
    # ------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("STAGE 1 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated files:")

    if RESUME_OUTPUT.exists():

        print(
            f"✓ {RESUME_OUTPUT}"
        )

    if JD_OUTPUT.exists():

        print(
            f"✓ {JD_OUTPUT}"
        )

    print("\nNext stage:")

    print(
        "NLP preprocessing + skill extraction"
    )

    print("\nDo NOT start scoring yet.")


# ==========================================================
# PROGRAM ENTRY
# ==========================================================

if __name__ == "__main__":

    main()