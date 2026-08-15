import fitz
import pandas as pd
from pathlib import Path


# Paths
RESUME_FOLDER = Path("data/resumes")
OUTPUT_FILE = Path("data/resume_text.csv")


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF resume."""
    text = ""

    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()

        return text.strip()

    except Exception as e:
        print(f"Error reading {pdf_path.name}: {e}")
        return ""


def process_resumes():
    records = []

    # Find all PDF files recursively
    pdf_files = list(RESUME_FOLDER.rglob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files.")

    for i, pdf_path in enumerate(pdf_files, start=1):

        text = extract_text_from_pdf(pdf_path)

        # Get parent folder as category if available
        category = pdf_path.parent.name

        records.append({
            "resume_id": i,
            "filename": pdf_path.name,
            "category": category,
            "resume_text": text,
            "text_length": len(text)
        })

        print(f"[{i}/{len(pdf_files)}] {pdf_path.name}")

    # Create DataFrame
    df = pd.DataFrame(records)

    # Save
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("\nResume extraction completed.")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Total resumes: {len(df)}")

    return df


if __name__ == "__main__":
    process_resumes()