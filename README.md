# 🚀 TalentLens — NLP-Based AI Recruitment Screening System

> An intelligent recruitment screening platform that uses Natural Language Processing (NLP) to compare Job Descriptions (JDs) with resumes, rank candidates, detect hidden-text resume manipulation, generate explanations, and perform what-if candidate analysis.

🔗 **Live Demo:** YOUR_LIVE_DEMO_LINK_HERE

🔗 **GitHub Repository:** YOUR_GITHUB_REPOSITORY_LINK_HERE

---

## 📌 Overview

TalentLens is an NLP-based recruitment intelligence system designed to automate the initial resume screening process.

Recruiters can:

- Upload multiple resumes
- Enter or upload a Job Description
- Extract relevant information from resumes and the JD
- Calculate candidate-to-JD similarity
- Match required skills
- Evaluate experience and qualifications
- Rank candidates
- Generate a Top-K shortlist
- Detect hidden/white-text resume manipulation
- Block suspicious candidates from final screening
- Generate candidate-specific explanations
- Perform "What-If" analysis by modifying candidate skills
- View all results through an interactive Streamlit dashboard

The system is designed as an **explainable and fraud-aware NLP recruitment pipeline**.

---

# 🎯 Problem Statement

Traditional resume screening can become difficult when recruiters receive hundreds of resumes for a single job.

Common problems include:

- Manual resume screening is time-consuming
- Keyword-based ATS systems may miss semantic relationships
- Candidates may manipulate resumes using hidden/white text
- Recruiters may not understand why a candidate received a particular score
- Comparing large numbers of candidates manually is inefficient
- Recruiters cannot easily simulate how adding/removing skills would affect a candidate's score

TalentLens addresses these problems using an automated NLP pipeline.

---

# 💡 Proposed Solution

TalentLens processes resumes through multiple stages:

```text
                ┌───────────────────┐
                │   Job Description  │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ JD Processing     │
                │ & Skill Extraction│
                └─────────┬─────────┘
                          │
                          ▼
┌─────────────────────────────────────────────┐
│               Resume Upload                 │
│          Multiple PDF Resumes               │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
             ┌───────────────────┐
             │ Resume Processing │
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │ NLP Matching      │
             │                   │
             │ TF-IDF            │
             │ Cosine Similarity │
             │ Skill Matching    │
             │ Experience        │
             │ Qualification     │
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │ Candidate Ranking │
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │ Top-K Shortlisting│
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │ Fraud Detection   │
             │                   │
             │ Hidden Text       │
             │ White Text        │
             │ Tiny Text         │
             │ Transparent Text  │
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │ Final Screening   │
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │ Explanation       │
             │ + What-If Analysis│
             └───────────────────┘

