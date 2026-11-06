import ast
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Career Lens | Resume Analyzer", page_icon="🎯", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {padding-top: 1.8rem; max-width: 1180px;}
.hero {padding: 1.5rem 1.7rem; border-radius: 18px; background: linear-gradient(120deg,#14213d,#244a73); color: white; margin-bottom: 1rem;}
.hero h1 {font-size: 2.35rem; margin: 0;}
.hero p {font-size: 1.05rem; opacity: .92; margin-bottom: 0;}
div[data-testid="stMetric"] {background: #f5f7fb; padding: 1rem; border-radius: 12px; border: 1px solid #e4e9f2;}
.small-note {color:#6b7280; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🎯 Career Lens</h1>
  <p>Resume analysis and job-fit estimation powered by your trained machine-learning model.</p>
</div>
""", unsafe_allow_html=True)

ARTIFACTS = {
    "model": Path("career_lens_model_compressed.pkl"),
    "scaler": Path("career_lens_scaler.pkl"),
    "features": Path("career_lens_features.pkl"),
}

@st.cache_resource
def load_artifacts():
    missing = [str(p) for p in ARTIFACTS.values() if not p.exists()]
    if missing:
        return None, None, None, missing
    model = joblib.load(ARTIFACTS["model"])
    scaler = joblib.load(ARTIFACTS["scaler"])
    features = joblib.load(ARTIFACTS["features"])
    return model, scaler, features, []

def parse_resume(uploaded_file):
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".txt"):
            return uploaded_file.getvalue().decode("utf-8", errors="ignore")
        if name.endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(uploaded_file)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if name.endswith(".docx"):
            from docx import Document
            doc = Document(uploaded_file)
            return "\n".join(p.text for p in doc.paragraphs)
    except Exception as exc:
        st.warning(f"Could not read the uploaded file automatically: {exc}")
    return ""

def norm(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()

def extract_skills(text, custom_skills=""):
    # Simple keyword-based extraction; edit this list to match your dataset/skills.
    catalog = [
        "python", "java", "c++", "c", "sql", "mysql", "html", "css", "javascript",
        "machine learning", "deep learning", "artificial intelligence", "generative ai",
        "data analysis", "data science", "pandas", "numpy", "scikit-learn", "sklearn",
        "tensorflow", "pytorch", "nlp", "natural language processing", "computer vision",
        "opencv", "excel", "power bi", "tableau", "statistics", "flask", "streamlit",
        "django", "git", "github", "aws", "azure", "communication", "project management",
        "feature engineering", "model building", "data visualization", "mongodb", "firebase",
        "kotlin", "android", "react", "node.js", "php", "r", "linux"
    ]
    lower = text.lower()
    found = [s for s in catalog if re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", lower)]
    custom = [x.strip().lower() for x in re.split(r"[,;\n]", custom_skills) if x.strip()]
    return sorted(set(found + custom))

def choose_dummy(features, prefix, selected):
    """Return the exact one-hot feature name matching a UI selection, if present."""
    candidates = [f for f in features if f.startswith(prefix)]
    if not candidates:
        return None
    wanted = str(selected).strip().lower()
    exact = [f for f in candidates if f[len(prefix):].strip().lower() == wanted]
    if exact:
        return exact[0]
    # Match a category when the displayed label is a readable subset of a training category.
    partial = [f for f in candidates if wanted and wanted in f[len(prefix):].strip().lower()]
    if partial:
        return partial[0]
    return None

model, scaler, feature_names, missing = load_artifacts()

if missing:
    st.warning("The trained model files are not beside app.py yet.")
    st.markdown("""
**First create/download the three model artifacts from your notebook:**
1. Run the final training cells that create `career_lens_model_compressed.pkl`, `career_lens_scaler.pkl`, and `career_lens_features.pkl`.
2. Download those three files from Colab.
3. Put all three files in the same folder as `app.py` (and commit them to the same private/public GitHub repository you deploy from, if appropriate).

The app intentionally does not retrain the model when someone visits the website.
""")
    with st.expander("Expected project files"):
        st.code("app.py\nrequirements.txt\ncareer_lens_model_compressed.pkl\ncareer_lens_scaler.pkl\ncareer_lens_features.pkl")
    st.stop()

# Find numeric features used in the notebook. Values are derived from the user's inputs.
NUMERIC_DEFAULTS = {
    "min_experience": 0.0, "max_experience": 0.0, "min_age": 0.0, "max_age": 0.0,
    "job_count": 0.0, "total_experience": 0.0, "skills_count": 0.0,
    "required_skills_count": 0.0, "matching_skills_count": 0.0,
    "matching_skills_percentage": 0.0, "highest_degree": 0.0, "education_count": 0.0,
    "company_count": 0.0, "position_count": 0.0, "location_count": 0.0,
    "responsibility_count": 0.0, "latest_passing_year": 0.0, "related_skills_count": 0.0,
}

st.sidebar.header("About Career Lens")
st.sidebar.write("Upload a resume, enter the target role and required skills, then generate an estimated match score.")
st.sidebar.caption("This is a student project prototype. Review the result manually; it is not a hiring decision.")

left, right = st.columns([1.05, 1])
with left:
    st.subheader("1. Candidate resume")
    uploaded = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"])
    resume_text = parse_resume(uploaded)
    if uploaded:
        st.caption(f"Uploaded: {uploaded.name}")
        if resume_text:
            st.success("Resume text extracted.")
        else:
            st.info("Text extraction was empty. You can still enter skills manually.")
    skills_manual = st.text_area(
        "Skills (comma-separated)",
        placeholder="Python, SQL, Pandas, Machine Learning",
        help="Detected keywords from the uploaded resume are combined with these skills."
    )
    degree_label = st.selectbox("Highest education", [
        "School", "Diploma", "Bachelor", "Master", "PhD", "Unknown"
    ])
    education_count = st.number_input("Number of education qualifications", min_value=0, max_value=10, value=1)
    latest_year = st.number_input("Latest passing year", min_value=0, max_value=2100, value=2026)
    total_experience = st.number_input("Total work experience (years)", min_value=0.0, max_value=60.0, value=0.0, step=0.5)
    company_count = st.number_input("Number of previous companies", min_value=0, max_value=50, value=0)
    position_count = st.number_input("Number of previous positions", min_value=0, max_value=50, value=0)

with right:
    st.subheader("2. Target job")
    job_title = st.text_input("Job title", placeholder="e.g. Machine Learning Intern")
    major_field = st.selectbox("Candidate's field of study", [
        "Computer/IT", "Data/AI", "Mechanical", "Civil",
        "Electrical/Electronics", "Business/Finance", "Science", "Other"
    ])
    education_requirement = st.selectbox("Minimum education requirement", [
        "Unknown", "Bachelor", "Master", "Diploma", "PhD", "High School"
    ])
    required_text = st.text_area(
        "Required skills",
        placeholder="Python, Machine Learning, Pandas, Scikit-learn",
        help="Separate skills with commas or new lines."
    )
    min_exp = st.number_input("Minimum experience required (years)", min_value=0.0, max_value=60.0, value=0.0, step=0.5)
    max_exp = st.number_input("Maximum experience required (years)", min_value=0.0, max_value=60.0, value=0.0, step=0.5)
    min_age = st.number_input("Minimum age (if specified)", min_value=0, max_value=100, value=0)
    max_age = st.number_input("Maximum age (if specified)", min_value=0, max_value=100, value=0)

st.divider()
analyze = st.button("✨ Analyze resume match", type="primary", use_container_width=True)

if analyze:
    candidate_skills = extract_skills(resume_text, skills_manual)
    required_skills = [s.strip().lower() for s in re.split(r"[,;\n]", required_text) if s.strip()]
    candidate_set = set(candidate_skills)
    required_set = set(required_skills)
    matches = sorted(candidate_set.intersection(required_set))
    matching_count = len(matches)
    required_count = len(required_set)
    match_pct = (matching_count / required_count * 100) if required_count else 0.0

    degree_map = {"School": 1, "Diploma": 2, "Bachelor": 3, "Master": 4, "PhD": 5, "Unknown": 0}
    row = {feature: 0 for feature in feature_names}
    values = {
        "min_experience": min_exp, "max_experience": max_exp,
        "min_age": min_age, "max_age": max_age,
        "job_count": company_count, "total_experience": total_experience,
        "skills_count": len(candidate_set), "required_skills_count": required_count,
        "matching_skills_count": matching_count, "matching_skills_percentage": match_pct,
        "highest_degree": degree_map[degree_label], "education_count": education_count,
        "company_count": company_count, "position_count": position_count,
        "location_count": 0, "responsibility_count": len([x for x in resume_text.splitlines() if x.strip()]),
        "latest_passing_year": latest_year, "related_skills_count": matching_count,
    }
    for key, value in values.items():
        if key in row:
            row[key] = value

    # Set the relevant one-hot features. Unknown or unseen values safely remain all-zero.
    dummy_selections = [
        ("job_position_name_", job_title),
        ("major_field_category_", major_field),
        ("educationaL_requirements_", education_requirement),
    ]
    selected_dummy_notes = []
    for prefix, selected in dummy_selections:
        match = choose_dummy(feature_names, prefix, selected)
        if match:
            row[match] = 1
            selected_dummy_notes.append(f"{prefix}{selected}")
        else:
            selected_dummy_notes.append(f"No exact training category found for {selected!r}; category indicators left at zero.")

    X_input = pd.DataFrame([[row[f] for f in feature_names]], columns=feature_names)
    try:
        scaled = scaler.transform(X_input)
        prediction = float(model.predict(scaled)[0])
        # The target is a score, so display it as a percentage while preserving the raw prediction range.
        score = max(0.0, min(1.0, prediction)) * 100.0
        st.subheader("3. Match results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated ML match score", f"{score:.1f}%")
        m2.metric("Skill overlap", f"{match_pct:.1f}%")
        m3.metric("Skills matched", f"{matching_count} / {required_count}")
        st.progress(int(round(score)))
        if score >= 75:
            st.success("Strong estimated match. Review the matched and missing skills before applying.")
        elif score >= 50:
            st.info("Moderate estimated match. Strengthen the missing skills and tailor the resume.")
        else:
            st.warning("Lower estimated match. Compare the job requirements with the resume and improve relevant evidence.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Matched skills**")
            if matches:
                st.write(", ".join(matches))
            else:
                st.write("No exact skill-name matches found.")
        with c2:
            st.markdown("**Skills to add or demonstrate**")
            missing_skills = sorted(required_set - candidate_set)
            st.write(", ".join(missing_skills) if missing_skills else "No missing required skills found.")
        with st.expander("Model input checks"):
            st.write("Training features:", len(feature_names))
            st.write("Category mapping notes:", selected_dummy_notes)
            st.caption("A missing training category is encoded as all-zero indicators. This can reduce prediction reliability.")
        st.caption("Important: the score is a model estimate based on the notebook's training setup and these manually supplied inputs. It is not a probability of getting hired.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info("Confirm that the model, scaler, and feature-list files were all generated from the same notebook run and are stored beside app.py.")

with st.expander("How this app works"):
    st.markdown("""
- Extracts selectable text from PDF, DOCX, or TXT resumes.
- Uses keyword matching to identify skills and compare them with the job's required skills.
- Builds the numeric and one-hot input features expected by the saved notebook model.
- Applies the saved scaler and Random Forest model to estimate a match score.

**Limitations:** keyword extraction is intentionally simple; scanned PDFs need OCR before text can be extracted. The notebook's original training features are based on a dataset, so manual inputs and unseen job categories may not perfectly represent a real resume/job pair.
""")
