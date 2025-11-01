# app.py
import streamlit as st
import re
import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

# NLTK resources (ensure downloaded)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
STOPWORDS = set(stopwords.words("english"))

# ----------------------------
# Utilities
# ----------------------------
def sanitize_for_pdf(s: str) -> str:
    """Replace problematic unicode with safe alternatives and fallback."""
    if not isinstance(s, str):
        s = str(s)
    replacements = {
        "\u2013": "-", "\u2014": "-", "\u2022": "*", "\u00A0": " ",
        "\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"', "\u2026": "..."
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    # encode/decode to remove others not in latin-1; reportlab accepts utf-8 but we still normalize
    return s

def simple_clean(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    # normalize whitespace and remove weird chars but keep punctuation for summaries
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\xa0', ' ', text)
    return text.strip()

def tokenize_and_lemmatize(text: str) -> str:
    # basic tokenization + lemmatization-light via wordnet lemmatizer can be added if needed.
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    tokens = [t.lower() for t in text.split() if t.lower() not in STOPWORDS and len(t)>2]
    return " ".join(tokens)

def extract_contact(text: str):
    emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    phones = re.findall(r'\+?\d[\d\-\s]{7,}\d', text)
    emails = list(dict.fromkeys(emails))
    phones = list(dict.fromkeys(phones))
    return emails, phones

def brief_snippet(text: str, lines=3):
    # return first few non-empty lines (for the brief shown in UI)
    lines_list = [ln.strip() for ln in text.splitlines() if ln.strip()!=""]
    return "\n".join(lines_list[:lines]) if lines_list else text[:300]

# ----------------------------
# Skill lists & detection
# ----------------------------
# Expand this list as needed; it's used for skill matching.
SKILL_KEYWORDS = [
    'python','java','c++','c#','sql','javascript','react','node','html','css',
    'machine learning','deep learning','nlp','tensorflow','pytorch','pandas','numpy',
    'excel','tableau','power bi','aws','azure','gcp','docker','kubernetes','spark',
    'hadoop','linux','git','rest','api','django','flask','matlab','r'
]
# normalize for substring search (include multiword)
SKILL_KEYWORDS_LOWER = sorted({s.lower() for s in SKILL_KEYWORDS}, key=len, reverse=True)

def extract_skills_from_text(text: str):
    t = text.lower()
    found = []
    for skill in SKILL_KEYWORDS_LOWER:
        # word boundary check for single-word tokens; substring ok for multiwords
        if " " in skill:
            if skill in t:
                found.append(skill)
        else:
            # simple token match
            if re.search(r'\b' + re.escape(skill) + r'\b', t):
                found.append(skill)
    return sorted(set(found))

# ----------------------------
# ATS scoring (improved)
# ----------------------------
def compute_ats_score(resume_raw: str, jd_skills:list):
    """
    ATS breakdown:
      - Contact info presence: 10
      - Section headers (education, experience, skills, projects, certifications): 10
      - Education mention: 15
      - Experience mentions / internships / years: 15
      - Skill coverage vs JD required: 40
    """
    score_components = {}
    text = resume_raw.lower()

    # Contact
    emails, phones = extract_contact(resume_raw)
    contact_points = 0
    if emails: contact_points += 1
    if phones: contact_points += 1
    score_components['Contact (0-10)'] = round((contact_points/2)*10,2)

    # Sections
    section_keywords = ['education','experience','skills','projects','certification','certifications']
    found_sections = sum(1 for k in section_keywords if re.search(r'\b'+k+r'\b', text))
    score_components['Sections (0-10)'] = round((min(found_sections,5)/5)*10,2)

    # Education
    edu_terms = ['btech','b.tech','b.e','bachelor','master','mtech','m.tech','phd','msc','mba','mca']
    edu_found = any(k in text for k in edu_terms)
    score_components['Education (0-15)'] = 15.0 if edu_found else 0.0

    # Experience mention
    exp_terms = ['experience','intern','internship','year','years','worked','project','projects']
    exp_found = any(k in text for k in exp_terms)
    score_components['Experience mention (0-15)'] = 15.0 if exp_found else 0.0

    # Skill coverage relative to JD
    resume_skills = extract_skills_from_text(resume_raw)
    if jd_skills:
        matched = set(resume_skills) & set(jd_skills)
        coverage = len(matched)/len(jd_skills)
        score_components['Skill coverage (0-40)'] = round(min(coverage,1.0)*40,2)
    else:
        score_components['Skill coverage (0-40)'] = 0.0

    total = sum(score_components.values())
    # ensure within 0-100
    total = round(min(max(total,0),100),2)
    return total, score_components, resume_skills

# ----------------------------
# Matching / similarity
# ----------------------------
def compute_match_score(resume_text, jd_text):
    # preprocess lightly for tfidf
    r = tokenize_and_lemmatize(resume_text)
    j = tokenize_and_lemmatize(jd_text)
    vect = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    tf = vect.fit_transform([r,j])
    sim = cosine_similarity(tf[0], tf[1])[0][0]
    return round(sim*100,2), sim

# ----------------------------
# Visualizations (return bytes images)
# ----------------------------
def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

def create_ats_bar(score_components: dict):
    labels = list(score_components.keys())
    vals = [score_components[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7,3))
    sns = __import__('seaborn')
    sns.barplot(x=vals, y=labels, palette='Blues_r', orient='h')
    ax.set_xlabel('Points')
    ax.set_xlim(0,40)  # max per component varies, but overall axis helpful
    ax.set_title('ATS Score Breakdown')
    plt.tight_layout()
    return fig

def create_skill_pie(matched_count, missing_count):
    fig, ax = plt.subplots(figsize=(4,4))
    labels = ['Matched', 'Missing']
    sizes = [matched_count, missing_count]
    if matched_count+missing_count==0:
        sizes=[1,0]
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title('Skill Match Overview')
    plt.tight_layout()
    return fig

def create_skill_bar(resume_skills, job_skills):
    # counts are 1 if present; display two bars side-by-side as counts
    skills = sorted(list(set(resume_skills+job_skills)))
    resume_vals = [1 if s in resume_skills else 0 for s in skills]
    job_vals = [1 if s in job_skills else 0 for s in skills]
    fig, ax = plt.subplots(figsize=(8, max(3, len(skills)*0.25)))
    y = np.arange(len(skills))
    ax.barh(y-0.15, resume_vals, height=0.3, label='Resume')
    ax.barh(y+0.15, job_vals, height=0.3, label='Job Req')
    ax.set_yticks(y); ax.set_yticklabels(skills)
    ax.set_xlabel('Presence (1 = present)')
    ax.set_title('Skill Presence: Resume vs Job')
    ax.legend()
    plt.tight_layout()
    return fig

# ----------------------------
# PDF Report using reportlab (with images)
# ----------------------------
def generate_pdf_report(resume_raw, jd_raw, brief_resume, brief_jd,
                        ats_total, ats_components, match_pct, matched_skills, missing_skills,
                        resume_skills, job_skills, figs_bytes):
    """figs_bytes: dict of name->BytesIO PNG objects"""
    # sanitize text blocks
    resume_snip = sanitize_for_pdf(brief_resume)
    jd_snip = sanitize_for_pdf(brief_jd)
    stylesheet = getSampleStyleSheet()
    doc = SimpleDocTemplate(os.path.join(tempfile.gettempdir(),"ATS_Detailed_Report.pdf"), pagesize=letter)
    elems = []

    elems.append(Paragraph("AI Resume Analysis - Detailed Report", stylesheet['Title']))
    elems.append(Spacer(1,6))
    elems.append(Paragraph("This report gives an ATS-style evaluation, resume-JD fit score, skill-gap analysis and recommendations.", stylesheet['Normal']))
    elems.append(Spacer(1,12))

    elems.append(Paragraph("Resume Brief:", stylesheet['Heading3']))
    elems.append(Paragraph(resume_snip, stylesheet['Normal']))
    elems.append(Spacer(1,8))

    elems.append(Paragraph("Job Description Brief:", stylesheet['Heading3']))
    elems.append(Paragraph(jd_snip, stylesheet['Normal']))
    elems.append(Spacer(1,8))

    elems.append(Paragraph("ATS Score (overall): " + str(ats_total) + " / 100", stylesheet['Heading3']))
    elems.append(Paragraph("ATS Breakdown:", stylesheet['Normal']))
    for k,v in ats_components.items():
        elems.append(Paragraph(f"{k}: {v}", stylesheet['Normal']))
    elems.append(Spacer(1,8))

    elems.append(Paragraph(f"Resume–JD Match Score: {match_pct}%", stylesheet['Heading3']))
    elems.append(Spacer(1,6))
    elems.append(Paragraph("Matched skills: " + (", ".join(matched_skills) or "None"), stylesheet['Normal']))
    elems.append(Paragraph("Missing skills: " + (", ".join(missing_skills) or "None"), stylesheet['Normal']))
    elems.append(Spacer(1,8))

    # Add visuals images
    for name, b in figs_bytes.items():
        elems.append(Paragraph(name, stylesheet['Heading4']))
        img_path = os.path.join(tempfile.gettempdir(), f"{name}.png")
        with open(img_path, "wb") as f:
            f.write(b.getbuffer())
        elems.append(Image(img_path, width=400, height=300))
        elems.append(Spacer(1,6))

    elems.append(Paragraph("Improvement Suggestions:", stylesheet['Heading3']))
    suggestions = [
        "Add missing JD skills to resume if you possess them, in a 'Skills' section.",
        "Quantify achievements with numbers (e.g., improved X by Y%).",
        "Include project descriptions with technologies used.",
        "Add any relevant certifications and courses."
    ]
    for s in suggestions:
        elems.append(Paragraph("• " + s, stylesheet['Normal']))
    elems.append(Spacer(1,12))

    doc.build(elems)
    return os.path.join(tempfile.gettempdir(),"ATS_Detailed_Report.pdf")

# ----------------------------
# Streamlit App UI
# ----------------------------
st.set_page_config(page_title="Advanced ATS Resume Analyzer", layout="wide")
st.title("Advanced ATS Resume Analyzer — Detailed Evaluation")

with st.sidebar:
    st.header("About")
    st.write("Upload one Resume `.txt` and one Job Description `.txt` (paste text into files if needed).")
    st.write("App computes: ATS breakdown, Resume–JD match (TF-IDF), Skill gap, Visuals and generates a detailed PDF report.")
    st.markdown("---")
    st.write("Tips:")
    st.write("- Ensure resume contains a Skills section for good extraction.")
    st.write("- Expand `SKILL_KEYWORDS` in code for domain-specific terms.")

col1, col2 = st.columns([1,2])
with col1:
    resume_file = st.file_uploader("Upload resume (.txt)", type=['txt'])
    jd_file = st.file_uploader("Upload job description (.txt)", type=['txt'])
    run_button = st.button("Run Analysis")

with col2:
    st.info("After uploading both files, click **Run Analysis**. Summary and charts will appear here.")

if run_button:
    if not resume_file or not jd_file:
        st.error("Please upload both files.")
    else:
        raw_resume = resume_file.read().decode('utf-8', errors='ignore')
        raw_jd = jd_file.read().decode('utf-8', errors='ignore')

        # Brief summaries
        resume_brief = brief_snippet(raw_resume, lines=6)
        jd_brief = brief_snippet(raw_jd, lines=6)

        st.subheader("Resume Brief")
        st.text(resume_brief)
        st.subheader("Job Description Brief")
        st.text(jd_brief)

        # compute
        match_pct, raw_sim = compute_match_score(raw_resume, raw_jd)
        ats_total, ats_components, resume_skills = compute_ats_score(raw_resume, extract_skills_from_text(raw_jd))
        # skill lists
        jd_skills = extract_skills_from_text(raw_jd)
        matched = sorted(list(set(resume_skills) & set(jd_skills)))
        missing = sorted(list(set(jd_skills) - set(resume_skills)))

        # Display scores
        st.markdown("### Scores")
        c1, c2 = st.columns(2)
        c1.metric("ATS Score", f"{ats_total} / 100")
        c2.metric("Resume–JD Match", f"{match_pct}%")

        # Visuals
        st.subheader("Visualizations")
        fig1 = create_ats_bar(ats_components)
        st.pyplot(fig1)

        fig2 = create_skill_pie(len(matched), len(missing))
        st.pyplot(fig2)

        fig3 = create_skill_bar(resume_skills, jd_skills)
        st.pyplot(fig3)

        # Wordcloud of resume top words
        st.subheader("Resume Wordcloud (top tokens)")
        wc = WordCloud(width=800, height=300, background_color='white').generate(tokenize_and_lemmatize(raw_resume))
        fig_wc = plt.figure(figsize=(10,3))
        plt.imshow(wc, interpolation='bilinear'); plt.axis('off')
        st.pyplot(fig_wc)

        # Matched & Missing lists
        st.subheader("Skill Gap Analysis")
        st.write("Matched skills:", ", ".join(matched) if matched else "None")
        st.write("Missing skills (from JD):", ", ".join(missing) if missing else "None")

        # Generate PDF
        st.subheader("Download Detailed Report (PDF)")
        # Prepare images bytes
        figs_bytes = {
            "ATS Breakdown": fig_to_bytes(fig1),
            "Skill Match Overview": fig_to_bytes(fig2),
            "Skill Presence Bar": fig_to_bytes(fig3),
            "Resume Wordcloud": fig_to_bytes(fig_wc)
        }
        pdf_path = generate_pdf_report(raw_resume, raw_jd, resume_brief, jd_brief,
                                       ats_total, ats_components, match_pct, matched, missing,
                                       resume_skills, jd_skills, figs_bytes)
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name="ATS_Detailed_Report.pdf", mime="application/pdf")
