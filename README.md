
# AI Resume Matcher (ATS Analyzer)

## 🧠 Overview
The **AI Resume Matcher (ATS Analyzer)** is an intelligent web app that analyzes a resume against a given job description (JD) and generates a detailed **ATS (Applicant Tracking System) report**. The system uses **Natural Language Processing (NLP)** to calculate an ATS score, match percentage, and perform **skill gap analysis**, **resume–JD fit visualization**, and **improvement suggestions**.

## 🚀 Features
- Upload a **Resume (PDF)** and **Job Description (Text/PDF)**.
- View **brief details** of both Resume and JD.
- Generate a detailed **ATS Report** including:
  - Resume Summary
  - Job Description Summary
  - ATS Match Score
  - Resume–JD Fit Analysis
  - Skill Gap Analysis
  - Key Missing Skills
  - Improvement Suggestions
  - **Visualizations** (Pie charts, Bar charts for match ratio and skill categories)
- Modern UI with left-side panel explaining the app and right-side dynamic analysis area.

## 🧩 Tech Stack
- **Frontend:** Streamlit (Python)
- **Backend:** NLP-based processing using spaCy and scikit-learn
- **Visualization:** Matplotlib, Plotly
- **PDF Handling:** PyMuPDF (fitz), FPDF

## 📁 Project Structure
```
AI_Resume_Matcher/
│
├── app.py                     # Main Streamlit application
├── resume_analysis.py          # Core NLP-based resume and JD comparison logic
├── report_generator.py         # Generates detailed ATS report (PDF)
├── requirements.txt            # Python dependencies
├── sample_resume.pdf           # Example resume file
├── sample_jd.txt               # Example job description
└── README.md                   # Project documentation (this file)
```

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/AI-Resume-Matcher.git
cd AI-Resume-Matcher
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App
```bash
streamlit run app.py
```

## 📊 Outputs
- **ATS Score** (0–100%)
- **Matched Keywords**
- **Skill Gap Report**
- **Improvement Areas**
- **Visual Match Charts**
- **Downloadable Detailed Report (PDF)**

## 📘 Example Usage
1. Upload your **resume** and **job description**.
2. The app will automatically analyze both and generate insights.
3. View all results and charts directly within the web app.
4. Download your full **ATS analysis report (PDF)**.

## 📄 License
This project is open-source under the **MIT License**.

## 💡 Author
Developed by **Akshat Kumarey**
