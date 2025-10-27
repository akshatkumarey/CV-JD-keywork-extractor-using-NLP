# Resume and Job Description Keyword Extractor using NLP

# Project Code
# =====================================
# Resume and Job Description Keyword Extractor
# =====================================

## Step 1: Import Libraries
import spacy
import re
import os
from nltk.corpus import stopwords
import nltk

## Download required NLTK data
nltk.download('stopwords')

## Load spaCy English model
nlp = spacy.load('en_core_web_sm')

# =====================================
# Step 2: Define File Paths
# =====================================
DATA_PATH = "data"
resume_file = os.path.join(DATA_PATH, "sample_resume.txt")
jd_file = os.path.join(DATA_PATH, "job_description.txt")

# =====================================
# Step 3: Text Preprocessing Function
# =====================================
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation/numbers
    text = text.lower()
    tokens = [word for word in text.split() if word not in stopwords.words('english')]
    return ' '.join(tokens)

# =====================================
# Step 4: NLP Processing and Extraction
# =====================================
def extract_entities(text):
    doc = nlp(text)
    entities = {"PERSON": [], "ORG": [], "EDUCATION": [], "SKILLS": [], "EXPERIENCE": []}
    
    # 4.1 Named Entity Recognition (NER)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            entities[ent.label_].append(ent.text)
    
    # 4.2 Education keywords
    education_keywords = ['btech', 'mtech', 'bachelor', 'master', 'phd', 'computer science', 'engineering', 'degree']
    for word in education_keywords:
        if word in text.lower():
            entities["EDUCATION"].append(word.title())

    # 4.3 Skill extraction (custom list)
    skills = [
        'python', 'java', 'sql', 'c++', 'machine learning', 'deep learning', 
        'data analysis', 'excel', 'communication', 'leadership', 'html', 'css',
        'javascript', 'tensorflow', 'pandas', 'numpy', 'data visualization'
    ]
    for skill in skills:
        if skill in text.lower():
            entities["SKILLS"].append(skill.title())

    # 4.4 Experience detection
    exp_keywords = ['experience', 'worked', 'intern', 'project', 'year']
    for word in exp_keywords:
        if word in text.lower():
            entities["EXPERIENCE"].append(word.title())

    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    return entities

# =====================================
# Step 5: Load and Process Files
# =====================================
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

resume_text = read_file(resume_file)
jd_text = read_file(jd_file)

clean_resume = clean_text(resume_text)
clean_jd = clean_text(jd_text)

# Extract information
resume_entities = extract_entities(clean_resume)
jd_entities = extract_entities(clean_jd)

# =====================================
# Step 6: Display Results
# =====================================
def display_output(title, entities):
    print("\n" + "="*50)
    print(title)
    print("="*50)
    for key, values in entities.items():
        print(f"{key}: {values}")

display_output("Resume Keywords Extracted", resume_entities)
display_output("Job Description Keywords Extracted", jd_entities)

# =====================================
# Step 7 (Optional): Match Skills Between Resume and Job Description
# =====================================
common_skills = set(resume_entities["SKILLS"]) & set(jd_entities["SKILLS"])
print("\nCommon Skills Between Resume and Job Description:")
print(list(common_skills) if common_skills else "No common skills found.")

## Project Overview
This mini-project demonstrates how **Natural Language Processing (NLP)** can be applied to extract and compare key information from textual data — specifically **resumes** and **job descriptions**.

The system automatically identifies important keywords such as:
- **Skills**
- **Education details**
- **Experience**
- **Organizations**
- **Common skills** between resume and job description

This project showcases foundational NLP techniques including **text preprocessing**, **tokenization**, **named entity recognition (NER)**, and **rule-based keyword extraction**.

---

## Features
✅ Clean and preprocess text data  
✅ Extract entities like PERSON, ORG, and EDUCATION using spaCy  
✅ Extract skills using a custom keyword list  
✅ Compare and display common skills between resume and job description  
✅ Simple and interpretable output  

---

## 🧩 Project Structure
```
📂 NLP-Resume-Keyword-Extractor/
 ├── Project.ipynb
 ├── data/
 │    ├── sample_resume.txt
 │    └── job_description.txt
 ├── README.md
 └── requirements.txt
```

- `Project.ipynb` → Main Jupyter Notebook containing the code  
- `data/` → Folder containing text files (resume and job description)  
- `README.md` → Documentation file (this one)  
- `requirements.txt` → List of required Python libraries  

---

## 📄 Input Files
You must create a folder named `data` in the same directory as your notebook.

Inside it, place two text files:
```
data/
 ├── sample_resume.txt
 └── job_description.txt
```

Each file should contain plain text (no formatting).

Example:

**sample_resume.txt**
```
Akshat Kumarey
BTech in Computer Science
Worked as Data Analyst Intern at Infosys.
Skills: Python, Machine Learning, SQL, Communication
```

**job_description.txt**
```
We are hiring a Python Developer skilled in SQL and Machine Learning.
Excellent communication and teamwork skills are required.
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/NLP-Resume-Keyword-Extractor.git
cd NLP-Resume-Keyword-Extractor
```

### 2️⃣ Install Required Libraries
Make sure you have Python 3.10+ installed, then install dependencies:
```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn’t exist, install manually:
```bash
pip install spacy nltk
python -m spacy download en_core_web_sm
```

---

## 🧠 How It Works

### Step 1: Text Preprocessing
- Convert text to lowercase
- Remove punctuation and numbers
- Remove stopwords
- Tokenize words

### Step 2: Named Entity Recognition (NER)
- Identify **PERSON**, **ORG**, and **GPE** entities using `spaCy`

### Step 3: Keyword Extraction
- Extract **skills** and **education keywords** using predefined lists

### Step 4: Matching Common Skills
- Compare skill lists between resume and job description
- Display overlapping skills

---

## 📊 Sample Output
```
📄 Resume Keywords Extracted
PERSON: ['Akshat Kumarey']
ORG: ['Infosys']
EDUCATION: ['Btech', 'Computer Science']
SKILLS: ['Python', 'Machine Learning', 'SQL', 'Communication']
EXPERIENCE: ['Intern', 'Experience']

📋 Job Description Keywords Extracted
ORG: []
EDUCATION: []
SKILLS: ['Python', 'Machine Learning', 'SQL', 'Communication']

✅ Common Skills Between Resume and Job Description:
['Python', 'SQL', 'Communication']
```

---

## 🧰 Technologies Used
| Library | Purpose |
|----------|----------|
| **spaCy** | Tokenization, POS tagging, Named Entity Recognition |
| **NLTK** | Stopword removal |
| **re (Regex)** | Text cleaning |
| **Python 3.12+** | Programming language |

---

## 🧩 NLP Concepts Demonstrated
- Text Cleaning & Preprocessing  
- Tokenization  
- Stopword Removal  
- Lemmatization  
- Named Entity Recognition (NER)  
- Rule-based Keyword Extraction  
- Skill Matching / Similarity  

---

## 📈 Future Enhancements
- Use **TF-IDF** or **Word2Vec** for smarter keyword extraction  
- Add **BERT embeddings** for semantic similarity  
- Build a **Streamlit web app** for interactive use  
- Automate **resume ranking** based on JD matching  

---

## 👨‍💻 Author
**Akshat Kumarey**  
📍 B.Tech Computer Engineering  
💬 GitHub: [@your-github-username](https://github.com/your-github-username)

---

## 📜 License
This project is licensed under the MIT License — feel free to use, modify, and share.
