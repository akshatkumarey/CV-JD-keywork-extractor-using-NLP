# 🧠 Resume and Job Description Keyword Extractor using NLP

## 📌 Project Overview
This mini-project demonstrates how **Natural Language Processing (NLP)** can be applied to extract and compare key information from textual data — specifically **resumes** and **job descriptions**.

The system automatically identifies important keywords such as:
- **Skills**
- **Education details**
- **Experience**
- **Organizations**
- **Common skills** between resume and job description

This project showcases foundational NLP techniques including **text preprocessing**, **tokenization**, **named entity recognition (NER)**, and **rule-based keyword extraction**.

---

## 🚀 Features
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
