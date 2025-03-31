from flask import Flask, render_template, request, redirect, url_for
import os
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

global_ranked_results = []

@app.route("/")
def home():
    return render_template("a.html")

@app.route("/ai_resume_screening")
def ai_resume_screening():
    return render_template("ai_resume_screening.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/results")
def results():
    return render_template("results.html", ranked_results=global_ranked_results)

@app.route("/detailed_breakdown/<int:rank>")
def detailed_breakdown(rank):
    if 1 <= rank <= len(global_ranked_results):
        resume = global_ranked_results[rank - 1]
        return render_template("detailed_breakdown.html", resume=resume)
    return redirect(url_for("results"))

@app.route("/process_resumes", methods=["POST"])
def process_resumes():
    job_desc = request.form.get("job_desc")
    uploaded_files = request.files.getlist("resumes")

    if not job_desc or not uploaded_files:
        return redirect(url_for("ai_resume_screening"))

    file_paths = []
    for file in uploaded_files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        file_paths.append(file_path)

    # Reset the ranked results for new processing
    global global_ranked_results
    global_ranked_results = rank_resumes(job_desc, file_paths)

    print("Updated Ranked Results:", global_ranked_results)  # Debugging

    return redirect(url_for("results"))

def extract_text(file_path):
    ext = file_path.split(".")[-1].lower()
    if ext == "pdf":
        reader = PdfReader(file_path)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()]).lower()
    elif ext == "docx":
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs]).lower()
    elif ext in ["jpg", "jpeg", "png"]:
        img = Image.open(file_path)
        return pytesseract.image_to_string(img).lower()
    return ""

# Keywords for skills and job titles
SKILLS_KEYWORDS = ["HTML", "CSS", "JavaScript", "React", "Angular", "Web Development", "Problem-solving", "Debugging", "Teamwork", "Leadership", "User Experience"]
JOB_TITLES_KEYWORDS = ["Front-End Developer", "Software Engineer", "Web Developer"]

def extract_entities(text):
    skills = [word for word in SKILLS_KEYWORDS if word.lower() in text]
    job_titles = [word for word in JOB_TITLES_KEYWORDS if word.lower() in text]
    return skills, job_titles

def infer_job_title(skills):
    if "React" in skills or "JavaScript" in skills:
        return "Front-End Developer"
    elif "Web Development" in skills:
        return "Web Developer"
    else:
        return "General IT Role"

def rank_resumes(job_desc, file_paths):
    job_desc_text = job_desc.lower()
    resumes_data = []
    total_skills = len(SKILLS_KEYWORDS)

    resume_texts = []  # Store all extracted resume texts
    resume_filenames = []  # Corresponding filenames

    # 🟢 Extract text from all resumes
    for file_path in file_paths:
        resume_text = extract_text(file_path)
        if resume_text:  # Only store valid resumes
            resume_texts.append(resume_text)
            resume_filenames.append(os.path.basename(file_path))

    if not resume_texts:
        return []  # Return empty if no valid resumes were processed

    # 🟢 FIX: Use TF-IDF to compare job description against all resumes in **one batch**
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([job_desc_text] + resume_texts)
    similarities = cosine_similarity(vectors[0], vectors[1:])[0] * 100  # Compare job_desc with all resumes

    for i, resume_text in enumerate(resume_texts):
        skills, job_titles = extract_entities(resume_text)

        suggested_title = job_titles[0] if job_titles else infer_job_title(skills)
        experience_match = 80 if "project" in resume_text else 50
        additional_match = 60 if "teamwork" in resume_text or "leadership" in resume_text else 40
        education_match = 100 if "bachelor" in resume_text or "master" in resume_text else 50
        certificate_bonus = 20 if any(word in resume_text for word in ["certificate", "certified", "achievement"]) else 0

        overall_match = round(
            (0.35 * similarities[i]) +
            (0.25 * (len(skills) / total_skills) * 100) +
            (0.15 * experience_match) +
            (0.1 * additional_match) +
            (0.1 * education_match) +
            (0.05 * certificate_bonus), 2
        )

        resumes_data.append({
            "filename": resume_filenames[i],
            "overall_match": overall_match,
            "suggested_title": suggested_title,
            "skills": skills,
            "achievements": ["Achievement 1", "Achievement 2"],
            "experience": ["Experience 1", "Experience 2"],
            "certificates": ["Certificate 1", "Certificate 2"],
        })

    # 🟢 FIX: Ensure sorting before ranking
    resumes_data.sort(key=lambda x: x["overall_match"], reverse=True)

    # Assign ranks after sorting
    for i, resume in enumerate(resumes_data):
        resume["rank"] = i + 1

    return resumes_data

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
