from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import json
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from werkzeug.utils import secure_filename
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
from fuzzywuzzy import fuzz
import re

# --- New imports ---
import openai
import dotenv

dotenv.load_dotenv()  # load environment variables from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

global_ranked_results = []

job_data = []
job_title_classifier = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return ' '.join(text.split())

# --- New function to enrich job descriptions using OpenAI ---
def enrich_job_with_openai(description):
    prompt = f"""
Extract skills, education, and experience needed from the following job description. Return JSON with keys: skills, experience, education.

Job Description:
\"\"\"{description}\"\"\"
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response['choices'][0]['message']['content']
        parsed = json.loads(content)
        return {
            "skills": [normalize_text(s) for s in parsed.get("skills", [])],
            "experience": [normalize_text(e) for e in parsed.get("experience", [])],
            "education": [normalize_text(ed) for ed in parsed.get("education", [])]
        }
    except Exception as e:
        print("[ERROR] OpenAI job enrichment failed:", e)
        return {"skills": [], "experience": [], "education": []}

def retrain_model(enrich_jobs=False):
    global job_data, job_title_classifier
    try:
        with open("job_data.json", "r") as f:
            job_data = json.load(f)

        for job in job_data:
            # Optionally enrich job using OpenAI (set enrich_jobs=True to enable)
            if enrich_jobs:
                enriched = enrich_job_with_openai(job["description"])
                job["skills"] = enriched["skills"]
                job["experience"] = enriched["experience"]
                job["education"] = enriched["education"]

            for field in ["skills", "experience", "education"]:
                job[field] = [normalize_text(item) for item in job.get(field, [])]

        X_train = [job["description"] for job in job_data]
        y_train = [job["job_title"] for job in job_data]
        job_title_classifier.fit(X_train, y_train)
        print("[INFO] Model retrained successfully with", len(job_data), "jobs")
    except Exception as e:
        print(f"[ERROR] Failed to retrain model: {e}")

class JobDataChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if "job_data.json" in event.src_path:
            print("[INFO] job_data.json modified. Reloading and retraining model...")
            retrain_model()

def start_file_watcher():
    event_handler = JobDataChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()
    threading.Thread(target=observer.join, daemon=True).start()

retrain_model()  # default: no enrichment here to avoid repeated API calls on startup
start_file_watcher()

def extract_text(file_path):
    ext = file_path.split(".")[-1].lower()
    try:
        if ext == "pdf":
            reader = PdfReader(file_path)
            text = " ".join(page.extract_text() or "" for page in reader.pages)
        elif ext == "docx":
            doc = Document(file_path)
            text = "\n".join(para.text for para in doc.paragraphs)
        elif ext in ["jpg", "jpeg", "png"]:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
        else:
            return ""
        return normalize_text(text)
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
    return ""

def keyword_match(resume_text, keywords):
    matches = 0
    resume_lines = resume_text.splitlines()
    normalized_resume_text = normalize_text(resume_text)

    for keyword in keywords:
        normalized_kw = normalize_text(keyword)

        # Exact phrase match
        if normalized_kw in normalized_resume_text:
            matches += 1
            continue

        # Partial match in lines
        for line in resume_lines:
            if normalized_kw in normalize_text(line):
                matches += 0.7
                break

        # Fuzzy match with each word
        for resume_word in normalized_resume_text.split():
            if fuzz.ratio(normalized_kw, resume_word) > 75:
                matches += 0.5
                break

    return matches

def compute_match_score(resume_text, job_info):
    if not resume_text.strip():
        return 0, 0, 0, 0

    skill_matches = keyword_match(resume_text, job_info["skills"])
    exp_matches = keyword_match(resume_text, job_info["experience"])
    edu_matches = keyword_match(resume_text, job_info["education"])

    skill_score = (skill_matches / len(job_info["skills"])) * 100 if job_info["skills"] else 0
    exp_score = (exp_matches / len(job_info["experience"])) * 100 if job_info["experience"] else 0
    edu_score = (edu_matches / len(job_info["education"])) * 100 if job_info["education"] else 0

    total_score = round(skill_score * 0.5 + exp_score * 0.3 + edu_score * 0.2, 2)
    return total_score, skill_score, exp_score, edu_score

def infer_job_title_from_resume(resume_text):
    if not resume_text.strip():
        return "Unknown", []

    predicted_title = job_title_classifier.predict([resume_text])[0]
    matched_keywords = []

    for job in job_data:
        for kw in job["skills"] + job["experience"] + job["education"]:
            if kw in resume_text:
                matched_keywords.append(kw)

    suggested_titles = list({
        job["job_title"]
        for job in job_data
        if any(kw in matched_keywords for kw in job["skills"] + job["experience"] + job["education"])
    })

    if not suggested_titles:
        suggested_titles = [predicted_title]

    return predicted_title, suggested_titles

def generate_suggestions(job_title, resume_text):
    job_info = next((job for job in job_data if job["job_title"] == job_title), None)
    if not job_info:
        return ["⚠️ No job criteria available for this role."]

    def missing(items):
        return [item for item in items if normalize_text(item) not in resume_text]

    suggestions = []
    if missing_skills := missing(job_info["skills"]):
        suggestions.append("Skills to improve: " + ", ".join(missing_skills))
    if missing_education := missing(job_info["education"]):
        suggestions.append("Education to add: " + ", ".join(missing_education))
    if missing_experience := missing(job_info["experience"]):
        suggestions.append("Experience to gain: " + ", ".join(missing_experience))

    return suggestions if suggestions else ["✅ Resume meets all key criteria."]

def extract_additional_sections(text):
    additional = {}
    patterns = {
        "certificates": r"certificates?[:\-\n](.*?)(?:\n\n|$)",
        "achievements": r"achievements?[:\-\n](.*?)(?:\n\n|$)",
        "additional_skills": r"additional skills?[:\-\n](.*?)(?:\n\n|$)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            additional[key] = match.group(1).strip()
        else:
            additional[key] = "Not found"
    return additional

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
        return render_template("detailed_breakdown.html", resume=global_ranked_results[rank - 1])
    return redirect(url_for("results"))

@app.route("/process_resumes", methods=["POST"])
def process_resumes():
    job_desc = request.form.get("job_desc")
    uploaded_files = request.files.getlist("resumes")

    if not job_desc or not uploaded_files or uploaded_files[0].filename == '':
        return redirect(url_for("ai_resume_screening"))

    file_paths = []
    for file in uploaded_files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        file_paths.append(file_path)

    global global_ranked_results
    global_ranked_results = rank_resumes(job_desc, file_paths)

    return redirect(url_for("results"))

def rank_resumes(job_desc, file_paths):
    resumes_data = []

    for file_path in file_paths:
        resume_text = extract_text(file_path)
        if not resume_text.strip():
            continue

        predicted_title, suggested_titles = infer_job_title_from_resume(resume_text)
        suggested_title = predicted_title

        best_job = None
        best_score = 0

        for job in job_data:
            score, _, _, _ = compute_match_score(resume_text, job)
            if score > best_score:
                best_score = score
                best_job = job
                suggested_title = job["job_title"]

        if best_job:
            overall_match, skill_score, exp_score, edu_score = compute_match_score(resume_text, best_job)
            suggestions = generate_suggestions(suggested_title, resume_text)
            status = (
                "✅ Strong Match" if overall_match >= 75 else
                "⚠️ Partial Match" if overall_match >= 50 else
                "❌ Weak Match"
            )

            def split_present_missing(items):
                return {
                    "present": [{"name": i, "positive": True} for i in items if normalize_text(i) in resume_text],
                    "missing": [{"name": i, "positive": False} for i in items if normalize_text(i) not in resume_text]
                }

            resume_data = {
                "rank": 0,
                "filename": os.path.basename(file_path),
                "overall_match": overall_match,
                "status": status,
                "suggested_title": suggested_title,
                "predicted_title": predicted_title,
                "alternative_titles": suggested_titles,
                "suggestions": suggestions,
                "skill_match": f"{skill_score:.1f}%",
                "exp_match": f"{exp_score:.1f}%",
                "edu_match": f"{edu_score:.1f}%",
                "technical_skills": split_present_missing(best_job["skills"]),
                "relevant_experience": split_present_missing(best_job["experience"]),
                "education_required": split_present_missing(best_job["education"]),
                "certificates": split_present_missing([best_job.get("certificates", "certificates")]).get("missing", []),
                "achievements": split_present_missing([best_job.get("achievements", "achievements")]).get("missing", []),
                "additional_skills": split_present_missing([best_job.get("additional_skills", "additional skills")]).get("missing", []),
            }
        else:
            resume_data = {
                "rank": 0,
                "filename": os.path.basename(file_path),
                "overall_match": 0,
                "status": "❓ No Matching Job Found",
                "suggested_title": "Unknown",
                "predicted_title": predicted_title,
                "alternative_titles": suggested_titles,
                "suggestions": ["⚠️ No matching job title in our database"],
                "skill_match": "N/A",
                "exp_match": "N/A",
                "edu_match": "N/A",
                "technical_skills": {"present": [], "missing": []},
                "relevant_experience": {"present": [], "missing": []},
                "education_required": {"present": [], "missing": []},
                "certificates": [],
                "achievements": [],
                "additional_skills": [],
            }

        resumes_data.append(resume_data)

    resumes_data.sort(key=lambda x: x["overall_match"], reverse=True)
    for i, resume in enumerate(resumes_data):
        resume["rank"] = i + 1

    return resumes_data


# --- New route to add a job with OpenAI enrichment ---
@app.route("/add_job", methods=["POST"])
def add_job():
    # Expect JSON with job_title and description fields
    data = request.json
    if not data or "job_title" not in data or "description" not in data:
        return jsonify({"error": "job_title and description required"}), 400

    new_job = {
        "job_title": data["job_title"],
        "description": data["description"]
    }

    # Use OpenAI to enrich new job
    enriched = enrich_job_with_openai(new_job["description"])
    new_job.update(enriched)

    # Normalize all fields
    for field in ["skills", "experience", "education"]:
        new_job[field] = [normalize_text(item) for item in new_job.get(field, [])]

    # Load existing data, append new job, save back
    try:
        if os.path.exists("job_data.json"):
            with open("job_data.json", "r") as f:
                existing_jobs = json.load(f)
        else:
            existing_jobs = []

        existing_jobs.append(new_job)

        with open("job_data.json", "w") as f:
            json.dump(existing_jobs, f, indent=2)

        retrain_model()  # retrain model with updated data

        return jsonify({"message": "Job added and model retrained", "job": new_job}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
