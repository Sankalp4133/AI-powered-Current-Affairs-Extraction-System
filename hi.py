import os
import re
import pytesseract
from flask import Flask, request, render_template, send_file
from pdf2image import convert_from_path
import joblib

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
INTERMEDIATE_FOLDER = "intermediate"
RESULT_FOLDER = "results"

# Load the model and vectorizer
model = joblib.load("improved_current_affairs_classifier (2).pkl")
vectorizer = joblib.load("improved_vectorizer (2).pkl")

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INTERMEDIATE_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Utility Functions

def clean_text(text):
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def predict_article(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    pred = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0][pred]
    label = "✅ Current Affair" if pred == 1 else "❌ Not Current Affair"
    return f"{label} [Score: {round(proba, 2)}]"


def extract_text_using_tesseract(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    full_text = ""
    for img in images:
        text = pytesseract.image_to_string(img, lang='eng')
        full_text += text + "\n\n"
    return full_text


def fix_broken_sentences(ocr_text):
    fixed_text = ""
    paragraphs = ocr_text.split("\n\n")
    for para in paragraphs:
        lines = para.split("\n")
        joined = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if joined and not joined.endswith((".", "!", "?", ":", ";")):
                joined += " " + line
            else:
                joined += "\n" + line
        fixed_text += joined.strip() + "\n\n"
    return fixed_text


def separate_articles(cleaned_text):
    lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
    articles = []
    current_article = []
    for line in lines:
        if line.isupper() or re.match(r"^[A-Z][A-Z\s:,\-'&]{5,}$", line):
            if current_article:
                articles.append("\n".join(current_article))
                current_article = [line]
            else:
                current_article.append(line)
        else:
            current_article.append(line)
    if current_article:
        articles.append("\n".join(current_article))
    return articles


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if uploaded_file and uploaded_file.filename.endswith(".pdf"):
            pdf_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(pdf_path)

            extracted_text = extract_text_using_tesseract(pdf_path)
            fixed_text = fix_broken_sentences(extracted_text)
            articles = separate_articles(fixed_text)

            output_lines = []
            for article in articles:
                prediction = predict_article(article)
                output = f"{prediction}: {article}\n"
                output_lines.append(output)

            output_text = "\n".join(output_lines)
            output_file = os.path.join(RESULT_FOLDER, "classified_articles.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output_text)

            return render_template("result.html", output=output_text, filename="classified_articles.txt")

    return render_template("index.html")

@app.route("/download/<filename>")
def download_file(filename):
    file_path = os.path.join(RESULT_FOLDER, filename)
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
