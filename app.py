from flask import Flask, request, jsonify
import requests
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class HuggingFaceSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize_reviews(self, reviews, max_chunk_size=800):
        combined_reviews = " ".join(reviews)
        
        if len(combined_reviews.split()) > max_chunk_size:
            print("Input too long, chunking into smaller parts.")
            chunks = self.chunk_text(combined_reviews, max_chunk_size)
            summaries = [self.summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text'] for chunk in chunks]
            return " ".join(summaries)
        else:
            summary = self.summarizer(combined_reviews, max_length=150, min_length=50, do_sample=False)
            return summary[0]['summary_text']

    def chunk_text(self, text, max_length):
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            if len(" ".join(current_chunk + [word]).split()) > max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
            else:
                current_chunk.append(word)

        if current_chunk:
            chunks.append(" ".join(current_chunk))  

        return chunks


def fetch_reviews(course_name, professor=None):
    url = "https://planetterp.com/api/v1/course"
    params = {"name": course_name, "reviews": "true"}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Error fetching reviews: {response.status_code}")
        return []

    data = response.json()
    reviews = data.get("reviews", [])

    if professor:
        reviews = [r for r in reviews if r.get("professor") == professor]
    
    review_texts = [r.get("review", "") for r in reviews]
    
    return review_texts

@app.route("/summarize", methods=["GET"])
def summarize():
    course_name = request.args.get("course")
    professor = request.args.get("professor")

    if not course_name:
        return jsonify({"error": "Course name is required"}), 400
    if not professor:
        return jsonify({"error": "Professor name is required"}), 400

    reviews = fetch_reviews(course_name, professor)
    if not reviews:
        return jsonify({"message": "No reviews found for this course and professor"})

    summarizer = HuggingFaceSummarizer()
    summary = summarizer.summarize_reviews(reviews)
    
    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(debug=True)
