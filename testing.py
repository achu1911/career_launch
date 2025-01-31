from transformers import pipeline

# Initialize the summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Example review text to summarize
review_text = """The professor is very knowledgeable and engaging. 
                 However, the assignments can be overwhelming, especially if you're not familiar with the subject matter. 
                 The grading is strict, but the final project is an opportunity to showcase what you've learned."""

# Get the summary
summary = summarizer(review_text, max_length=50, min_length=20, do_sample=False)

# Print the result
print("Summary:", summary[0]['summary_text'])
