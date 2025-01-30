from transformers import pipeline
import pandas as pd
import requests
import json
import csv


class HuggingFaceSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Initializes the summarizer with the given Hugging Face model.
        """
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize_reviews(self, reviews, max_chunk_size=800):
        """
        Summarizes a list of product reviews using a Hugging Face model.

        """
        combined_reviews = " ".join(reviews)
        chunks = self.chunk_text(combined_reviews, max_chunk_size)
        summaries = []
        for chunk in chunks:
            summary = self.summarizer(chunk, max_length=100, min_length=40, do_sample=False)
            summaries.append(summary[0]['summary_text'])

        return " ".join(summaries)

    def chunk_text(self, text, max_length):
        """Splits text into chunks of a specified maximum length."""
        words = text.split()
        return [" ".join(words[i:i + max_length]) for i in range(0, len(words), max_length)]

    def paraphrase_text(self, text, length):
        model = "facebook/bart-large-cnn"
        paraphraser = pipeline("summarization", model=model)
        paraphrased = paraphraser(text, max_length=200, min_length=60, do_sample=False)
        return paraphrased[0]['summary_text']




if __name__ == "__main__":
    reviews =[]
    url = "https://planetterp.com/api/v1/course"

    request= input("Which course do you want to review?")
    request2 = input("Please enter professor")
    params = {
        "name":request,
        "reviews": "true"

    }
    r = requests.get(url, params = params)
    if r.status_code == 200:
        data = r.json()  
        reviews= data['reviews']
        #print(reviews) 
    else:
        print("bad input parameter")

    with open('course_reviews.csv', mode='w', newline='', encoding='utf-8') as file:
            
            
            fieldnames = ['Course Name', 'Professor', 'Rating', 'Review', 'Expected Grade', 'Date']
            
            
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            
            writer.writeheader()

            for review in reviews:
                writer.writerow({
                    'Course Name': review['course'],  
                    'Professor': review['professor'],  
                    'Rating': review['rating'],  
                    'Review': review['review'],  
                    'Expected Grade': review['expected_grade'],
                    'Date': review['created']

                })

    df = pd.read_csv('course_reviews.csv')
    #print(df['Review'])



    filtered_df= df[df['Professor'] == request2]


    reviews = filtered_df['Review'].to_numpy().tolist()
    
    

    summarizer = HuggingFaceSummarizer()
    summary = summarizer.summarize_reviews(reviews)
    paraphrased = summarizer.paraphrase_text(summary, len(summary))
    #print("Summary:", summary)
    print("Paraphrased:", paraphrased)
    
