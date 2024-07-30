

#############
# Checking to see if reviews are in the proper format.
# Using SpacyTextBlob to conduct sentiment analysis of recipe reviews to be used along with review ratings to provide recipe recommendations based on reviews
# Using the mean sentiment of reviews for each recipe

import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from tqdm import tqdm
import time

# Load spaCy model and add SpacyTextBlob to the pipeline
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

# Load the dataset
reviews = pd.read_csv('reviews.csv')
# Ensure no missing values in the 'Review' column
reviews = reviews.dropna(subset=['Review'])  

# Define a function to check if the review is in the correct format
def is_valid_review(review):
    return isinstance(review, str) and len(review.strip()) > 0

# Filter out invalid reviews
reviews = reviews[reviews['Review'].apply(is_valid_review)]

# Perform sentiment analysis
def analyze_sentiment(review):
    doc = nlp(review)
    sentiment = doc._.blob.sentiment
    return sentiment.polarity, sentiment.subjectivity

# Track the time
start_time = time.time()

# Apply sentiment analysis to each review with a progress bar
tqdm.pandas()
reviews['Polarity'], reviews['Subjectivity'] = zip(*reviews['Review'].progress_apply(analyze_sentiment))

# Normalize ratings to be between -1 and 1
reviews['Normalized_Rating'] = reviews['Rating'] / 5 * 2 - 1

# Combine the normalized rating with polarity to create a combined sentiment score
reviews['Combined_Sentiment'] = (reviews['Polarity'] + reviews['Normalized_Rating']) / 2

# Calculate the mean combined sentiment for each RecipeId
mean_sentiment = reviews.groupby('RecipeId').agg({
    'Combined_Sentiment': 'mean',
    'Polarity': 'mean',
    'Subjectivity': 'mean'
}).reset_index()

# Track the time taken
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

# Display the mean sentiment DataFrame
print(mean_sentiment)

mean_sentiment.to_csv('review_sentiments', index = False)

