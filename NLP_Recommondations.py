

#############
# Creating embeddings using BERT model for content based recommendations
# Embeddings are created from the CombinedText.csv 

import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast
from tqdm import tqdm
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)


# Load the datasets
recipes = pd.read_csv('recipes.csv')
sorted_recipes = pd.read_csv('sorted_recipes_features.csv')

# Select relevant columns from the original recipes dataset
relevant_columns = ['RecipeId', 'RecipeCategory', 'Keywords']
recipes_relevant = recipes[relevant_columns]

# Merge the relevant columns into the sorted_recipes dataset
sorted_recipes = sorted_recipes.merge(recipes_relevant, on='RecipeId', how='left')


# Function to parse 'c(..)' format to list
def parse_to_list(column):
    if pd.isna(column):
        return []
    if column.startswith('c(') and column.endswith(')'):
        try:
            list_string = column.replace('c(', '[').replace(')', ']')
            return ast.literal_eval(list_string)
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing string: {column}")
            return []
    else:
        return column.split(',')

# Apply the function to the Keywords column
sorted_recipes['KeywordsList'] = sorted_recipes['Keywords'].apply(parse_to_list)
sorted_recipes['IngredientsList'] = sorted_recipes['IngredientsList'].apply(eval)

# Create a combined text column
def combine_text(row):
    parts = [
        str(row['Name']),
        str(row['RecipeCategory']) if pd.notna(row['RecipeCategory']) else '',
        ' '.join(row['KeywordsList']),
        ' '.join(row['IngredientsList'])
    ]
    return ' '.join(parts)

sorted_recipes['CombinedText'] = sorted_recipes.apply(combine_text, axis=1)

# Save combined text for verification
sorted_recipes['CombinedText'].to_csv('CombinedText.csv', index=False)

# Function to get BERT embeddings for text
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

start_time = time.time()
# Apply sentiment analysis to each review with a progress bar
tqdm.pandas()
# Get embeddings for all recipes
sorted_recipes['Embeddings'] = sorted_recipes['CombinedText'].progress_apply(get_bert_embeddings)
sorted_recipes.to_csv("Embeddings2.csv", index=False)

# Function to get recipe recommendations based on user input
def get_recommendations(user_input, sorted_recipes):
    user_embedding = get_bert_embeddings([user_input])
    recipe_embeddings = np.vstack(sorted_recipes['Embeddings'].values)
    
    # Compute cosine similarity between user input and all recipes
    sim_scores = cosine_similarity(user_embedding, recipe_embeddings).flatten()
    
    # Get the indices of the most similar recipes
    sim_indices = sim_scores.argsort()[-10:][::-1]
    
    # Get the top 10 most similar recipes
    recommendations = sorted_recipes.iloc[sim_indices]
    
    return recommendations

# Example user input
user_input = "Fish"

# Get recommendations based on user input
recommendations = get_recommendations(user_input, sorted_recipes)

# Display the recommendations
print("Top Recommendations based on user input:")
print(recommendations[['RecipeId', 'Name', 'CombinedText']])

# Save the combined text and embeddings to a CSV file for future use
sorted_recipes.to_csv('recipes_with_embeddings.csv', index=False)

# Track the time taken
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")