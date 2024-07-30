import pandas as pd
import ast
import numpy as np


#################
# This chunk selects the relevant columns, removes rows with NaN values,
# and filters rows based on the expected format for ingredient lists (removes invalid formats).
# It then parses the cleaned data and combines quantities and ingredients into a single string.
# Provides dataset with all consistent recipes (ingredients that properly match the quantities)

# Load the dataset
recipes = pd.read_csv('recipes.csv')

# Select the relevant columns 
data = recipes[['RecipeId', 'Name', 'RecipeIngredientQuantities', 'RecipeIngredientParts']]

# Create a new DataFrame with the selected columns
recipes_data = pd.DataFrame(data)

# Remove rows with NaN values  
recipes_data = recipes_data.dropna(subset=['RecipeIngredientQuantities', 'RecipeIngredientParts'])

# Function to check if the format of the ingredient lists is valid
def is_valid_format(series):
    overall_format = series.str.startswith('c(') & series.str.endswith(')')
    return overall_format

# Apply the format validation to both columns and filter the rows
valid_quantities = is_valid_format(recipes_data['RecipeIngredientQuantities'])
valid_parts = is_valid_format(recipes_data['RecipeIngredientParts'])

recipes_cleaned = recipes_data[valid_quantities & valid_parts].copy()


# Function to parse a string into a list
def parse_list(string):
    try:
        list_string = string.replace('c(', '[').replace(')', ']').replace('NA', 'None')
        return ast.literal_eval(list_string)
    except Exception as e:
        print(f"Error parsing string: {string}")
        return None
    
# Apply the parsing function to 'RecipeIngredientQuantities' and 'RecipeIngredientParts'
recipes_cleaned['QuantitiesList'] = recipes_cleaned['RecipeIngredientQuantities'].apply(parse_list)
recipes_cleaned['IngredientsList'] = recipes_cleaned['RecipeIngredientParts'].apply(parse_list)



# Function to check if the lengths of quantities and ingredients match
def check_length_match(row):
    return len(row['QuantitiesList']) == len(row['IngredientsList'])

# Identify rows with mismatched lengths
inconsistent_rows = recipes_cleaned[~recipes_cleaned.apply(check_length_match, axis=1)]

# Print inconsistent rows for inspection
print("# of Inconsistent Rows:")
print(len(inconsistent_rows[['RecipeIngredientQuantities', 'RecipeIngredientParts', 'QuantitiesList', 'IngredientsList']]))

# Remove inconsistent rows
recipes_consistent = recipes_cleaned[recipes_cleaned.apply(check_length_match, axis=1)].copy()

# Function to combine quantities and ingredients into a single string
def combine_quantities_ingredients(quantities, ingredients):
    combined = [f"{quantity} {ingredient}" for quantity, ingredient in zip(quantities, ingredients)]
    return ' '.join(combined)

# Apply the combining function to create the 'ingredients' column
recipes_consistent['ingredients'] = recipes_consistent.apply(
    lambda row: combine_quantities_ingredients(row['QuantitiesList'], row['IngredientsList']), axis=1
)

# Display the transformed DataFrame
print("\nTransformed DataFrame after removing inconsistent rows:")
recipes_consistent.to_csv('consistent_recipes', index = False)





##############

# This chunk does the same thing above except it keeps the mismatched rows (dataset with all recipes).
# It calculates the number of reviews for each recipe and merges this data with the recipes data.
# It then sorts the recipes by the number of reviews and saves the sorted dataset.

import pandas as pd
import ast
import numpy as np

# Load the dataset
recipes = pd.read_csv('recipes.csv')

# Select the relevant columns including RecipeName and RecipeId
data = recipes[['RecipeId', 'Name', 'RecipeIngredientQuantities', 'RecipeIngredientParts']]

# Create a new DataFrame with the selected columns
recipes_data = pd.DataFrame(data)

# Remove rows with NaN values before parsing
recipes_data = recipes_data.dropna(subset=['RecipeIngredientQuantities', 'RecipeIngredientParts'])

#  Remove rows not in the expected format and parse the cleaned data
def is_valid_format(series):
    overall_format = series.str.startswith('c(') & series.str.endswith(')')
    return overall_format

# Apply the format validation to both columns and filter the rows
valid_quantities = is_valid_format(recipes_data['RecipeIngredientQuantities'])
valid_parts = is_valid_format(recipes_data['RecipeIngredientParts'])

recipes_full = recipes_data[valid_quantities & valid_parts].copy()


# Function to parse a string into a list
def parse_list(string):
    try:
        list_string = string.replace('c(', '[').replace(')', ']').replace('NA', 'None')
        return ast.literal_eval(list_string)
    except Exception as e:
        print(f"Error parsing string: {string}")
        return None

# Apply the parsing function to 'RecipeIngredientQuantities' and 'RecipeIngredientParts'
recipes_full['QuantitiesList'] = recipes_full['RecipeIngredientQuantities'].apply(parse_list)
recipes_full['IngredientsList'] = recipes_full['RecipeIngredientParts'].apply(parse_list)

# Function to combine quantities and ingredients into a single string
def combine_quantities_ingredients(quantities, ingredients):
    combined = [f"{quantity} {ingredient}" for quantity, ingredient in zip(quantities, ingredients)]
    return ' '.join(combined)

# Apply the combining function to create the 'ingredients' column
recipes_full['ingredients'] = recipes_full.apply(
    lambda row: combine_quantities_ingredients(row['QuantitiesList'], row['IngredientsList']), axis=1
)

# Load reviews data
reviews = pd.read_csv('reviews.csv')

# Calculate the number of reviews for each recipe
review_counts = reviews['RecipeId'].value_counts().reset_index()
review_counts.columns = ['RecipeId', 'ReviewCount']

# Merge review counts with the recipes data
recipes_full = recipes_full.merge(review_counts, on='RecipeId', how='left')
recipes_full['ReviewCount'] = recipes_full['ReviewCount'].fillna(0)

# ESorting recipes by review count
sorted_recipes = recipes_full.sort_values(by='ReviewCount', ascending=False)

# Display sorted recipes
print("\nSorted Recipe Dataset by Review Counts")
sorted_recipes.to_csv('sorted_recipes_by_review_count.csv', index=False)



############
# This chunk converts the 'TotalTime' column from ISO 8601 duration format to total minutes.
# It also ensures the nutritional columns are present and properly formatted,
# and merges the recipes data with nutritional information.
# Creates a dataset with these features to be used for content based recommendations.


import pandas as pd
import numpy as np
from isodate import parse_duration

# Load the recipes dataset
recipes = pd.read_csv('recipes.csv') 

sorted_recipes = pd.read_csv('sorted_recipes_by_review_count.csv')

# Function to convert ISO 8601 duration to total minutes
def convert_duration(duration):
    try:
        return int(parse_duration(duration).total_seconds() / 60)
    except Exception as e:
        print(f"Error parsing duration: {duration}")
        return None

recipes.loc[recipes["TotalTime"] == 'PT-2M', "TotalTime"] = 'PT1H5M'

# Apply the conversion to the TotalTime column
recipes['TotalTime'] = recipes['TotalTime'].apply(convert_duration)

# Ensure the nutritional columns are present in dataset
nutritional_columns = ['Calories', 'ProteinContent', 'FatContent', 'CarbohydrateContent']
recipes[nutritional_columns] = recipes[nutritional_columns].apply(pd.to_numeric, errors='coerce')

recipes = recipes[['RecipeId', 'TotalTime'] + nutritional_columns]

# Merge sorted_recipes with the selected columns from recipes based on RecipeId
merged_recipes = pd.merge(sorted_recipes, recipes, on='RecipeId', how='left')

print(merged_recipes[['RecipeId', 'Name', 'TotalTime'] + nutritional_columns].head())

# Save the merged DataFrame to a new CSV file
merged_recipes.to_csv('sorted_recipes_features.csv', index=False)
