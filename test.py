# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:36:29 2023

@author: rishav
"""
import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True, help='Path to the trained_model directory')
parser.add_argument('--data_path', type=str, required=True, help='Path to the data.csv file')
parser.add_argument('--save_path', type=str, required=True, help='Path to the output.csv file')
args = parser.parse_args()

data_path = args.data_path
model_dir = args.model_dir
save_file=args.save_path

# Load the data into a pandas dataframe
df = pd.read_csv(data_path, header=None).dropna()

# Print the first 5 rows of the data
print(df.head())


import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# Preprocessing function to perform lemmatization, stop word removal, and POS tagging
def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    pos_tags = [tag[1] for tag in pos_tags]
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(tokens + pos_tags)

# Apply the preprocessing function to the text data
df['text'] = df[0].apply(preprocess)


# Load the best model from disk
with open(os.path.join(model_dir, '2018CH70302_model.pickle'), "rb") as file:
    model = pickle.load(file)

# Transform the test data
with open(os.path.join(model_dir, '2018CH70302_vectorizer.pickle'), "rb") as file:
    vectorizer = pickle.load(file)
    
X_test = vectorizer.transform(df["text"])

# Make predictions on the test data
y_pred = model.predict(X_test)

# Add the predictions to the test data dataframe
df["prediction"] = y_pred

# Save the results to a csv file
df['prediction'].to_csv(save_file, header=None,index=False)
