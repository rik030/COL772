# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:36:29 2023

@author: rishav
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='Path to the data.csv file')
parser.add_argument('--model_dir', type=str, required=True, help='Path to the trained_model directory')
args = parser.parse_args()

data_path = args.data_path
model_dir = args.model_dir

# Load the data into a pandas dataframe
df = pd.read_csv(data_path, header=None).dropna()

# Print the first 5 rows of the data
print(df.head())

# Plot a bar graph to show the distribution of the rating classes
sns.countplot(x=df[1], data=df)
plt.title("Rating class distribution")
plt.show()

# Plot a histogram to show the distribution of the length of the text data
df["text_length"] = df[0].apply(len)
sns.histplot(df["text_length"])
plt.title("Text length distribution")
plt.show()

import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Dividing it into X and Y
X,y=df[0], df[1].values

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
df['text'] = X.apply(preprocess)

X_arr=np.array(df['text'])

# Initialize the KFold cross-validation object
kf = KFold(n_splits=5, shuffle=True)

# Initialize a list to store the accuracy scores
scores = []
best_model = None
tfIdVec= None
best_score = 0.0

# Loop through the splits
for train_index, test_index in kf.split(X_arr):
    # Split the data into training and testing sets
    
    X_train, X_test = X_arr[train_index], X_arr[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Convert the text to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1,3), use_idf=True, lowercase=False, smooth_idf=True)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    
    # Train the model
    model = LinearSVC(class_weight='balanced', C=0.1)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate the accuracy score
    score = f1_score(y_test, y_pred, average='weighted')
    
    # Add the score to the list
    scores.append(score)
    
    # Save the best model
    if score > best_score:
        best_score = score
        best_model = model
        tfIdVec = vectorizer

# Calculate the average accuracy score
average_score = np.mean(scores)

# Print the average accuracy score
print("Average accuracy:", average_score)
print("Best accuracy:", best_score)

#making directory folder
os.makedirs(model_dir, exist_ok=True)
    
#Save the vectorizer and model in trained_model directory

with open(os.path.join(model_dir, '2018CH70302_model.pickle'), "wb") as file:
    pickle.dump(best_model, file)

with open(os.path.join(model_dir, '2018CH70302_vectorizer.pickle'), "wb") as file:
    pickle.dump(tfIdVec, file) 
