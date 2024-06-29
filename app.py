import pandas as pd
import re
import string
import nltk
import contractions
import emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from flask import Flask, request, jsonify, render_template

# Load the dataset
dataframe1 = pd.read_csv("data/depress_text.csv", index_col=0)
dataframe2 = pd.read_csv("data/non_depress_text.csv", index_col=0)

# Gabungkan kedua DataFrame
data = pd.concat([dataframe1, dataframe2])

# Reset the index of the DataFrame
data = data.reset_index()

# Data Cleaning
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def clean_text(text):
    # Remove double quotation marks
    text = text.replace('"', '')
    # Remove RT tags
    text = re.sub(r'^RT[\s]+', '', text)
    # Remove user tags (@Usertag)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    # Remove URLs
    text = re.sub(r'https?:\/\/\S+', '', text)
    # Remove hashtags and keep only the text
    text = re.sub(r'#', '', text)
    # Remove emojis
    text = emoji.demojize(text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove word repetition
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    # Handle contractions
    text = contractions.fix(text)
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords, perform stemming and lemmatization
    tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in tokens if word not in stop_words]
    text = ' '.join(tokens)
    return text

data['Cleaned_Text'] = data['Text'].apply(clean_text)

# Separate the features (text) and labels (sentiment)
X = data['Cleaned_Text']
y = data['Sentiment']

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline for Data Processing and Modeling
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.8, max_features=5000)),
    ('feature_selection', SelectKBest(score_func=chi2, k='all')),
    ('model', LinearSVC(dual=False))
])

# Hyperparameter Tuning using GridSearchCV
parameters = {
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__max_features': [1000, 5000, 10000],
    'model__C': [0.1, 1, 10]
}

grid_search = GridSearchCV(pipeline, parameters, cv=5, error_score='raise')
grid_search.fit(X_train, y_train)

# Best Model and Parameters
best_model = grid_search.best_estimator_
best_parameters = grid_search.best_params_

# Make Predictions on the Test Set
y_pred = best_model.predict(X_test)


app = Flask(__name__)  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/anonymous')
def anonymous():
    return render_template('anonymous.html')

@app.route('/mental_health_articles')
def mental_health_articles():
    return render_template('mental_health_articles.html')

@app.route('/depression_articles')
def depression_articles():
    return render_template('depression_articles.html')

import csv
import os

def read_csv_file(filename):
    data = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

@app.route('/dashboard')
def dashboard():
    # Read the Sentiment_data.csv file
    csv_file_path = os.path.join(os.path.dirname(__file__), 'Sentiment_data.csv')
    data = read_csv_file(csv_file_path)

    # Extract the text lengths from the CSV data and convert them to numbers
    text_lengths = [len(row['Text']) for row in data]

    # Calculate the mean text length
    meanLength = sum(text_lengths) / len(text_lengths)
    return render_template('dashboard.html' , meanLength=meanLength, data=data)



@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    cleaned_text = clean_text(text)
    # Vectorize the cleaned text
    text_vectorized = best_model.named_steps['tfidf'].transform([cleaned_text])

    # Make the prediction
    prediction = best_model.named_steps['model'].predict(text_vectorized)  
    depression_label = "Not Depressed" if prediction[0] == 0 else "Depressed"
    # Prepare the response as JSON
    response = {
        'result': depression_label
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
