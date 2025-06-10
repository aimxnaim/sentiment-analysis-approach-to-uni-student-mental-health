# SereneMinds

SereneMinds is a Flask web application that performs sentiment analysis on text related to university student mental health. A linear SVM classifier is trained on tweets labelled as depressed or non‑depressed. The project includes a web interface to anonymously submit text, visualise dataset statistics and read articles.

## Repository Layout

- **app.py** – main Flask application and machine learning pipeline
- **data/** – CSV datasets used for training
- **sentiment-analysis/** – additional notebooks and datasets
- **templates/** – HTML templates for the web interface
- **static/** – static assets used by the templates
- **Visualization/** – Jupyter notebook for dataset visualisation

## Dataset

Two CSV files are provided in the `data` folder. The first contains posts labelled as depressed:

```csv
Text,Sentiment
I'm losing my will to live.,1
The fear of asking for extensions or accommodations due to the stigma attached to mental health issues. #StigmaInAcademia,1
"The fear of failure and disappointing others, leading to paralyzing academic anxiety. #FearOfFailure",1
I can't escape the grip of this deep sadness.,1
"""Depression has affected my self-esteem and made me believe I'm not worthy of love or happiness."" #SelfEsteemStruggles",1
```

The second lists non‑depressed examples:

```csv
Text,Sentiment
"""I had a fantastic day at a food festival, savoring a variety of cuisines, sampling gourmet dishes, and experiencing culinary delights.""",0
"""Your dreams are worth pursuing. Stay committed, work hard, and never give up.""",0
"""The universe has a way of aligning things in divine timing. Trust the process and have faith in the journey.""",0
@hi_sweetye I hope so,0
"""Each day is an opportunity to create a life that aligns with my values, passions, and desires. I have the power to shape my own destiny.""",0
```

These are combined into `Sentiment_data.csv` and also used to generate visualisations shown on the dashboard.

## Pipeline

`app.py` performs data loading, cleaning and model training. A portion of the code is shown below:

```python
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
```

```python
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
```

```python
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
```

## Usage

1. Install dependencies (for example using `pip`):
   ```bash
   pip install flask pandas scikit-learn nltk contractions emoji
   python -m nltk.downloader punkt stopwords wordnet
   ```
2. Run the application:
   ```bash
   python app.py
   ```
3. Open `http://localhost:5000/` in your browser.

### Anonymous Analysis

The `anonymous.html` template contains a form to submit text anonymously:

```html
<section data-bs-version="5.1" class="form1 cid-tJSrDf7bMl mbr-parallax-background" id="form1-1u">
    <div class="mbr-overlay" style="opacity: 0.6; background-color: rgb(255, 255, 255);"></div>
    <div class="container-fluid">
        <div class="row justify-content-center">
            <div class="col-lg-8 mx-auto mbr-form">
                <form id="prediction-form" class="mbr-form form-with-styler" data-verified="">
                    <div class="row">
                        <div hidden="hidden" data-form-alert-danger="" class="alert alert-danger col-12">
                            Oops...! some problem!
                        </div>
                    </div>
                    <div class="dragArea row">
                        <div class="col-12">
                            <h1 class="mbr-section-title mb-4 mbr-fonts-style align-center display-2"><strong>Lets
                                    get started</strong></h1>
                        </div>
                        <div class="col-12">
                        </div>
                        <div class="col-md col-12 form-group mb-3" data-for="name">
                            <textarea type="text" name="name" placeholder="Enter your text" data-form-field="Text"
                                class="form-control" id="text-input"></textarea>
                        </div>
                        <div class="mbr-section-btn col-12 col-md-auto">
                            <button type="submit" class="btn btn-secondary display-4">Analyse</button>
                        </div>
                    </div>
                </form>
            </div>
        </div>
    </div>
</section>
```

## License

No license information is provided in this repository.

