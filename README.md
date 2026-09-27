# Hate Speech Detection System

An end-to-end Machine Learning project to detect hate speech and offensive language in text, built with Python, Scikit-Learn, and Streamlit. This project is structured modularly for easy understanding and presentation.

## Project Structure
```text
hate_speech_project/
│
├── data/                    # Contains raw and sample dataset (e.g., sample.csv)
├── models/                  # Stores the trained .pkl model & vectorizer
├── src/                     # Core Machine Learning pipeline modules
│   ├── preprocess.py        # Cleans text, removes stopwords, applies stemming
│   ├── train.py             # Model training (Logistic Regression & Naive Bayes)
│   └── predict.py           # Loads model for inference
│
├── app.py                   # Streamlit web frontend
├── generate_sample.py       # Script to generate a dummy dataset for testing
└── requirements.txt         # Required libraries
```

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Sample Data (If you don't have a dataset yet):**
   ```bash
   python generate_sample.py
   ```
   *This creates `data/sample.csv` to let you test the pipeline immediately.*

3. **Train the Model:**
   ```bash
   python -m src.train
   ```
   *This reads `data/sample.csv`, trains Logistic Regression & Naive Bayes, prints accuracy metrics, and saves the best model in `models/`.*

4. **Run the Web Application:**
   ```bash
   streamlit run app.py
   ```

---

## 🎓 Viva Explanation & Common Questions

If you are presenting this for a college project, here's how to explain your work confidently:

### 1. What is the goal of this project?
**Answer:** The goal is to build an NLP (Natural Language Processing) model that automatically classifies text into categories like "Hate Speech", "Offensive Language", or "Normal". This is useful for moderating content on social media and forums.

### 2. Explain the Preprocessing steps (`src/preprocess.py`).
**Answer:** Raw text contains noise. I used Regex and NLTK to clean it:
- Removed URLs, HTML tags, and punctuation.
- Converted text to lowercase.
- Removed **stopwords** (common words like "and", "the" that add no meaning).
- Applied **Stemming** (reducing words to their base form, e.g., "running" -> "run") using PorterStemmer to reduce the vocabulary size and improve accuracy.

### 3. What is TF-IDF? Why did you use it instead of Bag of Words?
**Answer:** TF-IDF stands for Term Frequency-Inverse Document Frequency.
- **Term Frequency:** How often a word appears in a text.
- **Inverse Document Frequency:** Penalizes words that appear everywhere (like "is", "a").
I used TF-IDF over simple Bag of Words because it gives more weight to unique, meaningful words important for classification, rather than just counting word occurrences. I also used `ngram_range=(1,2)` to capture bigrams (two-word sequences) to understand context better.

### 4. Which Machine Learning models did you use and why?
**Answer:** I compared **Logistic Regression** and **Naive Bayes**.
- **Naive Bayes** is a probabilistic model based on Bayes' Theorem. It performs excellently as a baseline for text classification because it assumes feature independence.
- **Logistic Regression** works very well on sparse data (like TF-IDF arrays).
The `train.py` script evaluates both using `accuracy_score` and `classification_report` and saves the best performing one.

### 5. What are `model.pkl` and `vectorizer.pkl`?
**Answer:** After training, I serialize (save) the state of the model and the TF-IDF vectorizer using the `joblib` library. This is so we don't have to retrain the model every time we want to predict a new sentence. The Streamlit app (`app.py`) simply loads these files and makes instant inferences.
