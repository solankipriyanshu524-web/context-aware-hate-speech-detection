# Context-Aware Hate Speech & Toxicity Detection System

An end-to-end Machine Learning and Natural Language Processing (NLP) system designed to detect hate speech and offensive content with context-aware disambiguation for dual-meaning words (Hinglish/English), featuring an interactive Streamlit dashboard.

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

## 🌟 Key Features

- **Context-Aware Intelligence:** Intelligently handles dual-meaning words (e.g., Hinglish/Hindi words that can be harmless pets/objects or offensive slurs depending on sentence context) to minimize false positives.
- **Robust NLP Pipeline:** Regex cleaning, stopword filtration, and Porter Stemming via NLTK.
- **TF-IDF Feature Representation:** Utilizes unigram and bigram representation (`ngram_range=(1,2)`) to preserve contextual nuance.
- **Machine Learning Core:** Evaluates Logistic Regression and Naive Bayes for high precision and ultra-fast inference (~12ms).
- **Interactive Streamlit Web Dashboard:** Includes live real-time analysis, automated text purification (censorship), and batch CSV processing.

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Scikit-Learn, NLTK, Pandas, NumPy, Joblib
- **Frontend / UI:** Streamlit

