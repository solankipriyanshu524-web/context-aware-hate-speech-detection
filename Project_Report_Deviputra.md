# Project Report: DEVIPUTRA - The Vagabond Hate Speech Detection System

**Author/Developer:** Priyanshu Solanki
**Project Focus:** End-to-End Context-Aware NLP Text Classification

---

## 1. Abstract
The "DEVIPUTRA" Hate Speech Detection System is a comprehensive Machine Learning and Natural Language Processing (NLP) application designed to automatically classify text into categories such as "Hate Speech", "Offensive Language", or "Normal". Built to moderate content on social media and forums, the system employs advanced text preprocessing, TF-IDF vectorization, and a Logistic Regression model to detect toxicity. Furthermore, it integrates a unique rule-based Context-Aware Intelligence layer specifically for Hindi/Hinglish vocabulary to correctly interpret dual-meaning words (e.g., distinguishing between "ghanta" as a temple bell vs. an offensive slang).

## 2. System Architecture & Methodology

### 2.1 Natural Language Processing (NLP) Pipeline
The project utilizes a structured pipeline to transform raw, noisy text into mathematical vectors that the machine learning model can understand.

1. **Purification (Preprocessing):** 
   Using the Regular Expressions (`re`) and Natural Language Toolkit (`nltk`) libraries, raw text is stripped of URLs, HTML formatting, and punctuation. The text is converted to lowercase, and "stopwords" (common words like 'and', 'the') are removed. This reduces noise so the model focuses purely on meaningful semantic data.
2. **Vectorization (TF-IDF):** 
   The cleaned text is processed using Term Frequency-Inverse Document Frequency (TF-IDF). This method scales the importance of words, heavily weighting rare, impactful words (like 'hate' or 'kill') while downplaying common ones. The system uses an `ngram_range=(1,2)` to capture bigrams, aiding in contextual understanding.

### 2.2 Machine Learning Core
The primary intelligence of the system relies on a **Logistic Regression** algorithm trained on thousands of samples. 
- Logistic Regression is highly effective on sparse text data (such as TF-IDF matrices), calculating the probability that an input belongs to specific toxicity classes based on learned feature weights.
- A **Multinomial Naive Bayes** model was also evaluated as a baseline for its universal excellence in text classification.
- The state of the best-performing model and the TF-IDF vectorizer are serialized via `joblib` into `model.pkl` and `vectorizer.pkl` for rapid, real-time inference without retraining.

### 2.3 Context-Aware Intelligence (Hybrid Approach)
One of the most advanced features of DEVIPUTRA is its ability to understand Hindi/Hinglish dual-meaning words, preventing false positives.
- A purely ML-based engine might blindly flag the word *kutta*.
- DEVIPUTRA overrides this by checking for safe-context keywords. If the text discusses an animal/pet (e.g., "mera pyaara kutta"), the intelligence layer classifies it as Normal. Conversely, if it is targeted as a slur, the ML prediction of Hate Speech holds.
- Other contextual edge cases handled include: *ghanta*, *saala*, *gadha*, *ullu*, and *bandar*.

---

## 3. Web Application & Features (UI/UX)
The frontend is powered by **Streamlit**, styled with a premium cinematic dark aesthetic (Glassmorphism, custom typography like 'Cinzel', animated Musashi graphics). The application boasts a multi-tab professional interface:

1. **The Scroll of Judgment (Live Analyzer):** 
   A real-time inference engine. When text is submitted, it is passed through the model. If toxicity is detected, the hateful segments are censored with a dictionary fallback system (e.g., obscuring bypassed profanities), and a "purified" safe version of the text is presented.
2. **Priyanshu Chatbot:** 
   An interactive conversational bot that explains the project's technical methodologies (NLP, Logistic Regression, TF-IDF, Preprocessing) to users, offering a unique avenue for project demonstration.
3. **Batch Purification (CSV Upload):** 
   Users can upload large `.csv` datasets containing massive arrays of text. The system concurrently processes every row, applies context-aware censorship, and generates a downloadable report with the final predictions.
4. **Project Insights Dashboard:** 
   A polished analytics screen showcasing system metrics: a vocabulary size of over 18,000 words, model accuracy of ~92.4%, and rapid inference times (~12ms), alongside graphical feature weight distributions.

---

## 4. Execution Guide

**Environment Setup:**
Ensure Python 3.8+ is installed.
```bash
pip install -r requirements.txt
```

**Running the Model Training (Optional):**
To parse `sample.csv` and retrain the models:
```bash
python -m src.train
```

**Launching the Application:**
To start the DEVIPUTRA interface:
```bash
streamlit run app.py
```

---

## 5. Viva / Presentation Q&A Reference

### Q: What is the primary difference between TF-IDF and simple Bag of Words?
**A:** Bag of Words simply counts how many times a word appears. TF-IDF considers Term Frequency but also applies an Inverse Document Frequency penalty to common words across the entire dataset. This ensures that words that genuinely define a class (like "hate") get higher mathematical weights than filler words.

### Q: Why use Logistic Regression over Deep Learning?
**A:** For textual data represented in highly sparse matrices (like TF-IDF arrays with ~18,000 dimensions), Logistic Regression achieves remarkable accuracy (92.4%) instantly, with ~12ms inference times, while requiring significantly fewer computational resources compared to a BERT or LSTM architecture.

### Q: How does your system deal with creative spelling of offensive words (e.g., "m@darchod")?
**A:** The `censor_text` function in `app.py` acts as a shield against obfuscation. It includes a `deobfuscate` mechanism that normalizes standard l33t-speak replacements (like @ to a, $ to s) and collapses drawn-out repeating letters before scanning against the strict profanity dictionary, ensuring accurate censoring alongside the ML prediction.

### Q: How do you handle Hindi words that can be both innocent or offensive depending on how they are used?
**A:** The project features "Context-Aware Intelligence." The system scans for specific, safe contextual indicators for words like "ghanta" or "kutta". If the context layer observes that a word is being used in its literal/safe definition (like a temple bell or a pet dog), it overrides the model and flags the text as safe. If no safe context is found, the system defaults to the ML layer's prediction.
