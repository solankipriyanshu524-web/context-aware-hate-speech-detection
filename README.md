# 🛡️ Multimodal Context-Aware Hate Speech & Toxicity Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![ASR](https://img.shields.io/badge/ASR-Faster--Whisper%20(OpenAI)-purple.svg)](https://github.com/SYSTRAN/faster-whisper)
[![UI](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)

An advanced **Multimodal (Voice + Text)** Natural Language Processing and Machine Learning system engineered to detect hate speech, toxicity, and offensive language. 

Unlike traditional keyword-based filters that generate high false-positive rates on conversational slang, this engine features a **Hybrid Context-Disambiguation Layer** specifically tuned for nuanced bilingual text (English, Hindi, and Hinglish), coupled with **OpenAI Faster-Whisper ASR** for real-time speech analysis.

---

## 🚀 Key Highlights & Architecture

```
                                  [ 🎙️ Audio Input (Mic/File) ]
                                                │
                                                ▼
                                  [ ⚡ OpenAI Faster-Whisper ASR ]
                                       (GPU/CUDA Accelerated)
                                                │
                                                ▼
  [ 📝 Text Input ] ──────────────► [ NLP Preprocessing & Cleansing ]
                                    (Regex, Stopwords, Stemming)
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
    [ 🧠 Context-Aware Disambiguation ]                   [ 📐 TF-IDF Vectorizer (1,2 N-grams) ]
    (Semantic Anchor & Co-occurrence Check)                                │
                     │                                                     ▼
                     │                                   [ 🤖 ML Classifier (Logistic Reg / NB) ]
                     │                                                     │
                     └──────────────────────────┬──────────────────────────┘
                                                ▼
                                    [ 🎯 Final Classification ]
                                    (Normal / Offensive / Hate Speech)
                                                │
                                                ▼
                             [ 🛡️ Real-Time Purification & Dashboard ]
```

---

## 🧠 How the System Understands "Context"

Standard Machine Learning classifiers often blindly penalize words that have dual meanings. Our engine solves this using **Co-Occurrence Semantic Anchor Verification**:

| Vocabulary | Safe Contextual Anchor (Classified as `Normal`) | Slur / Offensive Context (Classified as `Toxic`) |
| :--- | :--- | :--- |
| **`Ghanta` / `घंटा`** | Co-occurs with *mandir, pooja, aarti, bajna, time, hour, samay*. (e.g., *"Mandir me ghanta baj raha hai"*). | Used dismissively with zero religious/time anchors (e.g., *"Tujhe ghanta farak nahi padta"*). |
| **`Kutta` / `कुत्ता`** | Co-occurs with *pet, mera, pyaara, dog, doctor, street, khana*. (e.g., *"Mera kutta bohot cute hai"*). | Used as targeted abuse without animal/pet semantics (e.g., *"Tu kutta hai chup kar"*). |
| **`Saala` / `साala`** | Co-occurs with kinship terms: *behen, shaadi, family, rishtedaar*. (e.g., *"Mere saale ki shaadi hai"*). | Used as standalone abusive insult. |
| **`Gadha` / `गधा`** | Co-occurs with farm/animal context: *khet, animal, gaon, sawari*. | Targeted human cognitive insult. |

---

## 🎙️ Multimodal Voice Detection (Faster-Whisper)

- **Speech-to-Text Pipeline:** Uses `faster-whisper` (CTranslate2 implementation of OpenAI Whisper) for high-speed speech transcription.
- **Hardware Acceleration:** Native NVIDIA CUDA/cuDNN GPU execution with automatic fallback to optimized `int8` CPU inference.
- **Live Inference:** Transcribes microphone voice notes and uploaded `.wav` / `.mp3` files in real-time, then feeds the transcription directly into the context-aware toxicity evaluator.

---

## 📂 Project Structure

```text
hate_speech_project/
│
├── data/
│   └── sample.csv                 # Labeled training & benchmarking dataset
├── models/
│   ├── model.pkl                  # Serialized best-performing ML model
│   └── vectorizer.pkl             # Fitted TF-IDF N-gram feature extractor
├── src/
│   ├── preprocess.py              # Text cleansing, regex filtering, stopword removal
│   ├── train.py                   # Model training, hyperparameter tuning & evaluation
│   ├── predict.py                 # Hybrid Context-Aware prediction & decision engine
│   └── voice_detector.py          # OpenAI Whisper audio capture & GPU-accelerated ASR
│
├── app.py                         # Streamlit full-stack interactive web application
├── generate_sample.py             # Synthetic balanced data generator
└── requirements.txt               # Project dependencies
```

---

## 📊 Model Evaluation & Benchmarks

The system benchmarks multiple algorithms using Stratified 80/20 train-test splits on balanced contextual datasets:

| Model Architecture | Accuracy | Feature Representation | Inference Latency |
| :--- | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | ~89.1% | TF-IDF (Unigram + Bigram) | ~5ms |
| **Logistic Regression (Best)** | **~92.4%** | TF-IDF (5000 max features, N-gram 1-2) | **~12ms** |

---

## ⚡ Quick Start

### 1. Installation
Clone the repository and set up your Python environment:
```bash
git clone https://github.com/YOUR_USERNAME/context-based-hate-speech-detector.git
cd context-based-hate-speech-detector
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python -m src.train
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```

---

## 🛠️ Technology Stack

- **Machine Learning & NLP:** Scikit-Learn, NLTK, TF-IDF, Joblib
- **Speech Recognition (ASR):** Faster-Whisper (OpenAI Whisper via CTranslate2), Librosa, SoundFile
- **Application & Dashboard:** Streamlit, Pandas, NumPy
- **Execution & GPU:** NVIDIA CUDA, cuDNN

