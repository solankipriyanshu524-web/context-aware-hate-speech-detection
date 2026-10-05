# 🛡️ Multimodal Context-Aware Hate Speech Detection System
### 🔮 Phase 2: 3D Holographic Cyber Cortex & Multilingual BERT Reasoning (Live)

[![Phase 2](https://img.shields.io/badge/Phase%202-3D%20Hologram%20Live-brightgreen.svg)](#-phase-2-3d-holographic-cyber-cortex--multilingual-bert-hud)
[![3D Engine](https://img.shields.io/badge/3D%20Engine-Three.js%20WebGL-cyan.svg)](https://threejs.org/)
[![Languages](https://img.shields.io/badge/Languages-English%20%7C%20Hindi%20%7C%20Hinglish%20%7C%20Telugu-blue.svg)](#-how-the-system-understands-context)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![ASR](https://img.shields.io/badge/ASR-Faster--Whisper%20%2B%20WebSpeech-purple.svg)](https://github.com/SYSTRAN/faster-whisper)

> 🌌 **Phase 2 Upgrade:** Next-generation **3D Procedural Holographic Avatar (Three.js)** with real-time audio lip-sync, dynamic color-shifting (Safe Cyan / Slur Amber / Threat Red Alert), and deep contextual disambiguation across **English, Hindi, Hinglish, and Telugu**.

---

## ⚡ Quick Launch (Phase 2 3D Hologram HUD)

```bash
# 1. Start server from project directory
python -m http.server 8080

# 2. Open 3D Hologram HUD in your browser
http://localhost:8080/hologram_face_hud.html
```

---

## 📖 Overview

An advanced **Multimodal (Voice + Text)** Natural Language Processing and Machine Learning system engineered to detect hate speech, toxicity, and offensive language. 

Unlike traditional keyword-based filters that generate high false-positive rates on conversational slang, this engine features a **Hybrid Context-Disambiguation Layer** specifically tuned for nuanced multilingual text (English, Hindi, Hinglish, and Telugu), coupled with **OpenAI Faster-Whisper ASR** and **Three.js 3D Cyber Avatar** for real-time speech analysis and visualization.

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
- **Speech Recognition (ASR):** Faster-Whisper (OpenAI Whisper via CTranslate2), Web Speech API
- **Application & Dashboard:** Streamlit, Three.js (WebGL 3D Particles), Vanilla HTML5/CSS3 Cyberpunk HUD
- **Execution & GPU:** NVIDIA CUDA, cuDNN

---

## 🔮 Phase 2: 3D Holographic Cyber Cortex & Multilingual BERT HUD

Phase 2 elevates this project from a standard ML dashboard into a state-of-the-art **Sci-Fi 3D Holographic AI Cockpit** with multilingual BERT reasoning:

```
              ┌────────────────────────────────────────────────────────┐
              │     🌌 3D PROCEDURAL HOLOGRAPHIC CYBER CORTEX (Three.js)│
              │   • 3D Movie-Grade Particle Mesh (1,800+ nodes)         │
              │   • Audio-Reactive Facial Geometry & Speech Lip-Sync    │
              │   • Dynamic Color Shifting (Cyan / Amber / Red Alert)   │
              └───────────────────────────┬────────────────────────────┘
                                          │
    ┌─────────────────────────────────────┴─────────────────────────────────────┐
    ▼                                                                           ▼
[ 🎙️ Live Multilingual Voice STT ]                           [ 🧠 Multilingual BERT Context Engine ]
  • Dynamic Language Switcher (EN / HI / TE)                    • 18 Pre-Calibrated Disambiguation Benchmarks
  • Devanagari Phonetic English Threat Catching                • English: "Killing it" (Safe) vs "Kill you" (Threat)
  • Multilingual Web Speech Synthesis Output                    • Hindi: "Mandir ka ghanta" (Safe) vs Slang (Offensive)
                                                                • Telugu: "Intlo kukka" (Safe) vs Insult (Offensive)
```

### 🌟 Key Phase 2 Features

1. **3D Procedural Hologram Particle Avatar:**
   - 100% procedural 3D face mesh built with Three.js (No static 2D cutouts).
   - Real-time jaw displacement, eyelid blinking, and orbital particle swarm synchronized to audio speech playback.

2. **Multilingual Context Disambiguation Matrix (English, Hindi, Hinglish, Telugu):**
   - **English Nuances:** Differentiates praise metaphors (*"You are killing it on stage tonight"*) from literal threats (*"I will kill you"*).
   - **Hindi/Hinglish Nuances:** Resolves dual-use words like *Ghanta, Kutta, Saala, Gadha, Ullu* based on semantic context anchors.
   - **Telugu Nuances:** Classifies *Kukka* (కుక్క - pet vs slur), *Pichi* (పిచ్చి - hobby craze vs mental insult), *Donga* (దొంగ - affectionate child teasing vs thief).
   - **Devanagari Phonetic Speech Defense:** Detects English death threats spoken into Hindi microphones (e.g. `आई वांट टू किल यू` ➔ 98% Red Alert Threat).

3. **To Launch Phase 2 3D HUD:**
   ```bash
   # From project root
   python -m http.server 8080
   # Open in browser: http://localhost:8080/hologram_face_hud.html
   ```
