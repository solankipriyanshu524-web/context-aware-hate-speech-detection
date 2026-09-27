import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

from src.preprocess import clean_text

def train_and_evaluate(data_path, text_col='text', label_col='label', models_dir='models'):
    """
    Trains models for Hate Speech Detection and saves the best one.
    Assumes data_path points to a CSV with a text column and a label column.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Preprocessing
    print("Cleaning text data... this may take a moment.")
    df['clean_text'] = df[text_col].apply(clean_text)
    
    X = df['clean_text']
    y = df[label_col]
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature extraction using TF-IDF
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_vec, y_train)
    lr_preds = lr_model.predict(X_test_vec)
    lr_acc = accuracy_score(y_test, lr_preds)
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    print(classification_report(y_test, lr_preds))
    
    # Train Naive Bayes
    print("Training Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)
    nb_preds = nb_model.predict(X_test_vec)
    nb_acc = accuracy_score(y_test, nb_preds)
    print(f"Naive Bayes Accuracy: {nb_acc:.4f}")
    print(classification_report(y_test, nb_preds))
    
    # Save the best model
    os.makedirs(models_dir, exist_ok=True)
    
    # In this case, we prefer Logistic Regression for saving as it usually performs well and is robust
    best_model = lr_model if lr_acc >= nb_acc else nb_model
    model_name = 'logistic_regression.pkl' if lr_acc >= nb_acc else 'naive_bayes.pkl'
    
    joblib.dump(best_model, os.path.join(models_dir, 'model.pkl'))
    joblib.dump(vectorizer, os.path.join(models_dir, 'vectorizer.pkl'))
    
    print(f"Best model ({model_name}) and vectorizer saved to '{models_dir}/'.")

if __name__ == "__main__":
    # Example usage (will need dummy data to run directly)
    default_data_path = os.path.join('data', 'sample.csv')
    if os.path.exists(default_data_path):
        train_and_evaluate(default_data_path)
    else:
        print(f"Data not found at {default_data_path}. Please create a sample dataset first.")
