import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

from preprocess import clean_text

# Load dataset
base_dir = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(base_dir, "data", "spam.csv")

df = pd.read_csv(data_path, encoding='latin-1')

# Clean dataset
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Preprocess text
df['cleaned'] = df['message'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['label'], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ================================
# 🔥 TRAIN MULTIPLE MODELS
# ================================

# SVM (your main model)
svm_model = SVC(kernel='linear', class_weight='balanced')
svm_model.fit(X_train_tfidf, y_train)

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# ================================
# 📊 MODEL COMPARISON
# ================================

print("\n📊 Model Comparison:")
print("Naive Bayes Accuracy:", nb_model.score(X_test_tfidf, y_test))
print("Logistic Regression Accuracy:", lr_model.score(X_test_tfidf, y_test))
print("SVM Accuracy:", svm_model.score(X_test_tfidf, y_test))

# ================================
# 🎯 FINAL MODEL (choose one)
# ================================

# 👉 Keep SVM as final (as you requested)
model = svm_model

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("\nBest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix (Final Model)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ================================
# 💾 SAVE MODEL
# ================================

model_dir = os.path.join(base_dir, "model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "model.pkl")
vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(vectorizer_path, "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model saved at:", model_path)
print("\n✅ Model saved successfully!")