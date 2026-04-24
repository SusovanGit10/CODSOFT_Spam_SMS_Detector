import pickle
from preprocess import clean_text

# Load model
import os
import pickle
from preprocess import clean_text

# Get base directory (project root)
base_dir = os.path.dirname(os.path.dirname(__file__))

# Build correct paths
model_path = os.path.join(base_dir, "model", "model.pkl")
vectorizer_path = os.path.join(base_dir, "model", "vectorizer.pkl")

# Load files
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)


def predict_sms(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    result = model.predict(vector)[0]
    return "Spam" if result == 1 else "Ham"


if __name__ == "__main__":
    msg = input("Enter message: ")
    print("Prediction:", predict_sms(msg))

def predict_sms(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    result = model.predict(vector)[0]
    return "Spam" if result == 1 else "Ham"

# Test
if __name__ == "__main__":
    msg = input("Enter message: ")
    print("Prediction:", predict_sms(msg))