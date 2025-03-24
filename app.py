from flask import Flask, request, jsonify, send_from_directory
import os
import pickle
import re
import nltk
import pandas as pd
from flask_cors import CORS
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nltk.stem import WordNetLemmatizer

# Download stopwords and WordNet
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

app = Flask(__name__, static_folder="../frontend", static_url_path="/")
CORS(app)  # Enable CORS for frontend access

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
FAKE_PATH = os.path.join(BASE_DIR, "Fake.csv")
REAL_PATH = os.path.join(BASE_DIR, "True.csv")

# Ensure dataset files exist
if not os.path.exists(FAKE_PATH) or not os.path.exists(REAL_PATH):
    raise FileNotFoundError("Dataset files not found! Place 'Fake.csv' and 'True.csv' in the backend folder.")

# Load dataset
df_fake = pd.read_csv(FAKE_PATH)
df_real = pd.read_csv(REAL_PATH)

df_fake["label"] = 0  # Fake News
df_real["label"] = 1  # Real News

# Balance the dataset
min_size = min(len(df_fake), len(df_real))
df_fake = df_fake.sample(min_size, random_state=42)
df_real = df_real.sample(min_size, random_state=42)

# Add fake news patterns for better detection
fake_patterns = [
    "shocking discovery", "scientists are stunned", "government cover-up", 
    "cure for all diseases", "you won't believe", "miracle treatment", 
    "secret exposed", "banned news", "world leaders in panic", 
    "unbelievable breakthrough"
]
df_fake["text"] = df_fake["title"] + " " + df_fake["text"] + " " + " ".join(fake_patterns)

# Combine datasets
df = pd.concat([df_fake, df_real]).sample(frac=1).reset_index(drop=True)

# Preprocessing function
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Keep only letters
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]  # Lemmatize and remove stopwords
    return " ".join(text)

df["text"] = df["title"] + " " + df["text"]
df["text"] = df["text"].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)

# Feature extraction
if os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH):
    print("Loading pre-trained vectorizer and model...")
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    print("Training model...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=8000)  # Consistent feature extraction
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = XGBClassifier(n_estimators=200, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train_vec, y_train)
    
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

# Evaluate model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

@app.route("/")
def home():
    """Serve the frontend (index.html)."""
    return send_from_directory("../frontend", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Predict whether the input news is Fake or Real."""
    try:
        data = request.json
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400

        text = clean_text(data["text"])

        # Load trained vectorizer and model
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]
        result = "Fake News" if prediction == 0 else "Real News"
        
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/<path:filename>")
def serve_static_files(filename):
    """Serve static files like CSS and JS."""
    allowed_extensions = [".html", ".css", ".js", ".png", ".jpg", ".jpeg", ".svg"]
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        return jsonify({"error": "File type not allowed"}), 403
    return send_from_directory("../frontend", filename)

if __name__ == "__main__":
    app.run(debug=False)  # Disable debug mode for production
