from src import preprocess as p
import joblib, spacy
import os

base_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(base_dir)
model_path = os.path.join(parent_dir, "model")

# Load label, model, vectorizer
le = joblib.load(os.path.join(model_path, "label_encoder.pkl"))
model = joblib.load(os.path.join(model_path, "model_svc.pkl"))
vectorizer = joblib.load(os.path.join(model_path, "vectorizer.pkl"))
nlp = spacy.load("en_core_web_sm")

def preprocess_texts(texts, nlp):
    if isinstance(texts, str):
        texts = [texts]
    texts = [p.normalize_text(t) for t in texts]
    out = []
    for t in texts:
        doc = nlp(t)
        out.append(p.clean_text(doc))
    return out

def predict(texts):
    X_clean = preprocess_texts(texts, nlp)
    X_vec = vectorizer.transform(X_clean)
    y_pred = model.predict(X_vec)
    
    return le.inverse_transform(y_pred), X_clean
    