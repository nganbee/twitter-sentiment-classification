from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
from src import inference as inf
import os
import joblib, spacy

base_dir = os.path.dirname(os.path.abspath(__file__)) 
model_path = os.path.join(base_dir, "model")

# Load label, model, vectorizer
le = joblib.load(os.path.join(model_path, "label_encoder.pkl"))
model = joblib.load(os.path.join(model_path, "model_svc.pkl"))
vectorizer = joblib.load(os.path.join(model_path, "vectorizer.pkl"))
nlp = spacy.load("en_core_web_sm")

app = FastAPI()

class Items(BaseModel):
    texts: Union[List[str], str]

@app.post("/predict")
def predict(items: Items):
    texts = items.texts if isinstance(items.texts, list) else [items.texts]
    X_clean = inf.preprocess_texts(texts, nlp)
    X_vec   = vectorizer.transform(X_clean)
    y_pred  = model.predict(X_vec)
    labels  = le.inverse_transform(y_pred)
    return {"predictions": labels.tolist()}