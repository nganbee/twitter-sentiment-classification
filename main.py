from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src import inference as inf
import os
import joblib, spacy
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

base_dir = os.path.dirname(os.path.abspath(__file__)) 
model_path = os.path.join(base_dir, "models")

# Load label, model, vectorizer
le = joblib.load(os.path.join(model_path, "label_encoder.pkl"))
model = joblib.load(os.path.join(model_path, "model_svc.pkl"))
vectorizer = joblib.load(os.path.join(model_path, "vectorizer.pkl"))

lstm_model = load_model(os.path.join(model_path, "model_bilstm.keras"))
tokenizer = joblib.load(os.path.join(model_path, "tokenizer.pkl"))

nlp = spacy.load("en_core_web_sm")

templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class InputData(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None
    
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )

@app.post("/predict")
def predict_api(data: InputData):
    if not data.text:
        raise HTTPException(status_code=422, detail="Provide 'text'.")
    
    svc_label = inf.predict([data.text])[0] 

    seq = tokenizer.texts_to_sequences([data.text])
    padded = pad_sequences(seq, maxlen=50, padding='post')
    pred_probs = lstm_model.predict(padded)
    lstm_idx = pred_probs.argmax(axis=1)[0]
    lstm_label = le.inverse_transform([lstm_idx])[0]

    return {
        "svc": str(svc_label),
        "bilstm": str(lstm_label)
    }