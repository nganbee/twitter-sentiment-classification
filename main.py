from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
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
templates = Jinja2Templates(directory="templates")

class InputData(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None
    
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict_api(data: InputData):
    if data.text is not None:
        items = [data.text]
    elif data.texts is not None:
        items = data.texts
    else:
        raise HTTPException(status_code=422, detail="Provide 'text' or 'texts'.")
    labels = inf.predict(items)
    return {"predictions": [str(x) for x in labels]}