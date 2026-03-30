import pandas as pd
import re
import spacy
import os
from sklearn.preprocessing import LabelEncoder
import joblib

base_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(base_dir)
data_train_path = os.path.join(parent_dir, "data", "raw", "twitter_training.csv")
data_val_path = os.path.join(parent_dir, "data", "raw", "twitter_validation.csv")
model_path = os.path.join(parent_dir, "models")
save_train_path = os.path.join(parent_dir, "data", "processed", "training.csv")
save_val_path = os.path.join(parent_dir, "data", "processed", "val.csv")

def normalize_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    else:
        return ''
        

def clean_text(doc):
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and token.is_alpha and len(token) > 2
    ]
    
    return " ".join(tokens)

def label_sentiment(df):
    df['sentiment'] = df['sentiment'].astype(str).str.strip().str.lower()
    df['sentiment'] = df['sentiment'].replace({'irrelevant':'neutral'})

    return df

def preprocess_data(df, le = None):
    
    df = df.dropna(subset = ['content', 'sentiment']).copy()
    
    texts = df['content'].apply(normalize_text).tolist()
    df['text'] = [clean_text(doc) for doc in nlp.pipe(texts, batch_size=50, n_process=2)]
    
    df = label_sentiment(df) 
    
    if le is None:
        le = LabelEncoder()
        le.fit(df['sentiment'])
    df['label'] = le.transform(df['sentiment'])
    
    mask = (df['text'].isnull()) | (df['text'].str.strip() == '') | (df['text'].str.lower() == 'null')
    df = df[~mask]
    
    X = df['text']
    y = df['label']
    
    return X, y, le

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    # name the index
    df_train = pd.read_csv(data_train_path, header=None, names = ["id", "entity", "sentiment", "content"])
    df_val = pd.read_csv(data_val_path, header=None, names = ["id", "entity", "sentiment", "content"])

    print("Train")
    X_train, y_train, le = preprocess_data(df_train)
    print("Val")
    X_val, y_val, le = preprocess_data(df_val, le)

    # save processed data
    pd.DataFrame({'text': X_train, 'label': y_train}).to_csv(save_train_path, index=False)
    pd.DataFrame({'text': X_val, 'label': y_val}).to_csv(save_val_path, index=False)

    # save label encoder
    joblib.dump(le, os.path.join(model_path, "label_encoder.pkl"))