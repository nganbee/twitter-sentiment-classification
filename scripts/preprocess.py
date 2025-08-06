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
model_path = os.path.join(parent_dir, "model")
save_train_path = os.path.join(parent_dir, "data", "processed", "training.csv")
save_val_path = os.path.join(parent_dir, "data", "processed", "val.csv")

nlp = spacy.load("en_core_web_sm")
# name the index
df_train = pd.read_csv(data_train_path, header=None, names = ["id", "entity", "sentiment", "content"])
df_val = pd.read_csv(data_val_path, header=None, names = ["id", "entity", "sentiment", "content"])

def clean_text(text):
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and token.is_alpha and len(token) > 2
    ]
    
    return " ".join(tokens)

def label_sentiment(df):
    df['sentiment'] = df['sentiment'].apply(lambda x : "Neutral" if x == "Irrelevant" else x)
    
    return df

def preprocess_data(df):
    
    texts = df['content'].apply(
        lambda x : re.sub(r'[^a-zA-Z\s]', ' ', x.lower())
    ).tolist()
    
    df['text'] = [clean_text(doc) for doc in nlp.pipe(texts, batch_size=50, n_process=2)]
    
    df = label_sentiment(df) 
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['sentiment'])
    
    mask = (df['text'].isnull()) | (df['text'].str.strip() == '') | (df['text'].str.lower() == 'null')
    df = df[~mask]
    
    return df, le

df_train, le = preprocess_data(df_train)
df_val = preprocess_data(df_val)

# save processed data
df_train.to_csv(save_train_path, index=False)
df_val.to_csv(save_val_path, index= False)

# save label encoder
joblib.dump(le, os.path.join(model_path, "label_encoder.pkl"))





    


    
    
    
    
