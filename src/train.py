import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.svm import LinearSVC
from evaluate import evaluate_model

base_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(base_dir)
data_train_path = os.path.join(parent_dir, "data", "processed", "training.csv")
data_val_path = os.path.join(parent_dir, "data", "processed", "val.csv")
model_path = os.path.join(parent_dir, "models")

#Load data
df_train = pd.read_csv(data_train_path)
df_val = pd.read_csv(data_val_path)

#Set the train and the test
X_train = df_train['text']
y_train = df_train['label']
X_test = df_val['text']
y_test = df_val['label']

#TF-IDF
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_tf = tfidf.fit_transform(X_train)
X_test_tf = tfidf.transform(X_test)

joblib.dump(tfidf, os.path.join(model_path, "vectorizer.pkl"))

#Load label encoder
le = joblib.load(os.path.join(model_path, "label_encoder.pkl"))

#Train model
model = LinearSVC(C=4)
model.fit(X_train_tf, y_train)

#Evaluate model
evaluate_model(model, X_test_tf, y_test, le)

joblib.dump(model, os.path.join(model_path, "model_svc.pkl"))
