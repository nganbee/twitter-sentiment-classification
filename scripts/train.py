import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

base_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(base_dir)
data_train_path = os.path.join(parent_dir, "data", "processed", "training.csv")
data_val_path = os.path.join(parent_dir, "data", "processed", "val.csv")
model_path = os.path.join(parent_dir, "model")

df_train = pd.read_csv(data_train_path)
df_val = pd.read_csv(data_val_path)

X_train = df_train['text']
y_train = df_train['label']
X_test = df_val['text']
y_test = df_val['label']

#TF-IDF
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_tf = tfidf.fit_transform(X_train)
X_test_tf = tfidf.transform(X_test)

le = joblib.load(os.path.join(model_path, "label_encoder.pkl"))

model = LinearSVC(C=10)
model.fit(X_train_tf, y_train)
y_pred = model.predict(X_test_tf)

y_test = le.inverse_transform(y_test)
y_pred = le.inverse_transform(y_pred)
print(classification_report(y_test, y_pred))

joblib.dump(model, os.path.join(model_path, "model_svc.pkl"))
