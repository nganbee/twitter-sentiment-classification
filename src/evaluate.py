from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test, label_encoder=None):
    y_pred = model.predict(X_test)
    
    if label_encoder:
        y_test = label_encoder.inverse_transform(y_test)
        y_pred = label_encoder.inverse_transform(y_pred)
        
    print(classification_report(y_test, y_pred))
