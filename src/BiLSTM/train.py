import pandas as pd
import os
from tensorflow import keras
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
import pickle

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    parent_dir = os.path.dirname(base_dir)
    parent_dir = os.path.dirname(parent_dir)
    data_train_path = os.path.join(parent_dir, "data", "processed", "training.csv")
    data_val_path = os.path.join(parent_dir, "data", "processed", "val.csv")
    model_path = os.path.join(parent_dir, "models")

    df_train = pd.read_csv(data_train_path)
    df_val = pd.read_csv(data_val_path)

    X_train = df_train['text']
    y_train = df_train['label']
    X_val = df_val['text']
    y_val = df_val['label']

    vocab_size = 10000
    embedding_dim = 64
    max_length = 50
    num_classes = 3

    #Tokenize and vectorize
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq   = tokenizer.texts_to_sequences(X_val)

    X_train_seq = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
    X_val_seq   = pad_sequences(X_val_seq, maxlen=max_length, padding='post')

    model = keras.Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
        Dropout(0.5), # Adding dropout to avoid overfitting
        Dense(3, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy', # using labels = [0, 1, 2]
        optimizer='adam', 
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(
        X_train_seq, y_train, 
        epochs=10, 
        batch_size=32, 
        validation_data=(X_val_seq, y_val),
        callbacks=[early_stop]
    )

    lstm_model_path = os.path.join(model_path, "model_bilstm.h5")
    model.save(lstm_model_path)
    
    tokenizer_path = os.path.join(model_path, "tokenizer.pkl")
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == "__main__":
    main()