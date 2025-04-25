import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Bidirectional, GRU, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and prepare the data
df = pd.read_csv("text_train_data.csv")
texts = df["content"].astype(str).values
labels = df["sentiment"].values

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Tokenization
vocab_size = 10000
max_len = 100
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

# ============================
# BiGRU Model
# ============================
def build_bigru_model():
    input_layer = Input(shape=(max_len,))
    x = Embedding(vocab_size, 128)(input_layer)
    x = Bidirectional(GRU(64))(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output, name="BiGRU")

# ============================
# CNN Model
# ============================
def build_cnn_model():
    input_layer = Input(shape=(max_len,))
    x = Embedding(vocab_size, 128)(input_layer)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output, name="CNN")

# ============================
# Create and compile models
# ============================
bigru_model = build_bigru_model()
cnn_model = build_cnn_model()

models = [bigru_model, cnn_model]

for model in models:
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    print(f"Training {model.name}...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=32)

# ============================
# Ensemble Prediction (average probabilities)
# ============================
def ensemble_predict(models, X):
    preds = [model.predict(X, verbose=0) for model in models]
    avg_preds = np.mean(preds, axis=0)
    return np.argmax(avg_preds, axis=1)

# Evaluate
y_pred = ensemble_predict(models, X_test)
accuracy = np.mean(y_pred == y_test)
print(f"\nEnsemble Accuracy: {accuracy:.4f}")

# Save models and tokenizer
for model in models:
    model.save(f"{model.name.lower()}_model.h5")
import pickle
with open("DL_text_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("DL_text_ensemble_model.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
