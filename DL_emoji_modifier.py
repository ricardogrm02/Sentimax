import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Sentiment groupings
positive = {"joy", "happiness", "relief", "fun", "love", "surprise", "enthusiasm"}
neutral = {"neutral", "empty"}
negative = {"anger", "fear", "sadness", "shame", "disgust", "boredom", "hate", "worry", "disappointment"}

# Paths
model_path = "DL_emoji_model.h5"
tokenizer_path = "DL_emoji_tokenizer.pkl"
label_encoder_path = "DL_label_encoder.pkl"

# Hyperparameters
vocab_size = 10000
max_len = 100
embedding_dim = 128

# Load or train model
if os.path.exists(model_path) and os.path.exists(tokenizer_path) and os.path.exists(label_encoder_path):
    print("Loading existing DL emoji model and tokenizer...")
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
else:
    print("Training new DL emoji model...")
    # Prepare data
    df = pd.read_csv("train_emoji_data.csv")

    def map_to_polarity(label):
        if label in positive:
            return "positive"
        elif label in neutral:
            return "neutral"
        elif label in negative:
            return "negative"
        else:
            return "unknown"

    df["polarity"] = df["sentiment"].apply(map_to_polarity)
    df = df[df["polarity"] != "unknown"].reset_index(drop=True)

    texts = df["content"].astype(str).values
    labels = df["polarity"].values

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

    def build_bigru_model():
        input_layer = Input(shape=(max_len,))
        x = Embedding(vocab_size, embedding_dim)(input_layer)
        x = Bidirectional(GRU(64))(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation="relu")(x)
        output_layer = Dense(num_classes, activation="softmax")(x)
        return Model(inputs=input_layer, outputs=output_layer, name="BiGRU_Emoji")

    model = build_bigru_model()
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    # Save model and tokenizer
    model.save(model_path)
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

# Function to get polarity boost
def get_polarity_boost(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    preds = model.predict(padded, verbose=0)[0]
    return {label_encoder.classes_[i]: 1 + preds[i] for i in range(len(preds))}

# Standalone usage testing
if __name__ == "__main__":
    user_text = input("Enter emoji text to analyze polarity: ")
    boost_factors = get_polarity_boost(user_text)
    print("\nBoost Factors:")
    print(boost_factors)