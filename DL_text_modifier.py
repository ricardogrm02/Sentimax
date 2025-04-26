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

# Prepare data
df = pd.read_csv("text_train_data.csv")

def map_to_polarity(label):
    if label in positive:
        return "positive"
    elif label in neutral:
        return "neutral"
    elif label in negative:
        return "negative"
    else:
        return "unknown"

# Map original labels to positive/neutral/negative
df["polarity"] = df["sentiment"].apply(map_to_polarity)
df = df[df["polarity"] != "unknown"].reset_index(drop=True)

texts = df["content"].astype(str).values
labels = df["polarity"].values

# Tokenizer setup
vocab_size = 10000
max_len = 100
embedding_dim = 128

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# Label encoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

# Build BiGRU model
def build_bigru_model():
    input_layer = Input(shape=(max_len,))
    x = Embedding(vocab_size, embedding_dim)(input_layer)
    x = Bidirectional(GRU(64))(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    output_layer = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=input_layer, outputs=output_layer, name="BiGRU")

model = build_bigru_model()
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train BiGRU
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Save trained model and tokenizer
model = tf.keras.models.load_model("bigru_sentiment_polarity_model.h5")
with open("sentiment_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("sentiment_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Function to get polarity boost
def get_polarity_boost(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
    preds = model.predict(padded)[0]
    return {label_encoder.classes_[i]: 1 + preds[i] for i in range(len(preds))}

# Optional standalone usage
if __name__ == "__main__":
    user_text = input("Enter text to analyze sentiment polarity: ")
    boost_factors = get_polarity_boost(user_text)
    print("\nBoost Factors:")
    print(boost_factors)
