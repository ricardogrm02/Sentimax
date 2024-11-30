import joblib
import emoji
import numpy as np

# Load trained models and vectorizers
text_model = joblib.load("sentimax_ensemble_model.pkl")
text_vectorizer = joblib.load("tfidf_vectorizer.pkl")

emoji_model = joblib.load("emoji_sentimax_ensemble_model.pkl")
emoji_vectorizer = joblib.load("emoji_tfidf_vectorizer.pkl")

label_encoder = joblib.load("label_encoder.pkl")  # Shared across both models

# Preprocess input to separate text and emojis
def preprocess_input(input_text):
    # Separate text and emojis
    text_only = "".join([char for char in input_text if not char in emoji.EMOJI_DATA])
    emojis_only = "".join([char for char in input_text if char in emoji.EMOJI_DATA])
    return text_only.strip(), emojis_only.strip()

# Predict sentiment for text
def predict_text_sentiment(text, text_vectorizer, text_model):
    if text:
        vectorized_text = text_vectorizer.transform([text])
        text_proba = text_model.predict_proba(vectorized_text)
        return text_proba
    return None  # No text to process

# Predict sentiment for emojis
def predict_emoji_sentiment(emojis, emoji_vectorizer, emoji_model):
    if emojis:
        vectorized_emojis = emoji_vectorizer.transform([emojis])
        emoji_proba = emoji_model.predict_proba(vectorized_emojis)
        return emoji_proba
    return None  # No emojis to process

# Combine predictions
def combine_predictions(text_proba, emoji_proba, text_weight=0.7, emoji_weight=0.3):
    if text_proba is not None and emoji_proba is not None:
        combined_proba = (text_weight * text_proba) + (emoji_weight * emoji_proba)
    elif text_proba is not None:
        combined_proba = text_proba
    elif emoji_proba is not None:
        combined_proba = emoji_proba
    else:
        return None  # No valid input
    return combined_proba

# Decode and display the sentiment
def decode_sentiment(combined_proba, label_encoder):
    if combined_proba is not None:
        top_index = combined_proba.argmax()
        top_sentiment = label_encoder.inverse_transform([top_index])[0]
        return top_sentiment, combined_proba
    return "No sentiment detected", None

# Full sentiment analysis pipeline
def analyze_sentiment(input_text, text_vectorizer, text_model, emoji_vectorizer, emoji_model, label_encoder):
    # Preprocess input
    text_only, emojis_only = preprocess_input(input_text)
    
    # Predict sentiments
    text_proba = predict_text_sentiment(text_only, text_vectorizer, text_model)
    emoji_proba = predict_emoji_sentiment(emojis_only, emoji_vectorizer, emoji_model)
    
    # Combine predictions
    combined_proba = combine_predictions(text_proba, emoji_proba)
    
    # Decode final sentiment
    final_sentiment, probabilities = decode_sentiment(combined_proba, label_encoder)
    
    return final_sentiment, probabilities

# Example usage
if __name__ == "__main__":
    user_input = input("Enter text with emojis: ")
    
    # Analyze sentiment
    final_sentiment, probabilities = analyze_sentiment(
        user_input, 
        text_vectorizer, text_model, 
        emoji_vectorizer, emoji_model, 
        label_encoder
    )
    
    # Output results
    print(f"\nFinal Sentiment: {final_sentiment}")
    if probabilities is not None:
        print("Probabilities for each sentiment:")
        for sentiment, prob in zip(label_encoder.classes_, probabilities[0]):
            print(f"{sentiment}: {prob:.4f}")
