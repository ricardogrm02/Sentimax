import joblib
import emoji
import numpy as np
import os
from sklearn.metrics import classification_report
from text_train_from_project_v1 import train_ensemble as text_model_setup
from emoji_train import train_ensemble as emoji_model_setup
from brainrot_slang_train import train_ensemble as slang_model_setup

if not os.path.exists("text_ensemble_model.pkl") and not os.path.exists("text_vectorizer.pk1"):
    text_model_setup()

if not os.path.exists("emoji_ensemble_model.pkl") and not os.path.exists("emoji_vectorizer.pk1"):
    emoji_model_setup()

if not os.path.exists("brainrot_slang_ensemble_model.pkl") and not os.path.exists("brainrot_slang_vectorizer.pk1"):
    slang_model_setup()

# Load trained models and vectorizers
text_model = joblib.load("text_ensemble_model.pkl")
text_vectorizer = joblib.load("text_vectorizer.pkl")

emoji_model = joblib.load("emoji_ensemble_model.pkl")
emoji_vectorizer = joblib.load("emoji_vectorizer.pkl")

brainrot_model = joblib.load("brainrot_slang_ensemble_model.pkl")
brainrot_vectorizer = joblib.load("brainrot_slang_vectorizer.pkl")

label_encoder = joblib.load("label_encoder.pkl")  # Shared across all models

# Preprocess input to separate text, emojis, and brainrot slang
def preprocess_input(input_text):
    # Separate emojis and text
    text_only = "".join([char for char in input_text if not char in emoji.EMOJI_DATA])
    emojis_only = "".join([char for char in input_text if char in emoji.EMOJI_DATA])
    # For brainrot slang, process the entire input
    brainrot_slang_input = input_text.strip()
    return text_only.strip(), emojis_only.strip(), brainrot_slang_input

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

# Predict sentiment for brainrot slang
def predict_brainrot_slang_sentiment(brainrot_input, brainrot_vectorizer, brainrot_model):
    if brainrot_input:
        vectorized_brainrot = brainrot_vectorizer.transform([brainrot_input])
        brainrot_proba = brainrot_model.predict_proba(vectorized_brainrot)
        return brainrot_proba
    return None  # No brainrot slang to process

# Combine predictions
def combine_predictions(text_proba, emoji_proba, brainrot_proba, text_weight=0.5, emoji_weight=0.3, brainrot_weight=0.2):
    combined_proba = None
    if text_proba is not None:
        combined_proba = text_weight * text_proba
    if emoji_proba is not None:
        combined_proba = combined_proba + (emoji_weight * emoji_proba) if combined_proba is not None else emoji_weight * emoji_proba
    if brainrot_proba is not None:
        combined_proba = combined_proba + (brainrot_weight * brainrot_proba) if combined_proba is not None else brainrot_weight * brainrot_proba
    return combined_proba

# Decode and display the sentiment
def decode_sentiment(combined_proba, label_encoder):
    if combined_proba is not None:
        sorted_indices = np.argsort(combined_proba[0])[::-1]  # Sort probabilities in descending order
        sorted_sentiments = [(label_encoder.inverse_transform([i])[0], combined_proba[0][i]) for i in sorted_indices]
        top_sentiment = sorted_sentiments[0][0]  # The sentiment with the highest probability
        return top_sentiment, sorted_sentiments
    return "No sentiment detected", None

# Evaluate and print F1 scores
def evaluate_model(y_test, y_pred, label_encoder):
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    print("\nEvaluation Metrics:")
    for sentiment, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"{sentiment.capitalize()}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1-Score={metrics['f1-score']:.4f}")

# Full sentiment analysis pipeline
def analyze_sentiment(input_text, text_vectorizer, text_model, emoji_vectorizer, emoji_model, brainrot_vectorizer, brainrot_model, label_encoder):
    # Preprocess input
    text_only, emojis_only, brainrot_slang_input = preprocess_input(input_text)
    
    # Predict sentiments
    text_proba = predict_text_sentiment(text_only, text_vectorizer, text_model)
    emoji_proba = predict_emoji_sentiment(emojis_only, emoji_vectorizer, emoji_model)
    brainrot_proba = predict_brainrot_slang_sentiment(brainrot_slang_input, brainrot_vectorizer, brainrot_model)
    
    # Combine the predictions
    combined_proba = combine_predictions(text_proba, emoji_proba, brainrot_proba)
    
    # Decode final sentiment
    final_sentiment, sorted_sentiments = decode_sentiment(combined_proba, label_encoder)
    
    return final_sentiment, sorted_sentiments

# Example usage
if __name__ == "__main__":
    user_input = input("Enter text with emojis and slang: ")
    
    # Analyze sentiment
    final_sentiment, sorted_sentiments = analyze_sentiment(
        user_input, 
        text_vectorizer, text_model, 
        emoji_vectorizer, emoji_model, 
        brainrot_vectorizer, brainrot_model, 
        label_encoder
    )
    
    # Output results
    print(f"\nFinal Sentiment: {final_sentiment}")
    print("Sentiments (sorted by probability):")
    for sentiment, prob in sorted_sentiments:
        print(f"{sentiment}: {prob:.4f}")
