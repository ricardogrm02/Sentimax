import joblib
import emoji
import numpy as np
import os
from sklearn.metrics import classification_report
from text_train_from_project_v1 import train_ensemble as text_model_setup
from emoji_train import train_ensemble as emoji_model_setup
from brainrot_slang_train import train_ensemble as slang_model_setup

# Ensuring that ALL the necessary pkl files for each ensemble model exist before loading them ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
if not os.path.exists('text_ensemble_model.pkl') or not os.path.exists('text_vectorizer.pkl'): # Check if either the text_ensemble_model.pkl file or text_vectorizer.pkl file are missing
    text_model_setup() #Create both the text_ensemble_model.pkl file and text_vectorizer.pkl file

if not os.path.exists('emoji_ensemble_model.pkl') or not os.path.exists('emoji_vectorizer.pkl'): # Check if either the emoji_ensemble_model.pkl file or emoji_vectorizer.pkl file are missing
    emoji_model_setup() #Create both the emoji_ensemble_model.pkl file and emoji_vectorizer.pkl file

if not os.path.exists('brainrot_slang_ensemble_model.pkl') or not os.path.exists('brainrot_slang_vectorizer.pkl'): # Check if either the brainrot_slang_ensemble_model.pkl file or brainrot_slang_vectorizer.pkl file are missing
    slang_model_setup() #Create both the brainrot_slang_ensemble_model.pkl file and brainrot_slang_vectorizer.pkl file

# Load the existing model, vectorizer, and label encoder ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
text_model = joblib.load("text_ensemble_model.pkl")                 # Load the model in the model path into the text_model variable
text_vectorizer = joblib.load("text_vectorizer.pkl")                # Load the vectorizer in the vectorizer path into the text_vectorizer variable

emoji_model = joblib.load("emoji_ensemble_model.pkl")               # Load the model in the model path into the emoji_model variable
emoji_vectorizer = joblib.load("emoji_vectorizer.pkl")              # Load the vectorizer in the vectorizer path into the emoji_vectorizer variable

brainrot_model = joblib.load("brainrot_slang_ensemble_model.pkl")   # Load the model in the model path into the brainrot_model variable
brainrot_vectorizer = joblib.load("brainrot_slang_vectorizer.pkl")  # Load the vectorizer in the vectorizer path into the brainrot_vectorizer variable

label_encoder = joblib.load("label_encoder.pkl")                    # Load the label encoder in the le path into the label_encoder variable

# Preprocess input to separate text, emojis, and brainrot slang ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
def preprocess_input(input_text):                                                       # Separate emojis and text
    text_only = "".join([char for char in input_text if not char in emoji.EMOJI_DATA])  # Takes input and iterates through each character
                                                                                        # If emoji, don't append into resulting list
                                                                                        # Resulting list is compressed into sentence by "".join and saved into text_only
    emojis_only = "".join([char for char in input_text if char in emoji.EMOJI_DATA])    # Takes input and iterates through each character
                                                                                        # If text, don't append into resulting list
                                                                                        # Resulting list is compressed into sentence by "".join and saved into emoji_only
    brainrot_slang_input = input_text.strip()                                           # For brainrot slang, process the entire input                                           
                                                                                        # .strip removes all leading and trailing whitespace (tabs, spaces, newlines) 
                                                                                        # Then store the filtered string into brainrot_slang_input
    return text_only.strip(), emojis_only.strip(), brainrot_slang_input                 # Return cleaned text, cleaned emojis, cleaned brainrot as a tuple

# Predict sentiment for text ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
def predict_text_sentiment(text, text_vectorizer, text_model):                          # Use text model and text vectorizer to predict sentiment for text
    if text:                                                                            # Only activates if there is text
        vectorized_text = text_vectorizer.transform([text])                             # Use vectorizer to turn the text into numerical values for the model to use
                                                                                        # Vectorizer only accepts lists so we have to put the text into a list before transforming
                                                                                        # transform function is what turns the text list into numerical values for the ML model
        text_proba = text_model.predict_proba(vectorized_text)                          # Use the ML model to predict the probabilities of the vectorized text
        return text_proba                                                               # Return the probabilities of sentiments (class labels)
    return None                                                                         # No text to process if return none

# Predict sentiment for emojis ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ 
def predict_emoji_sentiment(emojis, emoji_vectorizer, emoji_model):                     # Use emoji model and emoji vectorizer to predict sentiment for emoji
    if emojis:                                                                          # Only activates if there is emoji
        vectorized_emojis = emoji_vectorizer.transform([emojis])                        # Use vectorizer to turn the emoji into numerical values for the model to use
                                                                                        # Vectorizer only accepts lists so we have to put the text into a list before transforming
                                                                                        # transform function is what turns the emoji list into numerical values for the ML mode
        emoji_proba = emoji_model.predict_proba(vectorized_emojis)                      # Use the ML model to predict the probabilities of the vectorized emoji
        return emoji_proba                                                              # Return the probabilities of sentiments (class labels)
    return None                                                                         # No emoji to process if return none

# Predict sentiment for brainrot slang ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
def predict_brainrot_slang_sentiment(brainrot_input, brainrot_vectorizer, brainrot_model):  # Use brainrot model and brainrot vectorizer to predict sentiment for brainrot
    if brainrot_input:                                                                      # Only activate if there is brainrot
        vectorized_brainrot = brainrot_vectorizer.transform([brainrot_input])               # Use vectorizer to turn the brainrot into numerical values for the model to use 
                                                                                            # Vectorizer only acepts llists so we have to put the text into a list before transforming
                                                                                            # transform function is what turns the emoji list into numerical values for the ML mode
        brainrot_proba = brainrot_model.predict_proba(vectorized_brainrot)                  # Use the ML model to predict the probabilities of the vectorized emoji
        return brainrot_proba                                                               # Return the probabilities of the sentiments (class labels)
    return None                                                                             # No brainrot slang to process if return none                                          

# Combine predictions ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
def combine_predictions(text_proba, emoji_proba, brainrot_proba, text_weight = 0.5, emoji_weight = 0.3, brainrot_weight = 0.2): 
                                                                            # Determine the combined probability
    combined_proba = None                                                   # The combined probability default is none 
                                                                            # Weight of emoji is 0.3 since it is more common than brainrot
                                                                            #  Weight of text is 0.5 since it is most common
                                                                            # Weight of brainrot is 0.2 since it is the least common 
    if text_proba is not None:                                              # Activate if there is text probability
        combined_proba = text_weight * text_proba                           # Set combined_proba to text probaility * text weight
    if emoji_proba is not None and combined_proba is not None:              # Activate if there is emoji probability and combined_proba exists
        combined_proba = combined_proba + (emoji_weight * emoji_proba)      # Set combined_proba to the combined prob + (emoji probability * emoji weight)
    elif emoji_proba is not None and combined_proba is None:                # Activate if ther is emoji probability and combined_proba doesn't exist
        combined_proba = emoji_weight * emoji_proba                         # Set combined_proba to the emoji probaility * emoji weight
    if brainrot_proba is not None and combined_proba is not None:           # Activate if there is brainrot probability and combined_proba exists
        combined_proba = combined_proba + (brainrot_weight * brainrot_proba)# Set combined_proba to combined_proba + (brainrot probability * brainrot weight)
    elif brainrot_proba is not None and combined_proba is None:             # Activate if there is brain probaility and combined_proba doesn't exist 
        combined_proba = brainrot_weight * brainrot_proba                   # Set combined_proba to the brainrot probaility * brainrot weight
    return combined_proba                                                   # Return the combined probability

# Decode and display the sentiment ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
def decode_sentiment(combined_proba, label_encoder):    # Translate sentiment predictions into probabilities
    if combined_proba is not None:                      # Activates if combined_proba exist
        sorted_indices = np.argsort(combined_proba[0])[::-1]    # Sort probabilities in descending order
                                                                # argsort orders the probabilities from small to largest
                                                                # [::-1] reverses the order, so it is now largest to smalelst
                                                                # We do [0] only since we only deal with one input of text at a time
                                                                # If there were multiple inputs of text, then we would need to iterate
        sorted_sentiments = [(label_encoder.inverse_transform([i])[0], combined_proba[0][i]) for i in sorted_indices]   
                                                # We make a list of tuples sentiment, probability
                                                # We save label_encoder.inverse_transform([i])[0], combined_proba[0][i] 
                                                # Save the tuple into each index of sorted_sentiments
                                                # label_encoder is a list of numerical values, but it is turned back into sentiments (class_labels) with inverse_transform
                                                # The i indicates which sentiment to transform, and the [0] makes sure that only one sentiment gets transformed
                                                # For combined proba, since it's technically a 2d list with only one row, we do [0] to stay on row and [i] to change columns
        top_sentiment = sorted_sentiments[0][0] # The sentiment with the highest probability
        return top_sentiment, sorted_sentiments
    return None

# Evaluate and print F1 scores ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ 
def evaluate_model(y_test, y_pred, label_encoder):                          # Create Classification Report
    report = classification_report(y_test, y_pred,                          # y_test are the true class labels and y_pred are the predicted class labels
                                   target_names = label_encoder.classes_,   # target_names are the sentiments (class labels)
                                                                            # .classes_ allows only the unique sentiments (class labels) to be used
                                   output_dict = True)                      # Set output_dict to True so it shoes an actual like, table
                                                                            # If output_dict is set to False, it just prints out a string
    print("\nEvaluation Metrics:")
    for sentiment, metrics in report.items():                               # .items lets the code iterate through both keys and values in a dictionary; make tuples easy
                                                                            # sentiment is the sentiment, and the metrics is the value of the report like f1, etc
        if isinstance(metrics, dict):                                       # isinstance checks if the first param is in second param, so is current metric in dict
                                                                            # sentiment is a string of the sentiment
                                                                            # metrics is a dictionary of string key and float values
                                                                            # ie sentiment, metrics → "happy": {"precision": 0.85, "recall": 0.8, "f1-score": 0.825, "support": 20}
                                                                            # there is a colon to separate them because that's just how tuples are separated
            print(f"{sentiment.capitalize()}: Precision = {metrics['precision']:.4f}, Recall = {metrics['recall']:.4f}, F1-Score = {metrics['f1-score']:.4f}")

# Full sentiment analysis pipeline ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ 
def analyze_sentiment(input_text, text_vectorizer, text_model, emoji_vectorizer, emoji_model, brainrot_vectorizer, brainrot_model, label_encoder):
    text_only, emojis_only, brainrot_slang_input = preprocess_input(input_text)                                 # Preprocess input
    
    text_proba = predict_text_sentiment(text_only, text_vectorizer, text_model)                                 # Predict sentiments
    emoji_proba = predict_emoji_sentiment(emojis_only, emoji_vectorizer, emoji_model)                           # Predict sentiments
    brainrot_proba = predict_brainrot_slang_sentiment(brainrot_slang_input, brainrot_vectorizer, brainrot_model)# Predict sentiments
    
    combined_proba = combine_predictions(text_proba, emoji_proba, brainrot_proba)                               # Combine the predictions
    
    final_sentiment, sorted_sentiments = decode_sentiment(combined_proba, label_encoder)                        # Decode final sentiment
    
    return final_sentiment, sorted_sentiments

# Main Function ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ 
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