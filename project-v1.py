import os
import pandas as pd
import numpy as np
import easyocr
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Define paths to save the model and vectorizer
model_path = 'sentimax_ensemble_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

# Check if model and vectorizer are already saved
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    # Load the model and vectorizer if they exist
    ensemble_model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model and vectorizer loaded successfully.")
else:
    # Load data and train model if not already saved
    file_path = 'final_data_v2 - final_data_v2.csv'
    data = pd.read_csv(file_path)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Convert text to numerical data
    X = vectorizer.fit_transform(data['content'])
    y = data['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    MNB_Model = MultinomialNB()
    LR_Model = LogisticRegression(solver='saga')
    C_Model = ComplementNB()
    B_Model = BernoulliNB()
    KNN_Model = KNeighborsClassifier(n_neighbors=5)

    # Create ensemble model
    ensemble_model = VotingClassifier(
        estimators=[('nb', MNB_Model), ('lr', LR_Model), ('c', C_Model), ('b', B_Model), ('knn', KNN_Model)],
        voting='soft'
    )
    
    # Train the ensemble model
    ensemble_model.fit(X_train, y_train)

    # Save the trained model and vectorizer
    joblib.dump(ensemble_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print("Model and vectorizer trained and saved successfully.")

# Function to read text from an image using EasyOCR
def read_image(userInput):
    reader = easyocr.Reader(['en']) # Initialize the reader (specify the language, e.g., 'en' for English)
    image_path = userInput + '.JPG' # Path to the image
    result = reader.readtext(image_path, detail=0) # Perform OCR on the image (detail=0 returns only text without bounding boxes)
    concatenated_text = "" # Variable to store concatenated result
    for line in result: # Process each line in the result
        if not line.endswith('\n'):
            concatenated_text += line + " " # If the line doesn't end with a newline, concatenate it with the next one
        else:
            concatenated_text += line # Add the line to the concatenated text
    return concatenated_text

# Main function for user input
mode = int(input("Please Select a method:\n1) Insert Text\n2) Insert Image\nEnter your choice: "))
if mode == 1:
    userInput = input("Please input text: ")
elif mode == 2:
    userInput = read_image(input("What is the name of the image file: "))

# Transform the new input using the loaded vectorizer
new_text_transformed = vectorizer.transform([userInput])

# Predict sentiment probabilities with the ensemble model
ensemble_proba = ensemble_model.predict_proba(new_text_transformed)

# Retrieve the class labels
ensemble_classes = ensemble_model.classes_

# Get top 5 predicted emotions from the ensemble
top_5_indices = np.argsort(ensemble_proba[0])[::-1][:5]
top_5_emotions = [(ensemble_classes[index], ensemble_proba[0][index]) for index in top_5_indices]

# Output the top 5 emotions
for emotion, probability in top_5_emotions:
    print(f"Ensemble Emotion: {emotion}, Probability: {probability:.4f}")