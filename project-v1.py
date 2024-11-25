import os
import pandas as pd
import numpy as np
import easyocr
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Import SMOTE for handling class imbalance
from imblearn.over_sampling import SMOTE

# Define paths to save the model, vectorizer, and label encoder
model_path = 'sentimax_ensemble_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'
label_encoder_path = 'label_encoder.pkl'

# Function to train and save the model
def train_and_save_model():
    # file_path = 'new_balanced_data.csv'
    file_path = 'n.csv'
    data = pd.read_csv(file_path)
    
    # Increase max_features to capture more text features
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X = vectorizer.fit_transform(data['content'])

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(data['sentiment'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE to handle class imbalance
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Initialize models with adjusted hyperparameters
    MNB_Model = MultinomialNB()
    LR_Model = LogisticRegression(solver='saga', class_weight='balanced', max_iter=1000)
    C_Model = ComplementNB()
    B_Model = BernoulliNB()
    KNN_Model = KNeighborsClassifier(n_neighbors=3)
    RF_Model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)

    # Create ensemble model
    ensemble_model = VotingClassifier(
        estimators=[
            ('nb', MNB_Model),
            ('lr', LR_Model),
            ('c', C_Model),
            ('b', B_Model),
            ('knn', KNN_Model),
            ('rf', RF_Model)
        ],
        voting='soft'
    )
    
    # Train the ensemble model on resampled data
    ensemble_model.fit(X_train_res, y_train_res)

    y_pred = ensemble_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save the trained model, vectorizer, and label encoder
    joblib.dump(ensemble_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(le, label_encoder_path)
    print("Model, vectorizer, and label encoder trained and saved successfully.")
    return ensemble_model, vectorizer, le

# Check if model, vectorizer, and label encoder are already saved
if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(label_encoder_path):
    user_choice = input("Model, vectorizer, and label encoder already exist. Do you want to:\n1) Use the existing model\n2) Delete and create a new model\nEnter your choice (1 or 2): ")
    if user_choice == '1':
        # Load the existing model, vectorizer, and label encoder
        ensemble_model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        le = joblib.load(label_encoder_path)
        print("Existing model, vectorizer, and label encoder loaded successfully.")
    elif user_choice == '2':
        # Delete existing model and train a new one
        os.remove(model_path)
        os.remove(vectorizer_path)
        os.remove(label_encoder_path)
        print("Existing model, vectorizer, and label encoder deleted.")
        ensemble_model, vectorizer, le = train_and_save_model()
    else:
        print("Invalid choice. Exiting.")
        exit()
else:
    # Train and save the model if it doesn't exist
    ensemble_model, vectorizer, le = train_and_save_model()

# Function to read text from an image using EasyOCR
def read_image(userInput):
    reader = easyocr.Reader(['en'])
    image_path = userInput + '.JPG'
    result = reader.readtext(image_path, detail=0)
    concatenated_text = " ".join(result)
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

# Retrieve the class labels (decode them)
ensemble_classes = le.inverse_transform(np.arange(len(ensemble_model.classes_)))

# Get top 5 predicted emotions from the ensemble
top_5_indices = np.argsort(ensemble_proba[0])[::-1]
top_5_emotions = [(ensemble_classes[index], ensemble_proba[0][index]) for index in top_5_indices]

# Output the top 5 emotions
print("\nTop 5 Predicted Emotions:")
for emotion, probability in top_5_emotions:
    print(f"Ensemble Emotion: {emotion}, Probability: {probability:.4f}")
