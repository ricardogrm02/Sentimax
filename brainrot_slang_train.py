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
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

# Define paths to save the model, vectorizer, and label encoder
model_path = 'brainrot_slang_ensemble_model.pkl'
vectorizer_path = 'brainrot_slang_vectorizer.pkl'
label_encoder_path = 'label_encoder.pkl'

# Function to train and save the model
def train_ensemble():
    # Load the slang datase
    data = pd.read_csv("slang_brainrot_unique_emotions.csv")
    data = data.dropna(subset=['content'])  # Remove rows with missing content
    data = data[data['content'].str.strip() != '']  # Remove rows with empty strings

    # Initialize a vectorizer with emoji-friendly tokenization
    vectorizer = TfidfVectorizer(
        max_features=10000,
        token_pattern=r"(?u)(?:\w+|\S)",  # Include emojis as valid tokens
        stop_words=None  # Avoid removing tokens
    )
    
    # Vectorize the 'content' column
    X = vectorizer.fit_transform(data['content'])

    # Encode sentiment labels
    le = LabelEncoder()
    y = le.fit_transform(data['sentiment'])

    # Split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models with hyperparameters
    MNB_Model = MultinomialNB()
    LR_Model = LogisticRegression(solver='saga', class_weight='balanced', max_iter=1000)
    C_Model = ComplementNB()
    B_Model = BernoulliNB()
    KNN_Model = KNeighborsClassifier(n_neighbors=3)
    RF_Model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
    SVM_Model = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
    SGDC_Model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, class_weight='balanced', random_state=42)

    # Create an ensemble model
    ensemble_model = VotingClassifier(
        estimators=[
            ('nb', MNB_Model),
            ('lr', LR_Model),
            ('c', C_Model),
            ('b', B_Model),
            ('knn', KNN_Model),
            ('rf', RF_Model),
            ('svm', SVM_Model),
            ('sgdc', SGDC_Model)
        ],
        voting='soft'
    )
    
    # Train the ensemble model
    ensemble_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = ensemble_model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save the trained model, vectorizer, and label encoder
    joblib.dump(ensemble_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(le, label_encoder_path)
    print("Model, vectorizer, and label encoder trained and saved successfully.")
    return ensemble_model, vectorizer, le


# Function to read text from an image using EasyOCR
def read_image(userInput):
    reader = easyocr.Reader(['en'])
    image_path = userInput + '.JPG'
    result = reader.readtext(image_path, detail=0)
    concatenated_text = " ".join(result)
    return concatenated_text

if __name__ == "__main__":
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
            ensemble_model, vectorizer, le = train_ensemble()
        else:
            print("Invalid choice. Exiting.")
            exit()
    else:
        # Train and save the model if it doesn't exist
        ensemble_model, vectorizer, le = train_ensemble()

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

    # Output emotions
    print("\nPredicted Emotions:")
    for emotion, probability in top_5_emotions:
        print(f"Ensemble Emotion: {emotion}, Probability: {probability:.4f}")
