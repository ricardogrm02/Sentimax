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

# Define paths to save the model, vectorizer, and label encoder ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
model_path = 'text_ensemble_model.pkl'
vectorizer_path = 'text_vectorizer.pkl'
label_encoder_path = 'label_encoder.pkl'

# Function to train and save the model ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ 
def train_ensemble():
    data = pd.read_csv('n.csv') # Load the main text dataset
    
    vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 10000)  # Turn the text into numerical values for models
                                                                                # stop_words = 'english' removes stop words
                                                                                # max_features = 10000 is a 10k word limit based on importance to TfidfVectorizer
    X = vectorizer.fit_transform(data['content'])                               # fit_transform turns the content into the numerical values for models
                                                                                # vectorizer allows us to use fit_transform since it is a TfidfVectorizer 

    le = LabelEncoder()                                                         # Turn labels into numerical values for models
    y = le.fit_transform(data['sentiment'])                                     # fit_transform turns the content into the numerical values for models
                                                                                # le allows us to use fit_transform since it is a LabelEncoder 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   # X_train is input data for training, y_train is label for training
                                                                                                # X_test is input for testing, y_test is label for testing
                                                                                                # X is input data (aka content), y is label (aka sentiment)
                                                                                                # test_size gives us an 80/20 train to test ratio
                                                                                                # random_state determines how the data is picked
                                                                                                # random_state = 42 is a specific rule to have the program pick the same values for both training and test

    # Initialize models with adjusted hyperparameters
    MNB_Model = MultinomialNB()                                                                                     # Make Multinomial Naive Bayes → good for counting, like how often words appear in document
    LR_Model = LogisticRegression(solver='saga', class_weight='balanced', max_iter=1000)                            # Make Logistic Regression → predicts a probability of something  happening, like whether a review is + or -
    C_Model = ComplementNB()                                                                                        # Make Complement Naive Bayes → similar to MNB, but better if categories have more examples than others
    B_Model = BernoulliNB()                                                                                         # Make Bernoulli Naive Bayes → works best when data is yes/no, true/false, 0/1
    KNN_Model = KNeighborsClassifier(n_neighbors=3)                                                                 # Make KNN → find things that are similar to each other and make predictions based on that
    RF_Model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)                   # Make Random Forest Classifier → asking forest of decision trees to make group decision; good w/ complex data
    SVM_Model = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)                    # Make Support Vector Model → finds best way to draw line to separate things into categories
    SGDC_Model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, class_weight='balanced', random_state=42)  # Make Stochastic Gradient Descent Classifier → fast and simple method for making predictions, esp w/ large data

    # Create ensemble model
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
    
    # Train the ensemble model on resampled data
    ensemble_model.fit(X_train, y_train)

    y_pred = ensemble_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save the trained model, vectorizer, and label encoder
    joblib.dump(ensemble_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(le, label_encoder_path)
    print("Model, vectorizer, and label encoder trained and saved successfully.")
    return ensemble_model, vectorizer, le

# Function to read text from an image using EasyOCR ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
def read_image(userInput):
    reader = easyocr.Reader(['en'])
    image_path = userInput + '.JPG'
    result = reader.readtext(image_path, detail=0)
    concatenated_text = " ".join(result)
    return concatenated_text

# Main function of the code ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
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