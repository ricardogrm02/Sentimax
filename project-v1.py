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
    
    vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 10000)  # TfidfVectorizer turns the text into numerical values for models
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

    MNB_Model = MultinomialNB()                                                                                     # Make Multinomial Naive Bayes → good for counting, like how often words appear in document
    LR_Model = LogisticRegression(solver='saga', class_weight='balanced', max_iter=1000)                            # Make Logistic Regression → predicts a probability of something  happening, like whether a review is + or -
    C_Model = ComplementNB()                                                                                        # Make Complement Naive Bayes → similar to MNB, but better if categories have more examples than others
    B_Model = BernoulliNB()                                                                                         # Make Bernoulli Naive Bayes → works best when data is yes/no, true/false, 0/1
    KNN_Model = KNeighborsClassifier(n_neighbors=3)                                                                 # Make KNN → find things that are similar to each other and make predictions based on that
    RF_Model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)                   # Make Random Forest Classifier → asking forest of decision trees to make group decision; good w/ complex data
    SVM_Model = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)                    # Make Support Vector Model → finds best way to draw line to separate things into categories
    SGDC_Model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, class_weight='balanced', random_state=42)  # Make Stochastic Gradient Descent Classifier → fast and simple method for making predictions, esp w/ large data

    ensemble_model = VotingClassifier(      # I create an ensemble to utilize multiple models to maximize accuracy using votingclassifier
        estimators=[                        # VotingClassifier combines predictions from all involved models to make a final decision
            ('nb', MNB_Model),              # List of tuples that is necesary for making ensembles
            ('lr', LR_Model),
            ('c', C_Model),
            ('b', B_Model),
            ('knn', KNN_Model),
            ('rf', RF_Model),
            ('svm', SVM_Model),
            ('sgdc', SGDC_Model)
        ],
        voting='soft'                       # Soft voting because I want the avg probabilities instead of a final prediction
    )
    
    ensemble_model.fit(X_train, y_train)    # Train the ensemble model with input X_train and labels y_train
                                            # fit() is used to train the model by learning patterns in the given data
                                            # After training, the ensemble will combine the predictions of individual models
   
    y_pred = ensemble_model.predict(X_test) # Using the newly trained ensemble model, predict the sentiment labels (e.g., "happy", "angry") using input X_test
                                            # Save the predicted sentiment labels into y_pred
   
    print(classification_report(y_test, y_pred, target_names = le.classes_))# classification_report is used to check performance of model w/ precision, recall, f1, support
                                                                            # Compares the predicted labels (y_pred) to the test labels (y_test)
                                                                            # target_names is set to le.classes_ to map the numerical values back to their original sentiments w/ .classes_

    joblib.dump(ensemble_model, model_path) # Save the ensemble model into the predetermined model path; save model for reuse
    joblib.dump(vectorizer, vectorizer_path)# Save the vectorizer into the predetermined vectorizer path; save vectorizer for reuse
    joblib.dump(le, label_encoder_path)     # Save the le into the predetermined le path; save le for reuse
    print("Model, vectorizer, and label encoder trained and saved successfully.") 
    return ensemble_model, vectorizer, le   # Returns the model, vectorizer, and le

# Function to read text from an image using EasyOCR ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
def read_image(userInput):
    reader = easyocr.Reader(['en'])                 # Set the language to english only
    image_path = userInput + '.JPG'                 # Setup the path to the image
    result = reader.readtext(image_path, detail=0)  # Read the text with the built in ocr function .readtext
    concatenated_text = " ".join(result)            # Combine all of the text to make a hopefully cohesive sentence
    return concatenated_text                        # Return the sentence

# Main function of the code ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
if __name__ == "__main__":
    # Check if model, vectorizer, and label encoder are already saved
    if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(label_encoder_path): # Enter this if all of the necessary paths exist
        user_choice = input("Model, vectorizer, and label encoder already exist. Do you want to:\n1) Use the existing model\n2) Delete and create a new model\nEnter 1 or 2: ")
        if user_choice == '1':                          # Load the existing model, vectorizer, and label encoder
            ensemble_model = joblib.load(model_path)    # Load the model in the model path into the ensemble_model variable
            vectorizer = joblib.load(vectorizer_path)   # Load the vectorizer in the vectorizer path into the vectorizer variable
            le = joblib.load(label_encoder_path)        # Load the label encoder in the le path into the le variable
            print("Existing model, vectorizer, and label encoder loaded successfully.")
        elif user_choice == '2':                                # Delete existing model and train a new one
            os.remove(model_path)                               # Delete currently saved model
            os.remove(vectorizer_path)                          # Delete currently saved vectorizer
            os.remove(label_encoder_path)                       # Delete currently saved le 
            print("Existing model, vectorizer, and label encoder deleted.")
            ensemble_model, vectorizer, le = train_ensemble()   # Call the train function to get all of the necessary files
        else:
            print("Invalid choice. Exiting.")
            exit()
    else:                                                   # Enter this if all of the necessary paths don't exist
        ensemble_model, vectorizer, le = train_ensemble()   # Call the train function to get all of the necessary files

    mode = int(input("Please Select a method:\n1) Insert Text\n2) Insert Image\nEnter your choice: "))  # Pick to break down from text or image
    if mode == 1:   # 1 for text
        userInput = input("Please input text: ")
    elif mode == 2: # 2 for image
        userInput = read_image(input("What is the name of the image file: "))
    
    new_text_transformed = vectorizer.transform([userInput])    # Transform the input into numerical values for the ML model to read
                                                                # userInput is put into a list because the vectorizer can only deal with lists of words

    ensemble_proba = ensemble_model.predict_proba(new_text_transformed)     # Use the ensemble model to predict the sentiment probabilities of the numerical values from the vectorizer
                                                                            # predict_proba allows us to predict the probabilities of all possible sentiments for the altered user input

    ensemble_classes = le.inverse_transform(np.arange(len(ensemble_model.classes_)))    # Retrieve the sentiments (class labels)
                                                                                        # len(ensemble_model.classes_) tells us how many unique sentiments (class labels) there are
                                                                                        # np.arrange makes a list of integers that is the value of its parameter
                                                                                        # np.arrange here would be 0 to len(ensemble_model.classes_)-1
                                                                                        # inverse_transform maps the numerical values back to their original sentiments (class labels)
                                                                                        # So we retrieve the original sentiments and then save them into the list that np.arrange has created
    
    top_indices = np.argsort(ensemble_proba[0])[::-1]   # This turns the indices into reverse order from largest to smallest probabilities
                                                        # np.argsort automatically sorts whatever is in the parenthesis from smallest to largest
                                                        # [::-1] is responsible for the reversing of the order that np.argsort did
                                                        # ensemble_proba[0] accesses the probabilities for the first input in the array
                                                        # If we had multiple inputs, then we would need to access other indices of ensemble_proba
                                                        # top_indices stores the indices of the sentiments (class labels), sorted by their probabilities

    top_emotions = [(ensemble_classes[index], ensemble_proba[0][index]) for index in top_indices]   # Create list of tuples of sentiment (class label) with its corresponding probability
                                                                                                    # For each index, save into top_emotions the (ensemble_classes[index], ensemble_proba[0][index])
                                                                                                    # ensemble_classes is sentiment (class label)
                                                                                                    # ensemble_proba[0][index] is probaility of the sentiment 
                                                                                                    # These sync up because the indices match, ensuring the probabilities align with the correct sentiments
                                                                                                    # The resulting list is ranked from the highest probability sentiment to the lowest bc of previous argsort and [::-1]


    print("\nPredicted Emotions:")              # Output emotions
    for emotion, probability in top_emotions:   # Go through the list of tuples and print each tuple out
        print(f"Ensemble Emotion: {emotion}, Probability: {probability:.4f}")