import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import easyocr
import joblib
from io import BytesIO
import base64
from combine import analyze_sentiment

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

# Import SMOTE for handling class imbalance
from imblearn.over_sampling import SMOTE

# Define paths to save the model, vectorizer, and label encoder
text_model_path = 'text_ensemble_model.pkl'
text_vectorizer_path = 'text_vectorizer.pkl'

#Define slang model paths
slang_model_path = "brainrot_slang_ensemble_model.pkl"
slang_vectorizer_path = 'brainrot_slang_vectorizer.pkl'

#Define emoji model paths
emoji_model_path = "emoji_ensemble_model.pkl"
emoji_vectorizer_path = 'emoji_vectorizer.pkl'

label_encoder_path = 'label_encoder.pkl'


# Function to ensure that input isn't empty
def checkValidInput(form_response):
    if len(form_response) > 0:
        return 200
    elif form_response is None or len(form_response) == 0:
        return 403

# Function to train and save the text model
def train_and_save_text_model():
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
    SVM_Model = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
    SGDC_Model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, class_weight='balanced', random_state=42)

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
    ensemble_model.fit(X_train_res, y_train_res)

    y_pred = ensemble_model.predict(X_test)
    #Printing Out the Classification Report
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save the trained model, vectorizer, and label encoder
    joblib.dump(ensemble_model, text_model_path)
    joblib.dump(vectorizer, text_vectorizer_path)
    joblib.dump(le, label_encoder_path)
    print("Text model, text vectorizer, and label encoder trained and saved successfully.")
    return ensemble_model, vectorizer, le

# Function to train and save the model
def train_and_save_slang_model():
    file_path = "slang_brainrot_unique_emotions.csv"
    data = pd.read_csv(file_path)
    
    # Check and clean the dataset
    print(f"Number of missing entries in 'content': {data['content'].isnull().sum()}")
    print(data.head())  # Check the first few rows
    data = data.dropna(subset=['content'])  # Remove rows with missing content
    data = data[data['content'].str.strip() != '']  # Remove rows with empty strings

    # Initialize a vectorizer with emoji-friendly tokenization
    vectorizer = TfidfVectorizer(
        max_features=10000,
        token_pattern=r"(?u)(?:\w+|\S)",  # Include emojis as valid tokens
        stop_words=None  # Avoid removing tokens
    )

    # Increase max_features to capture more text features
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)


    X = vectorizer.fit_transform(data['content'])

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(data['sentiment'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Initialize models with adjusted hyperparameters
    MNB_Model = MultinomialNB()
    LR_Model = LogisticRegression(solver='saga', class_weight='balanced', max_iter=1000)
    C_Model = ComplementNB()
    B_Model = BernoulliNB()
    KNN_Model = KNeighborsClassifier(n_neighbors=3)
    RF_Model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
    SVM_Model = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
    SGDC_Model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, class_weight='balanced', random_state=42)

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
    ensemble_model.fit(X_train_res, y_train_res)

    y_pred = ensemble_model.predict(X_test)
    #Printing Out the Classification Report
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save the trained model, vectorizer, and label encoder
    joblib.dump(ensemble_model, slang_model_path)
    joblib.dump(vectorizer, slang_vectorizer_path)
    print("Slang Model and Vectorizer and saved successfully.")
    return ensemble_model, vectorizer, le


# Function to train and save the model
def train_and_save_emoji_model():
    file_path = "emoji_emotions.csv"
    data = pd.read_csv(file_path)
    
    # Check and clean the dataset
    print(f"Number of missing entries in 'content': {data['content'].isnull().sum()}")
    print(data.head())  # Check the first few rows
    data = data.dropna(subset=['content'])  # Remove rows with missing content
    data = data[data['content'].str.strip() != '']  # Remove rows with empty strings

    # Initialize a vectorizer with emoji-friendly tokenization
    vectorizer = TfidfVectorizer(
        max_features=10000,
        token_pattern=r"(?u)(?:\w+|\S)",  # Include emojis as valid tokens
        stop_words=None  # Avoid removing tokens
    )


    X = vectorizer.fit_transform(data['content'])

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(data['sentiment'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Initialize models with adjusted hyperparameters
    MNB_Model = MultinomialNB()
    LR_Model = LogisticRegression(solver='saga', class_weight='balanced', max_iter=1000)
    C_Model = ComplementNB()
    B_Model = BernoulliNB()
    KNN_Model = KNeighborsClassifier(n_neighbors=3)
    RF_Model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
    SVM_Model = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
    SGDC_Model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, class_weight='balanced', random_state=42)

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
    ensemble_model.fit(X_train_res, y_train_res)

    y_pred = ensemble_model.predict(X_test)
    #Printing Out the Classification Report
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save the trained model, vectorizer, and label encoder
    joblib.dump(ensemble_model, emoji_model_path)
    joblib.dump(vectorizer, emoji_vectorizer_path)
    print("Emoji model, and emoji vectorizer trained and saved successfully.")
    return ensemble_model, vectorizer, le

# Function to read text from an image using EasyOCR
def read_image(image_bytes):
    reader = easyocr.Reader(['en'])
    
    # EasyOCR can read from a BytesIO object directly
    result = reader.readtext(image_bytes, detail=0)
    
    # Concatenate the text found in the image
    concatenated_text = " ".join(result)
    return concatenated_text


# Check if model, vectorizer, and label encoder are already saved
def initialize_model(form_response): 
    if not os.path.exists(text_model_path) and not os.path.exists(text_vectorizer_path):
      ensemble_model, vectorizer, le = train_and_save_text_model()
    
    if os.path.exists(text_model_path) and os.path.exists(text_vectorizer_path):
        # Load the existing model, vectorizer, and label encoder
        ensemble_model = joblib.load(text_model_path)
        vectorizer = joblib.load(text_vectorizer_path)
        le = joblib.load(label_encoder_path)
        print("Exsting TEXT model, vectorizer, and label encoder loaded successfully.")


    if not os.path.exists(slang_model_path) and not os.path.exists(slang_vectorizer_path):
      slang_model, slang_vectorizer = train_and_save_slang_model()
    
    if os.path.exists(slang_model_path) and os.path.exists(slang_model_path):
        # Load the existing model, vectorizer, and label encoder
        slang_model = joblib.load(slang_model_path)
        slang_vectorizer = joblib.load(slang_vectorizer_path)
        print("Exsting SLANG model, vectorizer, and label encoder loaded successfully.")


    if not os.path.exists(emoji_model_path) and not os.path.exists(emoji_vectorizer_path):
        emoji_model, emoji_vectorizer = train_and_save_emoji_model()

    if os.path.exists(emoji_model_path) and os.path.exists(emoji_vectorizer_path):
        # Load the existing model, vectorizer, and label encoder
        emoji_model = joblib.load(emoji_model_path)
        emoji_vectorizer = joblib.load(emoji_vectorizer_path)
        print("Exsting EMOJI model, vectorizer, and label encoder loaded successfully.")



    # Main function for user input
    # if mode == 1:
    #     userInput = input("Please input text: ")
    # elif mode == 2:
    #     userInput = read_image(input("What is the name of the image file: "))
    #     print(userInput)

    if isinstance(form_response, str):
        userInput = form_response
    elif isinstance(form_response, bytes):
        userInput = read_image(form_response)

    final_sentiment, sorted_sentiments = analyze_sentiment(userInput, vectorizer, ensemble_model, emoji_vectorizer, emoji_model, slang_vectorizer, slang_model, le)
    print(sorted_sentiments)

    bar_graph_labels = [sentiment for sentiment, _ in sorted_sentiments]
    bar_graph_values = [prob for _, prob in sorted_sentiments]

    print(bar_graph_labels)
    print(bar_graph_values)

    # # Transform the new input using the loaded vectorizer
    # new_text_transformed = vectorizer.transform([userInput])

    # # Predict sentiment probabilities with the ensemble model
    # ensemble_proba = ensemble_model.predict_proba(new_text_transformed)

    # # Retrieve the class labels (decode them)
    # ensemble_classes = le.inverse_transform(np.arange(len(ensemble_model.classes_)))

    # # Get top 5 predicted emotions from the ensemble
    # top_5_indices = np.argsort(ensemble_proba[0])[::-1]
    # top_5_emotions = [(ensemble_classes[index], ensemble_proba[0][index]) for index in top_5_indices]

    # Output the top 5 emotions
    # print("\nTop 5 Predicted Emotions:")
    # for emotion, probability in top_5_emotions:
    #     print(f"Ensemble Emotion: {emotion}, Probability: {probability:.4f}")

    # #Creating Pie Chart
    # explode = [0.05] * len(ensemble_classes)
    # plt.figure(figsize=(7, 7))
    # plt.pie(ensemble_proba.ravel(), labels=ensemble_classes, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors, explode=explode, labeldistance=1.1, pctdistance=0.9)
    # plt.title('Emotion Probablity Distribution')
    # plt.axis('equal') 
    
    #Initalizing and creating a new bar graph
    plt.figure()
    #Limiting the Y Values to range from 0 to 1
    plt.ylim(0, 1)
    #Creating a bar graph from model labels and predicited probablities
    bars = plt.bar(bar_graph_labels, bar_graph_values, color= "#26f7fd", width= 0.65)
    # Rotate the X-axis labels to prevent overlapping, and assigining each label to the bar to the right
    plt.xticks(rotation=45, ha='right') 
    #Setting the different Y values in the graph
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  
    #Giving a label name to each axis and titling the graph
    plt.xlabel('Emotions')
    plt.ylabel('Probability of Emotion')
    plt.title('Probability Distribution Bar Graph')
    #Only Showing to test without pushing to frontend
    # plt.show()

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2 + 0.02, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=8)
        
    plt.tight_layout()

    #Saving the bar graph to a bytes buffer 
    image_buffer = BytesIO()
    #Converting the image to jpeg file
    plt.savefig(image_buffer, format = 'jpeg')
    image_buffer.seek(0)

    #Turning the bytes buffer to a base64 string to use as src for piechart image in front end
    encoded_bar_graph = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
    image_buffer.close()

    #Returning the Text from the user, and the encoded pie chart
    return userInput, encoded_bar_graph



#Just a Simple test function to ensure that user input from front end was being sent to the python script backend, can be removed
def text_handler(text):
    print("Text Successfuly sent to the python script")
    return True
