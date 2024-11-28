import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import easyocr
import joblib
from io import BytesIO
import base64

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


# Function to ensure that input isn't empty
def checkValidInput(form_response):
    if len(form_response) > 0:
        return 200
    elif form_response is None or len(form_response) == 0:
        return 403

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
    #Printing Out the Classification Report
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save the trained model, vectorizer, and label encoder
    joblib.dump(ensemble_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(le, label_encoder_path)
    print("Model, vectorizer, and label encoder trained and saved successfully.")
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
    if not os.path.exists(model_path) and not os.path.exists(vectorizer_path) and not os.path.exists(label_encoder_path):
      ensemble_model, vectorizer, le = train_and_save_model()

    if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(label_encoder_path):
            # Load the existing model, vectorizer, and label encoder
            ensemble_model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            le = joblib.load(label_encoder_path)
            print("Exsting model, vectorizer, and label encoder loaded successfully.")

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

    #Creating Pie Chart
    explode = [0.05] * len(ensemble_classes)
    plt.figure(figsize=(7, 7))
    plt.pie(ensemble_proba.ravel(), labels=ensemble_classes, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors, explode=explode, labeldistance=1.1, pctdistance=0.9)
    plt.title('Emotion Probablity Distribution')
    plt.axis('equal') 

    #Saving the pie chart to a bytes buffer before
    image_buffer = BytesIO()
    #Converting the image to jpeg file
    plt.savefig(image_buffer, format = 'jpeg')
    image_buffer.seek(0)

    #Turning the bytes buffer to a base64 string to use as src for piechart image in front end
    encoded_pie_chart = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
    image_buffer.close()


    ''' 
    In the Event we want to include a last minute bar graph, its created and functional
    just have to convert it to base 64 string, and include it in the results pages as the src in an <img> tag
    #Initalizing a new graph
    plt.figure()
    #Limiting the Y Values to range from 0 to 1
    plt.ylim(0, 1)
    #Creating a bar graph from model labels and predicited probablities
    plt.bar(ensemble_classes, ensemble_proba.ravel(), color='red')
    # Rotate the X-axis labels to prevent overlapping, and assigining each label to the bar to the right
    plt.xticks(rotation=45, ha='right') 
    #Setting the different Y values in the graph
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  
    #Giving a label name to each axis and titling the graph
    plt.xlabel('Emotions')
    plt.ylabel('Probability of Emotion')
    plt.title('Probability Distribution Bar Graph')
    #Only Showing to test without pushing to frontend
    plt.show()
'''
    #Returning the Text from the user, and the encoded pie chart
    return userInput, encoded_pie_chart



#Just a Simple test function to ensure that user input from front end was being sent to the python script backend, can be removed
def text_handler(text):
    print("Text Successfuly sent to the python script")
    return True