import pandas as pd
import numpy as np
import easyocr

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.svm import SVC

file_path = 'final_data_v2 - final_data_v2.csv'
data = pd.read_csv(file_path)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Convert text to numerical data
X = vectorizer.fit_transform(data['content'])
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


MNB_Model = MultinomialNB()
MNB_Model.fit(X_train, y_train)
y_pred = MNB_Model.predict(X_test)
# print(classification_report(y_test, y_pred, zero_division=1))


SVM_Model = SVC(kernel='linear', probability=True)
SVM_Model.fit(X_train, y_train)
y_pred = SVM_Model.predict(X_test)
# print(classification_report(y_test, y_pred))

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

# Main function ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
mode = int(input("Input via text (1) or via image (2)?"))
if mode == 1:
    userInput = input("Please input text: ")
elif mode == 2:
    userInput = read_image(input("What is the name of the image file: "))
new_text_transformed = vectorizer.transform([userInput])

# This is to predict a single emotion 
# predicted_sentiment = model.predict(new_text_transformed)

# This prediction gets multiple emotions
MNB_proba = MNB_Model.predict_proba(new_text_transformed)
SVM_proba = SVM_Model.predict_proba(new_text_transformed)

# This is to print a single emotion 
# print(f"The predicted sentiment for the new text is: {predicted_sentiment[0]}")

# This retrieves the labels for the classes
MNB_classes = MNB_Model.classes_
SVM_classes = SVM_Model.classes_

# This is to sort the emotions into a top 5 
MNB_top_5_indices = np.argsort(MNB_proba[0])[::-1][:5]
SVM_top_5_indices = np.argsort(SVM_proba[0])[::-1][:5]

# Get top 5 emotions
MNB_top_5_emotions = [(MNB_classes[index], MNB_proba[0][index]) for index in MNB_top_5_indices]
SVM_top_5_emotions = [(SVM_classes[index], SVM_proba[0][index]) for index in SVM_top_5_indices]

# Output top 5 emotions
for emotion, probability in MNB_top_5_indices:
    print(f"MNB Emotion: {emotion}, Probability: {probability:.4f}")
for emotion, probability in SVM_top_5_indices:
    print(f"SVM Emotion: {emotion}, Probability: {probability:.4f}")