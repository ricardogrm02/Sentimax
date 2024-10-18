import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

file_path = 'final_data_v2 - final_data_v2.csv'
data = pd.read_csv(file_path)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Convert text to numerical data
X = vectorizer.fit_transform(data['content'])
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred, zero_division=1))

# ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■

new_text = ["I love you."]

new_text_transformed = vectorizer.transform(new_text)

# This is to predict a single emotion 
# predicted_sentiment = model.predict(new_text_transformed)

# This prediction gets multiple emotions
proba = model.predict_proba(new_text_transformed)

# This is to print a single emotion 
# print(f"The predicted sentiment for the new text is: {predicted_sentiment[0]}")

# This retrieves the labels for the classes
classes = model.classes_

# This is to sort the emotions into a top 5 
top_5_indices = np.argsort(proba[0])[::-1][:5]

# Get top 5 emotions
top_5_emotions = [(classes[index], proba[0][index]) for index in top_5_indices]

# Output top 5 emotions
for emotion, probability in top_5_emotions:
    print(f"Emotion: {emotion}, Probability: {probability:.4f}")