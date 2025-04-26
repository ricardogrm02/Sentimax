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
from DL_brainrot_slang_modifier import get_polarity_boost

# Define paths to save the model, vectorizer, and label encoder
model_path = 'ML_brainrot_slang_ensemble_model.pkl'
vectorizer_path = 'ML_brainrot_slang_vectorizer.pkl'
label_encoder_path = 'ML_brainrot_slang_label_encoder.pkl'

# Function to train and save the model
def train_ensemble():
    data = pd.read_csv("train_brainrot_slang_data.csv")
    data = data.dropna(subset=['content'])
    data = data[data['content'].str.strip() != '']

    vectorizer = TfidfVectorizer(
        max_features=10000,
        token_pattern=r"(?u)(?:\w+|\S)",
        stop_words=None
    )
    X = vectorizer.fit_transform(data['content'])

    le = LabelEncoder()
    y = le.fit_transform(data['sentiment'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    MNB_Model = MultinomialNB()
    LR_Model = LogisticRegression(solver='saga', class_weight='balanced', max_iter=1000)
    C_Model = ComplementNB()
    B_Model = BernoulliNB()
    KNN_Model = KNeighborsClassifier(n_neighbors=3)
    RF_Model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
    SVM_Model = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
    SGDC_Model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, class_weight='balanced', random_state=42)

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

    ensemble_model.fit(X_train, y_train)

    y_pred = ensemble_model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(ensemble_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(le, label_encoder_path)
    print("Model, vectorizer, and label encoder trained and saved successfully.")
    return ensemble_model, vectorizer, le

def read_image(userInput):
    reader = easyocr.Reader(['en'])
    image_path = userInput + '.JPG'
    result = reader.readtext(image_path, detail=0)
    concatenated_text = " ".join(result)
    return concatenated_text

if __name__ == "__main__":
    if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(label_encoder_path):
        user_choice = input("Model exists. Use it (1) or retrain (2)? ")
        if user_choice == '1':
            ensemble_model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            le = joblib.load(label_encoder_path)
            print("Loaded existing model.")
        elif user_choice == '2':
            os.remove(model_path)
            os.remove(vectorizer_path)
            os.remove(label_encoder_path)
            ensemble_model, vectorizer, le = train_ensemble()
        else:
            print("Invalid choice. Exiting.")
            exit()
    else:
        ensemble_model, vectorizer, le = train_ensemble()

    mode = int(input("Select mode:\n1) Insert Text\n2) Insert Image\nChoice: "))
    if mode == 1:
        userInput = input("Input text: ")
    elif mode == 2:
        userInput = read_image(input("Image filename (without extension): "))

    # Vectorize input
    new_text_transformed = vectorizer.transform([userInput])

    # Predict ensemble probabilities
    ensemble_proba = ensemble_model.predict_proba(new_text_transformed)

    # Retrieve class labels
    ensemble_classes = le.inverse_transform(np.arange(len(ensemble_model.classes_)))

    # 🔥 Call the polarity booster from DL_brainrot_modifier
    polarity_boost = get_polarity_boost(userInput)

    # Apply the polarity boosts
    adjusted_probs = []
    for idx, label in enumerate(ensemble_classes):
        if label.lower() in {"joy", "happiness", "relief", "fun", "love", "surprise", "enthusiasm"}:
            boosted = ensemble_proba[0][idx] * polarity_boost.get("positive", 1)
        elif label.lower() in {"neutral", "empty"}:
            boosted = ensemble_proba[0][idx] * polarity_boost.get("neutral", 1)
        elif label.lower() in {"anger", "fear", "sadness", "shame", "disgust", "boredom", "hate", "worry", "disappointment"}:
            boosted = ensemble_proba[0][idx] * polarity_boost.get("negative", 1)
        else:
            boosted = ensemble_proba[0][idx]
        adjusted_probs.append(boosted)

    adjusted_probs = np.array(adjusted_probs)

    # Get top 5 boosted emotions
    top_5_indices = np.argsort(adjusted_probs)[::-1]
    top_5_emotions = [(ensemble_classes[i], adjusted_probs[i]) for i in top_5_indices]

    print("\nPredicted Emotions (with Polarity Boost):")
    for emotion, probability in top_5_emotions:
        print(f"Emotion: {emotion}, Boosted Probability: {probability:.4f}")
