from flask import Flask, request, jsonify
from flask_cors import CORS

# Importing from combine.py
from combine import (
    analyze_sentiment,
    text_vectorizer, text_model,
    emoji_vectorizer, emoji_model,
    brainrot_vectorizer, brainrot_model,
    label_encoder
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/analyze-sentiment', methods=['POST'])
def analyze_text_sentiment():
    try:
        # Get the JSON data from the frontend
        data = request.json
        input_text = data.get("text", "")

        if not input_text:
            return jsonify({"error": "No text provided"}), 400

        # Call the sentiment analysis function from combine.py
        final_sentiment, sorted_sentiments = analyze_sentiment(
            input_text,
            text_vectorizer, text_model,
            emoji_vectorizer, emoji_model,
            brainrot_vectorizer, brainrot_model,
            label_encoder
        )

        # Return the results as JSON
        return jsonify({
            "final_sentiment": final_sentiment,
            "sorted_sentiments": [
                {"sentiment": sentiment, "probability": prob}
                for sentiment, prob in sorted_sentiments
            ]
        }), 200
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
