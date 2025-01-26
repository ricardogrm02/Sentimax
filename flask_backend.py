from flask import Flask, request, jsonify
import joblib
from combine import (
    analyze_sentiment,
    text_vectorizer, text_model,
    emoji_vectorizer, emoji_model,
    brainrot_vectorizer, brainrot_model,
    label_encoder
)

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get input text from the request
        data = request.json
        input_text = data.get('text', '')

        # Analyze sentiment using the combine.py pipeline
        final_sentiment, sorted_sentiments = analyze_sentiment(
            input_text,
            text_vectorizer, text_model,
            emoji_vectorizer, emoji_model,
            brainrot_vectorizer, brainrot_model,
            label_encoder
        )

        # Format and return the results
        return jsonify({
            'final_sentiment': final_sentiment,
            'sorted_sentiments': [
                {'sentiment': sentiment, 'probability': prob}
                for sentiment, prob in sorted_sentiments
            ]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)