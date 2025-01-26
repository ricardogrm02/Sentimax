from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from combine import analyze_sentiment, text_vectorizer, text_model, emoji_vectorizer, emoji_model, brainrot_vectorizer, brainrot_model, label_encoder

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for React frontend

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    input_text = data.get('text', '')

    # Get sentiment analysis
    final_sentiment, sorted_sentiments = analyze_sentiment(
        input_text,
        text_vectorizer, text_model,
        emoji_vectorizer, emoji_model,
        brainrot_vectorizer, brainrot_model,
        label_encoder
    )

    # Generate a bar graph for sentiments
    sentiments = [s[0] for s in sorted_sentiments]
    probabilities = [s[1] for s in sorted_sentiments]
    plt.bar(sentiments, probabilities, color=['blue', 'green', 'red'])
    plt.xlabel('Sentiments')
    plt.ylabel('Probabilities')
    plt.title('Sentiment Analysis Results')

    # Save the graph as a base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    # Return the results
    return jsonify({
        'final_sentiment': final_sentiment,
        'sorted_sentiments': sorted_sentiments,
        'graph': graph_base64
    })

if __name__ == '__main__':
    app.run(debug=True)
