from flask import Flask, render_template, request
from input_processing import text_handler, initialize_model, checkValidInput
import os
from io import BytesIO
import base64

app = Flask(__name__, "/static")   # Flask constructor

# A decorator used to tell the application 
# which URL is associated with the function
@app.route('/', methods=['GET', 'POST'])
def hello():
    if request.method == 'GET':
        return render_template("index.html")  # Serving the home page
    if request.method == 'POST' and 'text' in request.form:  # Check for 'text' in form
        text_result = request.form.get("text")  # Get the text input
        status_code = checkValidInput(text_result)
        if status_code == 200:
            print(f'Status Code: {status_code}')
            user_input, bar_graph = initialize_model(text_result)  # Process the text
            return render_template("text_results.html", result=user_input, bar_graph = bar_graph)  # Pass result to template
        elif status_code == 403:
            print(f'Status Code: {status_code}')
            return render_template("index.html")
    if request.method == 'POST' and 'image' in request.files:
        image = request.files['image']
        image_bytes = image.read()
        image_data = BytesIO(image_bytes)
        user_image_path = base64.b64encode(image_data.read()).decode('utf-8')
        output, bar_graph = initialize_model(image_bytes)
        return render_template('image_results.html', user_image = user_image_path, bar_graph = bar_graph)

@app.route('/about.html')
def about():
    return render_template("about.html")



if __name__ == '__main__':
    app.run()
