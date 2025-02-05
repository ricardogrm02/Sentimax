# Sentimax

## TODO List
■ Nick TODO
• try to fix image text scraping off images w/ no text
• figure out how to combine scikitlearn, tensorflow, and hugging face together  
    →bring in LLAMA3, BERT  
• look at f1-score and if score value is low, consider providing more examples to get that number up  
    - the higher the f1-score, the better  
• use the following code to check the f1-score and such  
    ```  
    y_pred = ensemble_model.predict(X_test)  
    print(classification_report(y_test, y_pred))  
    ```  
• make the frontend responsive
    →tailwind, framer motion, react responsive, react intersection observer, radix ui, chakra ui
    →if you want full UI control, go with Tailwind CSS.
    →if you need responsive animations, use Framer Motion
    →if you want JS-based media queries, try React Responsive
    →if you need lazy loading, use React Intersection Observer

## How to Use
1) Make sure the current directory is `Sentimax`
2) Make sure that all necessary libraries are installed (sklearn, flask, etc)
3) Navigate to the `flask_backend.py` file and run it 
4) Change the current directory to `react-vite-app` 
5) Type into the terminal `npm run dev`
6) Open the link that appears in the terminal
7) Website is now usable
8) After having the results pop up, make sure to close it before trying a different set of text