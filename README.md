# Sentimax

■ Nick TODO
• test image text scraping
• DO NOT TOUCH THE JOY SENTIMENT DATA, IT WILL REDUCE ACCURACY
• Don't use CNN or else `Worry` will become a dominant sentiment
• Combining ML and DL may not work bc CNN and Bi-GRU both dropped the accuracies like crazy
• figure out how to combine scikitlearn, tensorflow, and hugging face together
• determine which models to use together
• look at the f1-score and if the score value is low, consider providing more examples to get that number up
    - the higher the f1-score, the better
• use the following code to check the f1-score and such
    ```
    y_pred = ensemble_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    ```
• add slang, brainrot, memes, and emojis to dataset
• try new combos after figuring out how to bring in LLAMA3 and BERT

■ Ricky TODO
• setup the frontend
• figure out how to allow the user to insert an image to then scrape it
• figure out how to let the user have the choice of typing text in or inserting an image to scrape text off of it
• probably is a library that can convert the image to numbers like a tokenizer