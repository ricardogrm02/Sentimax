# Sentimax

## TODO List
■ Nick TODO
• test image text scraping  
• DO NOT TOUCH THE JOY SENTIMENT DATA, IT WILL REDUCE ACCURACY  
• figure out how to combine scikitlearn, tensorflow, and hugging face together  
• determine which models to use together  
• avoid svm, randomforest, sdgc for taking too long  
• look at the f1-score and if the score value is low, consider providing more examples to get that number up  
    - the higher the f1-score, the better  
• use the following code to check the f1-score and such  
    ```  
    y_pred = ensemble_model.predict(X_test)  
    print(classification_report(y_test, y_pred))  
    ```  
• add slang, brainrot, memes, and emojis to dataset  
• trying to use simpler classifiers   
• try new combos after figuring out how to bring in LLAMA3 and BERT  

■ Ricky TODO  
• setup the frontend  
• figure out how to allow the user to insert an image to then scrape it  
• figure out how to let the user have the choice of typing text in or inserting an image to scrape text off of it  
• probably is a library that can convert the image to numbers like a tokenizer  

## How to Use
1) Make sure the current directory is `Sentimax`
2) Make sure that all necessary libraries are installed (sklearn, flask, etc)
3) Navigate to the `user_interface.py` file and run it 
4) Wait a bit for the project to load
5) A local website link will be made in the terminal, ctrl+left click it to open the project website
6) User can use an image with text or just text to check for sentiment (must click submit, pressing enter won't work)
7) After the results display, you can return to main page via "Try Model" button at the top of the page
