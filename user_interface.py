from flask import Flask, render_template, request
app = Flask(__name__,"/static")   ## Flask constructor 
  
# A decorator used to tell the application 
# which URL is associated function 
@app.route('/')       
def hello(): 
    return render_template("index.html")

    print("Hello")
if __name__=='__main__': 
   app.run() 