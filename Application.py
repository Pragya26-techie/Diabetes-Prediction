from flask import Flask, request,app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__,template_folder =r"C:\Diabetes Prediction\templates")

scaler = pickle.load(open(r"C:\Diabetes Prediction\Models\standardScalar.pkl","rb"))
model = pickle.load(open(r"C:\Diabetes Prediction\Models\modelForPrediction.pkl","rb"))

## Route for homepage

@app.route('/')
def index():
    #print("Current working directory:", os.getcwd())
    #templates_path = os.path.join(os.getcwd(), 'templates')
    #print("Looking for templates in:", templates_path)
    #print("Template files:", os.listdir(templates_path))
    return render_template('index.html')

## Route for Single data point prediction
@app.route('/predictdata',methods = ['GET','POST'])
def predict_datapoint():
    result = ""

    if request.method == 'POST':
        Pregnancies = int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data = scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict = model.predict(new_data)

        if predict[0] == 1:
            result = 'Diabetic'
        else:
            result = 'Non-Diabetic'
        
        return render_template('single_prediction.html',result = result)
    else:
        return render_template('home.html')
    
if __name__ == "__main__":
        app.run(host = "0.0.0.0",debug = True)

    
