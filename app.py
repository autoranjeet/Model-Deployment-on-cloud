from flask import Flask,request,render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
pickle_in = open('rf1.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    salary = int(request.form['salary'])
    age = int(request.form['age'])
    balance = int(request.form['balance'])
    previous = int(request.form['previous'])
    duration = int(request.form['duration'])
    campaign = int(request.form['campaign'])
    blue_collor = int(request.form['blue-collor'])
    entreprenuer = int(request.form['entrepreneur'])
    housemaid = int(request.form['housemaid'])
    management = int(request.form['management'])
    retired = int(request.form['retired'])
    self_employed = int(request.form['self-employed'])
    services = int(request.form['services'])
    student = int(request.form['student'])
    technician = int(request.form['technician'])
    unemployed = int(request.form['unemployed'])
    job_unknown = int(request.form['unknown'])
    married = int(request.form['married'])
    single = int(request.form['single'])
    secondary = int(request.form['secondary'])
    tertiary = int(request.form['tertiary'])
    edu_unknown = int(request.form['unknown'])
    default_yes = int(request.form['d_yes'])
    housing_yes = int(request.form['h_yes'])
    data = [[salary,age,balance,previous,duration,campaign,blue_collor,entreprenuer,housemaid,management,retired,self_employed,services,
    student,technician,unemployed,job_unknown,married,single,secondary,tertiary,edu_unknown,default_yes,housing_yes]]
    
    prediction = classifier.predict(data)

    if prediction[0] == 1:
        client_response = 'Yes'
    else:
        client_response = 'No'

    return render_template('index.html', prediction_text='The customer will responsed as {}'.format(client_response))

if __name__ == "__main__":
    app.run(debug = True)