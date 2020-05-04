import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import math
import os
app = Flask(__name__)

@app.route('/')
def home():
    if os.path.exists("templates/visualize.html"):
        os.remove("templates/visualize.html")
    return render_template('index.html', l=0)

@app.route('/visualize',methods=['POST'])
def visualize():
    return render_template('visualize.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = request.form['days']
    if(request.form['company'] == 'TCS'):
        model = pickle.load(open('model_TCS.pkl', 'rb'))
        graph_title = 'TCS'
    elif(request.form['company'] == 'INFY'):
        model = pickle.load(open('model_INFY.pkl','rb'))
        graph_title = 'Infosys'
    elif(request.form['company'] == 'Cipla'):
        model = pickle.load(open('model_Cipla.pkl','rb'))
        graph_title = 'Cipla'
    elif(request.form['company'] == 'Lupin'):
        model = pickle.load(open('model_Lupin.pkl','rb'))
        graph_title = 'Lupin Pharmaceuticals'
    elif(request.form['company'] == 'TM'):
        model = pickle.load(open('model_TM.pkl','rb'))
        graph_title = 'Tata Motors'
    elif(request.form['company'] == 'FM'):
        model = pickle.load(open('model_FM.pkl','rb'))
        graph_title = 'Force Motors'
    else:
        return render_template('index.html', error='Please select a company!!!',l=0)
    int_features =int(int_features)
    if(int_features < 1 or int_features > 299):
        return render_template('index.html', error='Oops!! Value too high or too low. Try entering a value greater than 0 and less than 300',l=0)
    y = np.exp(model.forecast(int_features)[0])
    y = np.round(y,2)
    if int_features == 1:
        output = y[0]
        return render_template('index.html', prediction_text=graph_title+' stock price is expected to be Rs. {}'.format(output), l=0)
    prediction = y[int_features-1]
    nod = int_features
    dateparse = lambda dates: pd.datetime.strptime(dates, '%d-%m-%Y')
    dates = pd.read_csv('future_dates.csv', parse_dates=['Date'], date_parser=dateparse)
    x = [dates[i:i+nod] for i in range(0,dates.shape[0],nod)]
    x = pd.DataFrame(x)
    x = dates.loc[0:nod-1]
    x['Price'] = y
    fig = px.line(x, x='Date', y='Price', title = graph_title + ' stock price prediction')
    if os.path.exists("templates/visualize.html"):
        os.remove("templates/visualize.html")
    fig.write_html("templates/visualize.html")
    date = x.loc[nod-1]
    date = np.array(date)
    df = x
    s = pd.Series(i for i in range(1,nod+1))
    df.set_index(s)
    
    return render_template('index.html', prediction_text=graph_title + ' stock price is expected to be Rs. {}'.format(prediction), tables=[df.to_html(classes='data')], titles=df.columns.values, vs_text = "Click here to visualize", l=1)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    y = prediction[0]
    return jsonify(y)

if __name__ == "__main__":
    app.run(debug=True)
