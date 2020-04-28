import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import matplotlib.pyplot as plt
import pandas as pd
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = request.form['days']
    int_features =int(int_features)
    prediction = model.forecast(int_features)[0]
    print("***************************************")
    print(prediction)
    prediction = prediction[int_features-1]
    output = round(prediction, 2)
    # output = 2
    nod = int_features
    dateparse = lambda dates: pd.datetime.strptime(dates, '%d-%m-%Y')
    dates = pd.read_csv('dates.csv', parse_dates=['Date'], date_parser=dateparse)
    y = model.forecast(nod)[0]
    x = [dates[i:i+nod] for i in range(0,dates.shape[0],nod)]
    x = pd.DataFrame(x)
    x = dates.loc[0:nod-1]
    x['Price'] = y
    plt.plot(x['Date'],x['Price'])
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.savefig('Graph.png')
    img = "http://127.0.0.1"+ "/Graph.png"
    date = x.loc[nod-1]
    date = np.array(date)

    return render_template('index.html', prediction_text='TCS stock price be Rs. {}'.format(output), date = date )

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)