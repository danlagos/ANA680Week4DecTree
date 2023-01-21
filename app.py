from flask import Flask, render_template, request
import numpy as np
#import pickle
import joblib

app = Flask(__name__)

filename = 'dectree.pkl'

model = joblib.load(filename)

@app.route('/')

def index(): 
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    alcohol = request.form['alcohol']
    density = request.form['density']
    volatile_acidity = request.form['volatile_acidity']
    chlorides = request.form['chlorides']
          
    pred = model.predict(np.array([[alcohol, density, volatile_acidity, chlorides]], dtype=float))
    print(pred)
    return render_template('index.html', predict=str(pred))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)