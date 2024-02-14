import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
from sklearn.preprocessing import StandardScaler
import joblib
import pickle


app = Flask(__name__)
model = joblib.load('predict_emp.sav')

scaler = joblib.load('predict_emp_scaler.pkl')


@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = np.array(int_features).reshape(1,-1)
    scaled_features = scaler.transform(final_features)
    prediction = model.predict(scaled_features)
    print(int(prediction[0]))

    if int(prediction[0] < 0):
        prediction[0] = 1

    #output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="Number of approximate employees should be : {}".format(int(prediction[0])))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = int(prediction[0])
    if output < 0:
        output = 1
    return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)