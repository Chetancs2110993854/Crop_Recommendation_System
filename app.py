from urllib.request import Request
from flask import Flask, render_template, request , jsonify
import pickle
import numpy as np 

model = pickle.load(open('iril.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)] '''
    
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g'] 

    arr = np.array([[data1, data2, data3, data4,data5,data6,data7]])
    str1=""
    prediction = model.predict(arr)
    for i in prediction:
        str1 +=str(i)
    '''
    str1=""
    prediction = model.predict(final_features)
    for i in prediction:
        str1 +=str(i) '''

    #output = str(pred)
    #return render_template('after.html', data=pred)
    return render_template('index.html',prediction_text ='Crop most suitable is {}'.format(str1))

if __name__ == "__main__":
    app.run(debug=True) 
    #if ou















