from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math

model1 = pickle.load(open('model10.pkl', 'rb'))
model2, scaler = pickle.load(open('model11.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def man():
    return render_template('homepage.html')


@app.route('/predict', methods=['POST'])
def main():
    
   
    int_features = [x for x in request.form.values()]
  
    for i in range(len(int_features)):
        int_features[i] = float(int_features[i].replace(',',''))
    
    final = [np.array(int_features)]
    prediction = model1.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    output = float(output)
    output*=100
    diff = 100 - output
    
    amount = model2.predict(np.array(int_features).reshape(1,10))


    amount = scaler.inverse_transform(amount.reshape(-1,1)).flatten()
      
    amount = math.trunc(amount[0])
  
    output = math.trunc(output)
    
    amount = model2.predict(np.array(int_features).reshape(1,10))


    amount = scaler.inverse_transform(amount.reshape(-1,1)).flatten()
    
    amount = math.trunc(amount[0])
  
    return render_template('homepage.html', pred='Our models suggest that there is a {}% chance that you will file an insurance claim. Estimated amount is ${}.'.format(output, amount))

 

if __name__ == '__main__':
    main()
