import joblib
from flask import Flask, request, jsonify

#load the train model
model = joblib.load('I:\Common\Ganesh\mlops_github\model\model.pkl')

#initialize the app
app = Flask(__name__)

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.get_json(force=True)
    features = data['features']
    prediction = model.predict([features])
    return jsonify({'prediction' : int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)