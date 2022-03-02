# Antes de realizar esta parte hay que entrar en la
# terminal de nuevo y escribir dentro del directorio:
# poetry add flask

import pickle

from flask import Flask, jsonify, request

from converted_predict_service import predict_single

app = Flask('converted-predict')

with open('models/converted-model.pck', 'rb') as f:
    dv, model = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    converted, prediction = predict_single(customer, dv, model)

    result = {
        'converted': bool(converted),
        'converted_probability': float(prediction),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
    