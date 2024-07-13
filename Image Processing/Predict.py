from flask import Flask, request, jsonify
import numpy as np
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

with open("C:\Desktop\ML\Machine-Learning-Algorithm-Code-and-Notes-\Image Processing\model\digit_classifier.pkl", 'rb') as f:
    clf = pickle.load(f)

@app.route('/')
def index():
    return jsonify({'message': 'Hello, World!'})

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image_data = request.form['image_data']
        image_data = image_data.strip('[]')
        image_data = [int(x) for x in image_data.split(',')]
        
        prediction = clf.predict([image_data])
        return jsonify({'prediction': int(prediction[0])})

@app.route('/test', methods=['POST'])
def test():
    if request.method == 'POST':
        test = request.form['test']
        return jsonify({'test': test})

if __name__ == '__main__':
    app.run(debug=True)
