from flask import Flask, request, render_template, jsonify
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import SVC

app = Flask(__name__)

# Load your data and model
data = []
label = []

with open('cleaned_data.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if row:
            label.append(int(row[0]))
            data.append([int(x) for x in row[1:]])

data = np.array(data)
label = np.array(label)

# Shuffle data
permutation = np.random.permutation(len(data))
data = data[permutation]
label = label[permutation]

# Define your machine learning model
clf = SVC(gamma=0.001)
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.20, random_state=42)
clf.fit(X_train, y_train)

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
        accuracy = clf.score(X_test, y_test)
        print(f'Accuracy: {accuracy}')
        return jsonify({'prediction': int(prediction[0]), 'accuracy': accuracy})
    
@app.route('/test', methods=['POST'])
def test():
    if request.method=='POST':
        test= request.form['test']
        return jsonify({'test': test})
    

if __name__ == '__main__':
    app.run(debug=True)
