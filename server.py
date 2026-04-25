import numpy as np
from flask import Flask, request, jsonify, render_template
from network import Layer, NeuralNetwork, sigmoid, softmax

network = NeuralNetwork([
    Layer(784, 128, sigmoid), # Hidden Layer
    Layer(128, 10, softmax)   # Output Layer
])

network.layers[0].weights = np.load('w1.npy')
network.layers[0].bias    = np.load('b1.npy')
network.layers[1].weights = np.load('w2.npy')
network.layers[1].bias    = np.load('b2.npy')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    pixels = np.array(data['pixels'])
    predictions = network.forward(pixels)
    predicted_digit = np.argmax(predictions)
    return jsonify({'digit': int(predicted_digit)})
    

if __name__ == '__main__':
    app.run(debug=True)