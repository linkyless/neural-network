import numpy as np
from flask import Flask, request, jsonify, render_template
from network import Layer, NeuralNetwork, sigmoid, softmax

w1 = np.load('w1.npy')
hidden_size = w1.shape[0]

network = NeuralNetwork([
    Layer(784, hidden_size, sigmoid),
    Layer(hidden_size, 10, softmax)
])

network.layers[0].weights = w1
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
    pixels = np.array(data['pixels']).reshape(1, 784)
    predictions = network.forward(pixels)[0]
    return jsonify({'probabilities': predictions.tolist()})
    

if __name__ == '__main__':
    app.run(debug=True)