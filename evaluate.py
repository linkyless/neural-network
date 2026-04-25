import numpy as np
from keras.datasets import mnist
from network import Layer, NeuralNetwork, sigmoid, softmax

(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(10000, 784) / 255.0

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

precision = 0

for i in range(len(x_test)):
    data = x_test[i]
    predictions = network.forward(data)
    if np.argmax(predictions) == y_test[i]:
        precision = precision + 1


print(f"Precision: {precision / 10000 * 100}%")
