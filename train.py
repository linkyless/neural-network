import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from network import Layer, NeuralNetwork, sigmoid, softmax, cross_entropy
from scipy.ndimage import shift, rotate, zoom

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test  = x_test.reshape(10000, 784) / 255.0

learning_rate = 0.01
epochs = 10

hidden_layer = Layer(784, 128, sigmoid)
output_layer = Layer(128, 10, softmax)

network = NeuralNetwork([hidden_layer, output_layer])
costs = []

for epoch in range(epochs):
    cost_epoch = 0
    for i in range(len(x_train)):
        image = x_train[i]
    
        # augmentation
        angle = np.random.uniform(-15, 15)
        dx = np.random.randint(-3, 4)
        dy = np.random.randint(-3, 4)
        image = rotate(image, angle, reshape=False)
        image = shift(image, [dy, dx])
        
        # flat and forward
        data = image.reshape(784)
        predictions = network.forward(data)
        network.backprop(data, predictions, y_train[i], learning_rate)
        cost_epoch += cross_entropy(predictions, y_train[i])
    
    costs.append(cost_epoch / len(x_train))
    print(f"epoch {epoch + 1} completed. cost: {costs[-1]:.4f}")

np.save('w1.npy', network.layers[0].weights)
np.save('b1.npy', network.layers[0].bias)
np.save('w2.npy', network.layers[1].weights)
np.save('b2.npy', network.layers[1].bias)

plt.plot(costs)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.show()

fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i].reshape(28, 28), cmap='gray')
    pred = np.argmax(network.forward(x_test[i]))
    ax.set_title(f"real: {y_test[i]} pred: {pred}")
    ax.axis('off')


plt.show()

