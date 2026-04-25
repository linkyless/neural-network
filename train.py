import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from network import Layer, NeuralNetwork, sigmoid, softmax, cross_entropy
from scipy.ndimage import shift, rotate, zoom

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test  = x_test.reshape(10000, 784) / 255.0

number_of_images = 60000
batch_size       = 32

learning_rate    = 0.1
epochs           = 20

hidden_size      = 256    # hidden layer number of neurons

hidden_layer     = Layer(784, hidden_size, sigmoid)
output_layer     = Layer(hidden_size, 10, softmax)

network = NeuralNetwork([hidden_layer, output_layer])
costs = []

for epoch in range(epochs):
    cost_epoch = 0
    n_batches = 0

    indexes = np.random.permutation(number_of_images)

    for start in range(0, number_of_images, batch_size):
        batch_indexes = indexes[start:start + batch_size]
        X = x_train[batch_indexes]
        Y = y_train[batch_indexes]
        current_batch_size = X.shape[0]

        augmented = np.empty_like(X)
        for j in range(current_batch_size):
            angle = np.random.uniform(-15, 15)
            dx = np.random.randint(-3, 4)
            dy = np.random.randint(-3, 4)
            img = rotate(X[j], angle, reshape=False)
            img = shift(img, [dy, dx])
            augmented[j] = img

        X_flat = augmented.reshape(current_batch_size, 784)
        predictions = network.forward(X_flat)
        network.backprop(X_flat, predictions, Y, learning_rate)
        cost_epoch += cross_entropy(predictions, Y)
        n_batches += 1

    costs.append(cost_epoch / n_batches)
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

