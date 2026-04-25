import numpy as np


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def diff_sigmoid(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(arr: np.ndarray) -> np.ndarray:
    exp_arr = np.exp(arr)
    exp_sum = np.sum(exp_arr, axis=-1, keepdims=True)
    new_arr = exp_arr / exp_sum
    return new_arr


def cross_entropy(predictions, correct_indices):
    n = len(correct_indices)
    return -np.mean(np.log(predictions[np.arange(n), correct_indices] + 1e-9))


class Layer:
    def __init__(self, n_prev, n_curr, activation):
        self.weights    = np.random.randn(n_curr, n_prev) * np.sqrt(2 / n_prev)
        self.bias       = np.zeros(n_curr)
        self.activation = activation

    def forward(self, activations: np.ndarray) -> np.ndarray:
        new_act = (activations @ self.weights.T) + self.bias
        self.Z = new_act # Z: pre-activation matrix 
        self.S = self.activation(new_act) # S: activation matrix
        return self.S
    
class NeuralNetwork:
    def __init__(self, layers: list[Layer]):
        self.layers = layers
    
    def forward(self, data: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            data = layer.forward(data)
        return data
    
    def backprop(self, input: np.ndarray, predictions: np.ndarray, correct_indexes: int, learning_rate: float):
        # Output Layer - Weights and Biases

        # dC/dZ2 = predictions - y, y = one-hot answer matrix
        # predictions = S2
        
        batch_size = input.shape[0]
        y = np.zeros_like(predictions) # prob of digits 0, 1, 2, ..., 9 in 32 rows
        y[np.arange(batch_size), correct_indexes] = 1
        dZ2 = predictions - y

        S1 = self.layers[-2].S

        # dC/dW2 = dZ2 * dZ2/dW2 / 32 = dZ2 * layers[-2].S / 32
        dW2 = dZ2.T @ S1 / batch_size

        # dC/db2 = dZ2 / 32
        db2 = np.mean(dZ2, axis=0)

        # Hidden Layer - Weights and Biases

        # dC/dS1 = dZ2 * dZ2/dS1 = dZ2 * W2 -> (10,) * (10, 128)
        dS1 = dZ2 @ self.layers[-1].weights

        # dC/dZ1 = dS1 * dS1/dZ1 = dS1 * diff_sigmoid(Z1)
        dZ1 = dS1 * diff_sigmoid(self.layers[-2].Z)

        # dC/dW1 = dZ1 * dZ1/dW1 = dZ1 * input
        dW1 = dZ1.T @ input / batch_size

        # dC/db1 = dZ1
        db1 = np.mean(dZ1, axis=0)

        # Update weights and biases

        self.layers[-1].weights -= learning_rate * dW2
        self.layers[-1].bias    -= learning_rate * db2
        self.layers[-2].weights -= learning_rate * dW1
        self.layers[-2].bias    -= learning_rate * db1
