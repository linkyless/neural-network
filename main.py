import numpy as np

class Layer:
    def __init__(self, n_prev, n_curr, activation):
        self.weights    = np.random.randn(n_curr, n_prev) * 0.01
        self.bias       = np.zeros(n_curr)
        self.activation = activation

    def forward(self, activations: np.ndarray) -> np.ndarray:
        new_act = (self.weights @ activations) + self.bias 
        self.Z = new_act # Z: pre-activation vector
        self.S = self.activation(new_act) # S: activation vector
        return self.S
    
class NeuralNetwork:
    def __init__(self, layers: list[Layer]):
        self.layers = layers
    
    def forward(self, data: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            data = layer.forward(data)
        return data
    
    def backprop(self, input: np.ndarray, predictions: np.ndarray, correct_index: int, learning_rate: float):
        # Output Layer - Weights and Biases

        # dC/dZ2 = predictions - y, y = one-hot answer vector
        # predictions = S2
        y = np.zeros(10) # prob of digits 0, 1, 2, ..., 9
        y[correct_index] = 1
        dZ2 = predictions - y

        # dC/dW2 = dZ2 * dZ2/dW2 = dZ2 * layers[-1].S -> (10,) * (128,) = (10, 128)
        dW2 = np.outer(dZ2, self.layers[-1].S)

        # dC/db2 = dZ2
        db2 = dZ2

        # Hidden Layer - Weights and Biases

        # dC/dS1 = dZ2 * dZ2/dS1 = dZ2 * W2 -> (10,) * (10, 128) -> (10,) * (128, 10) = (128,)
        dS1 = dZ2 @ self.layers[-1].weights.T # transpose

        # dC/dZ1 = dS1 * dS1/dZ1 = dS1 * diff_sigmoid(Z1) -> (128,) * (128,) = (128,)
        dZ1 = dS1 * diff_sigmoid(self.layers[-2].Z)
        
        # dC/dW1 = dZ1 * dZ1/dW1 = dZ1 * input -> (128,) * (784,) = (128, 784)
        dW1 = np.outer(dZ1, input)

        # dC/db1 = dZ1
        db1 = dZ1

        # Update weights and biases

        self.layers[-1].weights -= learning_rate * dW2
        self.layers[-1].bias    -= learning_rate * db2
        self.layers[-2].weights -= learning_rate * dW1
        self.layers[-2].bias    -= learning_rate * db1



def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def cost(prediction: float, real: float) -> float:
    return (prediction - real) ** 2


def diff_sigmoid(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(arr: np.ndarray) -> np.ndarray:
    exp_arr = np.exp(arr)
    exp_sum = np.sum(exp_arr)
    new_arr = exp_arr / exp_sum
    return new_arr

def cross_entropy(predictions: np.ndarray, correct_index: int) -> float:
    loss = -np.log(predictions[correct_index])
    return loss

# pred = np.random.randn(784)
# A = Layer(784, 128, sigmoid)
# Exit = Layer(128, 10, softmax)
# network = NeuralNetwork([A, Exit])
# print(network.forward(pred))
# print(np.sum(network.forward(pred)))