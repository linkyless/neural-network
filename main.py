import numpy as np

class Layer:
    def __init__(self, n_prev, n_curr, activation):
        self.weights = np.random.randn(n_curr, n_prev) * 0.01
        self.bias    = np.zeros(n_curr)
        self.activation = activation

    def forward(self, activations: np.ndarray) -> np.ndarray:
        new_act = (self.weights @ activations) + self.bias 
        return self.activation(new_act) # let X = Wb + bias -> sigmoid(X) or softmax(X)
    
class NeuralNetwork:
    def __init__(self, layers: list[Layer]):
        self.layers = layers
    
    def forward(self, data: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            data = layer.forward(data)
        return data


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