import numpy as np

class Layer:
    def __init__(self, n_prev, n_curr, activation):
        self.weights = np.random.randn(n_curr, n_prev) * 0.01
        self.bias    = np.zeros(n_curr)
        self.activation = activation

    def forward(self, activations: np.ndarray) -> np.ndarray:
        new_act = (self.weights @ activations) + self.bias 
        return self.activation(new_act) # let X = Wb + bias -> sigmoid(X) or softmax(X)
    


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

activations = np.array([0.4, 0.3, -0.121])
deep = Layer(activations.shape[0], 2, sigmoid)
new_activations = deep.forward(activations)
exit = Layer(new_activations.shape[0], 10, softmax)
print(exit.forward(new_activations))
print(np.sum(exit.forward(new_activations)))