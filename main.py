import numpy as np

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def cost(prediction: float, real: float) -> float:
    return (prediction - real) ** 2

def diff_sigmoid(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))
