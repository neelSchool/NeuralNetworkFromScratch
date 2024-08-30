import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

class NeuralNetwork:
    def __init__(self, layer_dims):
        self.params = self.init_params(layer_dims)

    def init_params(self, layer_dims):
        np.random.seed(3)
        params = {}
        L = len(layer_dims)
        for i in range(1, L):
            params['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1] * 0.01)
            params['b' + str(i)] = np.zeros(layer_dims[i], 1)
        return params

    def forward(self, X):
        self.cache = {}
        A = X
        L = len(self.params)
        for i in range(1, L + 1):
            Z = np.dot(self.params['W' + str(i)], A) + self.params['b' + str(i)]
            A = leaky_relu(Z)
            self.cache['A' + str(i)] = A
            self.cache['Z' + str(i)] = Z
        return A

    def backward(self, X, Y):
        m = Y.shape[1]
        L = len(self.params) // 2
        A_last = self.cache['A' + str(L)]
        dZ = A_last - Y
        dW= np.dot(dZ, self.cache['A' + str(L-1)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        gradients = {'dW' + str(L): dW, 'db' + str(L): db}

        for i in reversed(range(1, L)):
            dA = np.dot(self.params['W' + str(i + 1)].T, dZ)
            dZ = dA * leaky_relu_derivative(self.cache['Z' + str(i)])
            dW = np.dot(dZ, self.cache['A' + str(i - 1)].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            gradients['dW' + str(i)] = dW
            gradients['db' + str(i)] = db
        return gradients
