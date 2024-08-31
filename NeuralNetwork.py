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
    def update_params(self, gradients, learning_rate):
        for i in range(1, len(self.params) // 2 + 1):
            self.params['W' + str(i)] -= learning_rate * gradients['dW' + str(i)]
            self.params['b' + str(i)] -= learning_rate * gradients['db' + str(i)]

    def train(self, X, Y, epochs, learning_rate):
        for i in range(epochs):
            A = self.forward(X)
            gradients = self.backward(X, Y)
            self.update_params(gradients, learning_rate)
            if i % 100 == 0:
                loss = np.mean(np.square(Y-A))
                print(f"Epoch is {i}, Loss is {loss:.4f}")

np.random.seed(0)
X = np.random.randn(2, 200)
Y = (X[0] + X[1] > 1).astype(float).reshape(-1, 1)

layer_dims = [2,4,1]
nn = NeuralNetwork(layer_dims)
nn.train(X,Y,epochs = 100, learning_rate = 0.01)

predictions = nn.forward(X)

plt.scatter(X[0], X[1], c=predictions[0], cmap = 'rainbow', edgecolors = 'black')
plt.title("My Neural Network Prediction")
plt.xlabel("F1")
plt.ylabel("F2")
plt.colorbar(label="prediction")
plt.show()