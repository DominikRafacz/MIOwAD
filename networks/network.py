import numpy as np
import math


class NeuralNetwork:

    def __init__(self, weights, biases, activations):
        self.weights = weights
        self.biases = biases
        activations.insert(0, NeuralNetwork.linear)
        self.activations = activations

    @staticmethod
    def initialize_randomly(dims, activations):
        weights = [np.empty(0)] * (len(dims) - 1)
        biases = [np.empty(0)] * (len(dims) - 1)
        for i in range(len(dims) - 1):
            weights[i] = np.random.rand(dims[i + 1], dims[i])
            biases[i] = np.random.rand(dims[i + 1])
        return NeuralNetwork(weights, biases, activations)

    # def feed_forward(self, input_data):
    #     cur = input_data
    #     for i in range(len(self.weights)):
    #         cur = self.activations[i](cur.dot(self.weights[i].transpose()) + self.biases[i])
    #     return cur

    def train(self, X, y, batch_size=None, eta=1e-3):
        n = X.shape[0]
        if batch_size is None:
            batch_size = n
        stop_condition = False
        m = math.ceil(n / batch_size)
        while not stop_condition:
            inds = np.floor(np.random.permutation(n) / batch_size)
            # X_batches = [np.empty(0)] * m
            # y_batches = [np.empty(0)] * m
            for b in range(m):
                # X_batches[i] = X[inds == i, :]
                # y_batches[i] = y[inds == i, :]
                X_batch = X[inds == b, :]
                y_batch = y[inds == b, :]

                delta = [np.zeros(weight.shape) for weight in weights]
                for i in range(X_batch.shape[0]):
                    y_act = y_batch[i, :]
                    a = [np.empty(0)] * (len(weights) + 1)
                    a[0] = X_batch[i, :]
                    for k in range(len(weights)):
                        a[k + 1] = activations[k](a[k]).dot(weights[k].transpose() + biases[k])
                    y_hat = activations[len(weights)](a[len(weights)])

                    e = [np.empty(0)] * (len(weights) + 1)
                    e[len(weights)] = np.multiply(deriv(activations[len(weights)])(a[len(weights)]), y_hat - y_act)
                    for k in range(len(weights), 1, -1):
                        e[k - 1] = np.multiply(deriv(activations[k - 1])(a[k - 1]), e[k].dot(weights[k - 1]))

                    for k in range(len(weights)):
                        delta[k] = delta[k] + e[k + 1].reshape(-1, 1).dot(activations[k + 1](a[k]))


    @staticmethod
    def sigmoid(arr):
        return np.vectorize(lambda x: math.exp(x) / (1 + math.exp(x)))(arr)

    @staticmethod
    def linear(arr):
        return arr


def deriv(fun):
    if fun == NeuralNetwork.sigmoid:
        return lambda x: NeuralNetwork.sigmoid(x) * (1 - NeuralNetwork.sigmoid(x))
    elif fun == NeuralNetwork.linear:
        return lambda x: np.ones(len(x))
