import numpy as np
import math


class NeuralNetwork:

    def __init__(self, weights, biases, activations):
        self.weights = weights
        self.biases = biases
        self.activations = activations

    def feed_forward(self, input_data):
        cur = input_data
        for i in range(len(self.weights)):
            cur = self.activations[i](cur.dot(self.weights[i].transpose()) + self.biases[i])
        return cur

    @staticmethod
    def sigmoid(arr):
        return np.vectorize(lambda x: math.exp(x) / (1 + math.exp(x)))(arr)

    @staticmethod
    def linear(arr):
        return arr

