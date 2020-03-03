import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from networks.network import NeuralNetwork

input_data = np.array([1, 2, 3, 4])

nn = NeuralNetwork([np.array([[2, 1, 0, 12],
                              [4, -2, 5, 3]]),
                    np.array([[3, -2]])],
                   [np.array([1, 5]),
                    np.array([2])],
                   [NeuralNetwork.sigmoid, NeuralNetwork.linear])

nn.feed_forward(input_data)

# squares simple

df = pd.read_csv('data/regression/square-simple-training.csv')
xs = df.iloc[:, 1].values
ys = df.iloc[:, 2].values

plt.scatter(xs, ys)

network = NeuralNetwork([np.array([[100],
                                   [100],
                                   [100],
                                   [100],
                                   [100]]),
                         np.array([[100, 100, 100, 100, 100]])],
                        [np.array([0, 0, 0, 0, 0]),
                         np.array([0])],
                        [NeuralNetwork.sigmoid, NeuralNetwork.linear])


network.feed_forward(xs[0:1])

ys_new = np.zeros(100)

for i in range(xs.size):
    ys_new[i] = network.feed_forward(xs[i:i + 1])[0]

plt.figure()
plt.scatter(xs, ys)
plt.scatter(xs, ys_new)