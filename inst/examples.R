library(MIOwAD)
library(dplyr)
library(ggplot2)

X <- matrix(1:12, 3, 4)
y <- matrix(c(-5, 10, -12), 3)

network <- neural_network(4) +
  hidden_layer(10, "sigmoid") +
  output_layer(1, "linear")

network %>%
  randomize_weights() %>%
  train_network(X, y, num_epochs = 1000)

dat <- read.csv("data/regression/square-small-training.csv")
X <- scale(as.matrix(dat)[, 2, drop = FALSE])
y <- scale(as.matrix(dat)[, 3, drop = FALSE])

net <- neural_network(1) +
  hidden_layer(10, "sigmoid") +
  hidden_layer(10, "sigmoid") +
  output_layer(1, "linear")

net %>%
  randomize_weights() %>%
  train_network(X, y, num_epochs = 1000, eta = 1e-4) -> trained

trained %>%
  feed_network(X) -> fit

ggplot(data = data.frame(x = X[, 1], y_fit = fit[, 1], y_real = y[, 1]), aes(x = x)) +
  geom_point(mapping = aes(y = y_real)) +
  geom_point(mapping = aes(y = y_fit), color = "red")
