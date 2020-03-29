library(MIOwAD)
library(dplyr)
library(ggplot2)

# X <- matrix(1:12, 3, 4)
# y <- matrix(c(-5, 10, -12), 3)
#
# network <- neural_network(4) +
#   hidden_layer(10, "sigmoid") +
#   output_layer(1, "linear")
#
# network %>%
#   randomize_weights() %>%
#   train_network(X, y, num_epochs = 1000)

dat <- read.csv("data/classification/rings3-regular-training.csv")
X <- scale(as.matrix(dat)[, 1:2, drop = FALSE])
y <- as.matrix(dat)[, 3, drop = FALSE]
y <- cbind(ifelse(y == 0, 1, 0),
           ifelse(y == 1, 1, 0),
           ifelse(y == 2, 1, 0))

net <- neural_network(2) +
  hidden_layer(20, "sigmoid") +
  hidden_layer(30, "sigmoid") +
  hidden_layer(20, "sigmoid") +
  output_layer(3, "softmax")


set.seed(123)
net %>%
  randomize_weights() %>%
  train_network_momentum(X, y, num_epochs = 1000, eta = 3e-3, gamma = 0.8) -> trained

trained %>%
  feed_network(X) %>%
  select_max() -> fit

ggplot(data = data.frame(x = X[, 1], y = X[, 2], c_fit = fit[, 1],
                         c_real = dat[, 3], fit = (fit[, 1] == dat[, 3])),
       aes(x = x, y = y, color = fit, shape = as.factor(c_real))) +
  geom_point()


