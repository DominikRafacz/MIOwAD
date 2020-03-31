# loading necessary libraries
library(MIOwAD)     # this laboratories
library(magrittr)   # pipelines
library(ggplot2)    # plotting

# read datasets
dat_easy <- read.csv("data/classification/easy-training.csv")
dat_rings <- read.csv("data/classification/rings3-regular-training.csv")
dat_xor <- read.csv("data/classification/xor3-training.csv")

# extract X's and y's
X_easy <- scale(as.matrix(dat_easy)[, 1:2, drop = FALSE])
y_easy <- as.matrix(dat_easy)[, 3, drop = FALSE]
y_easy_ohe <- cbind(ifelse(y_easy == 0, 1, 0),
                    ifelse(y_easy == 1, 1, 0))

X_rings <- scale(as.matrix(dat_rings)[, 1:2, drop = FALSE])
y_rings <- as.matrix(dat_rings)[, 3, drop = FALSE]
y_rings_ohe <- cbind(ifelse(y_rings == 0, 1, 0),
                     ifelse(y_rings == 1, 1, 0),
                     ifelse(y_rings == 2, 1, 0))

X_xor <- scale(as.matrix(dat_xor)[, 1:2, drop = FALSE])
y_xor <- as.matrix(dat_xor)[, 3, drop = FALSE]
y_xor_ohe <- cbind(ifelse(y_xor == 0, 1, 0),
                   ifelse(y_xor == 1, 1, 0))

### easy

# building networks - two for each dataset
net_easy_lin <- neural_network(2) +   # input size = 2
  hidden_layer(2, "sigmoid") +       # sigmoid layer
  output_layer(2, "linear")           # output size = 3, softmax

net_easy_soft <- neural_network(2) +   # input size = 2
  hidden_layer(2, "sigmoid") +       # sigmoid layer
  output_layer(2, "softmax")          # output size = 3, softmax

# training
set.seed(123)
net_easy_lin %>%
  randomize_weights() %>%
  train_network_sgd(X_easy, y_easy_ohe, num_epochs = 1000,
                    eta = 1e-4, loss = "mse") -> trained_easy_lin_mse

net_easy_soft %>%
  randomize_weights() %>%
  train_network_sgd(X_easy, y_easy_ohe, num_epochs = 1000,
                    eta = 1e-4, loss = "mse") -> trained_easy_soft_mse

net_easy_soft %>%
  randomize_weights() %>%
  train_network_sgd(X_easy, y_easy_ohe, num_epochs = 1000,
                    eta = 1e-4, loss = "crossentropy") -> trained_easy_soft_ce

# plotting pace of convergence
ggplot(data.frame(val = unlist(c(trained_easy_lin_mse$training_history,
                          trained_easy_soft_mse$training_history,
                          trained_easy_soft_ce$training_history)),
                  activation = c(rep("linear", 1000), rep("softmax", 2000)),
                  loss = rep(c("mse", "mse", "crossentropy"), each = 1000),
                  epoch = rep(1:1000, 3)),
       aes(x = epoch, y = val, color = activation, linetype = loss)) +
  geom_line() +
  ggtitle("comparison of different loss and activation function", "for dataset 'easy'")

ggsave("lab/lab4/easy_loss.png", device = png())
dev.off()

# fitting
trained_easy_soft_ce %>%
  feed_network(X_easy) %>%
  select_max() -> fit_easy_soft_ce

trained_easy_lin_mse %>%
  feed_network(X_easy) %>%
  select_max() -> fit_easy_lin_mse

trained_easy_soft_mse %>%
  feed_network(X_easy) %>%
  select_max() -> fit_easy_soft_mse

# creating grid to display decision boundaries
X_grid <- data.frame(x = rep(seq(-2, 2, length.out = 100), each = 100),
                     y = rep(seq(-2, 2, length.out = 100), times = 100))

trained_easy_soft_ce %>%
  feed_network(as.matrix(X_grid)) %>%
  select_max() -> fit_grid

# plotting decision boundaries
ggplot() +
  geom_point(data = cbind(X_grid, fitted_value = as.factor(fit_grid[, 1])),
             mapping = aes(x = x, y = y, color = fitted_value)) +
  geom_point(data = data.frame(x = X_easy[, 1], y = X_easy[, 2],
                               true_value = factor(dat_easy[, 3], labels = c(0, 1))),
             mapping = aes(x = x, y = y, shape = true_value)) +
  ggtitle("network decision boundaries and true values",
          "for dataset 'easy' and model with softmax and crossentropy")

ggsave("lab/lab4/easy_bound.png", device = png())
dev.off()

### rings 3

# building networks - two for each dataset
net_rings_lin <- neural_network(2) +   # input size = 2
  hidden_layer(20, "sigmoid") +
  hidden_layer(20, "sigmoid") +
  hidden_layer(20, "sigmoid") +
  output_layer(3, "linear")           # output size = 3, softmax

net_rings_soft <- neural_network(2) +   # input size = 2
  hidden_layer(20, "sigmoid") +
  hidden_layer(20, "sigmoid") +
  hidden_layer(20, "sigmoid") +
  output_layer(3, "softmax")          # output size = 3, softmax

# training
set.seed(123)
net_rings_lin %>%
  randomize_weights() %>%
  train_network_momentum(X_rings, y_rings_ohe, num_epochs = 1000,
                    eta = 1e-4, loss = "mse") -> trained_rings_lin_mse

net_rings_soft %>%
  randomize_weights() %>%
  train_network_momentum(X_rings, y_rings_ohe, num_epochs = 1000,
                    eta = 3e-3, loss = "mse") -> trained_rings_soft_mse

net_rings_soft %>%
  randomize_weights() %>%
  train_network_sgd(X_rings, y_rings_ohe, num_epochs = 1000,
                    eta = 3e-4, loss = "crossentropy") -> trained_rings_soft_ce

# plotting pace of convergence
ggplot(data.frame(val = unlist(c(trained_rings_lin_mse$training_history,
                                 trained_rings_soft_mse$training_history,
                                 trained_rings_soft_ce$training_history)),
                  activation = c(rep("linear", 1000), rep("softmax", 2000)),
                  loss = rep(c("mse", "mse", "crossentropy"), each = 1000),
                  epoch = rep(1:1000, 3)),
       aes(x = epoch, y = val, color = activation, linetype = loss)) +
  geom_line() +
  ggtitle("comparison of different loss and activation function", "for dataset 'rings'")

ggsave("lab/lab4/rings_loss.png", device = png())
dev.off()

# fitting
trained_rings_soft_ce %>%
  feed_network(X_rings) %>%
  select_max() -> fit_rings_soft_ce

trained_rings_lin_mse %>%
  feed_network(X_rings) %>%
  select_max() -> fit_rings_lin_mse

trained_rings_soft_mse %>%
  feed_network(X_rings) %>%
  select_max() -> fit_rings_soft_mse

# creating grid to display decision boundaries
X_grid <- data.frame(x = rep(seq(-2, 2, length.out = 100), each = 100),
                     y = rep(seq(-2, 2, length.out = 100), times = 100))

trained_rings_soft_mse %>%
  feed_network(as.matrix(X_grid)) %>%
  select_max() -> fit_grid

# plotting decision boundaries
ggplot() +
  geom_point(data = cbind(X_grid, fitted_value = as.factor(fit_grid[, 1])),
             mapping = aes(x = x, y = y, color = fitted_value)) +
  geom_point(data = data.frame(x = X_rings[, 1], y = X_rings[, 2],
                               true_value = factor(dat_rings[, 3])),
             mapping = aes(x = x, y = y, shape = true_value)) +
  ggtitle("network decision boundaries and true values",
          "for dataset 'rings' and model with softmax and mse")

ggsave("lab/lab4/rings_bound.png", device = png())
dev.off()
### xor

# building networks - two for each dataset
net_xor_lin <- neural_network(2) +   # input size = 2
  hidden_layer(20, "sigmoid") +
  hidden_layer(30, "sigmoid") +
  hidden_layer(20, "sigmoid") +
  output_layer(2, "linear")           # output size = 2, softmax

net_xor_soft <- neural_network(2) +   # input size = 2
  hidden_layer(20, "sigmoid") +
  hidden_layer(30, "sigmoid") +
  hidden_layer(20, "sigmoid") +
  output_layer(2, "softmax")          # output size = 2, softmax

# training
set.seed(123)
net_xor_lin %>%
  randomize_weights() %>%
  train_network_momentum(X_xor, y_xor_ohe, num_epochs = 3000,
                    eta = 1e-4, loss = "mse") -> trained_xor_lin_mse

net_xor_soft %>%
  randomize_weights() %>%
  train_network_momentum(X_xor, y_xor_ohe, num_epochs = 3000,
                         eta = 3e-3, loss = "mse") -> trained_xor_soft_mse

net_xor_soft %>%
  randomize_weights() %>%
  train_network_sgd(X_xor, y_xor_ohe, num_epochs = 3000,
                    eta = 3e-4, loss = "crossentropy") -> trained_xor_soft_ce

# plotting pace of convergence
ggplot(data.frame(val = unlist(c(trained_xor_lin_mse$training_history,
                                 trained_xor_soft_mse$training_history,
                                 trained_xor_soft_ce$training_history)),
                  activation = c(rep("linear", 3000), rep("softmax", 6000)),
                  loss = rep(c("mse", "mse", "crossentropy"), each = 3000),
                  epoch = rep(1:3000, 3)),
       aes(x = epoch, y = val, color = activation, linetype = loss)) +
  geom_line() +
  scale_y_continuous(limits = c(0, 2)) +
  ggtitle("comparison of different loss and activation function", "for dataset 'xor'")

ggsave("lab/lab4/xor_loss.png", device = png())
dev.off()

# fitting
trained_xor_soft_ce %>%
  feed_network(X_xor) %>%
  select_max() -> fit_xor_soft_ce

trained_xor_lin_mse %>%
  feed_network(X_xor) %>%
  select_max() -> fit_xor_lin_mse

trained_xor_soft_mse %>%
  feed_network(X_xor) %>%
  select_max() -> fit_xor_soft_mse

# creating grid to display decision boundaries
X_grid <- data.frame(x = rep(seq(-2, 2, length.out = 100), each = 100),
                     y = rep(seq(-2, 2, length.out = 100), times = 100))

trained_xor_soft_mse %>%
  feed_network(as.matrix(X_grid)) %>%
  select_max() -> fit_grid

# plotting decision boundaries
ggplot() +
  geom_point(data = cbind(X_grid, fitted_value = as.factor(fit_grid[, 1])),
             mapping = aes(x = x, y = y, color = fitted_value)) +
  geom_point(data = data.frame(x = X_xor[, 1], y = X_xor[, 2],
                               true_value = factor(dat_xor[, 3])),
             mapping = aes(x = x, y = y, shape = true_value)) +
  ggtitle("network decision boundaries and true values",
          "for dataset 'xor' and model with softmax and mse")

ggsave("lab/lab4/xor_bound.png", device = png())
