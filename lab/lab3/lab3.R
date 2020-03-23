dsets <- c("square", "steps", "multimodal")

# load datasets
dat <- setNames(lapply(dsets, function(dset) {
  list(train = read.csv(paste0("data/regression/", dset, "-large-training.csv")),
       test = read.csv(paste0("data/regression/", dset, "-large-training.csv")))
}), dsets)

# split datasets into X, y, train and test
X <- setNames(lapply(dsets, function(dset) {
  list(train = scale(as.matrix(dat[[dset]][["train"]])[, 2, drop = FALSE]),
       test = scale(as.matrix(dat[[dset]][["test"]])[, 2, drop = FALSE]))
}), dsets)

y <- setNames(lapply(dsets, function(dset) {
  list(train = scale(as.matrix(dat[[dset]][["train"]])[, 3, drop = FALSE]),
       test = scale(as.matrix(dat[[dset]][["test"]])[, 3, drop = FALSE]))
}), dsets)

# create network architecture
net <- setNames(lapply(dsets, function(dset)
  neural_network(1) +
    hidden_layer(10, "sigmoid") +
    hidden_layer(10, "sigmoid") +
    hidden_layer(10, "sigmoid") +
    output_layer(1, "linear")), dsets)

# specify functions to train network
methods <- list(sgd = train_network_sgd, momentum = train_network_momentum, rmsprop = train_network_rmsprop)

# set seed
set.seed(123)

# train networks for each dataset with each method
trained <- setNames(lapply(dsets, function(dset) lapply(methods, function(method)
  method(randomize_weights(net[[dset]]), X[[dset]][["train"]], y[[dset]][["train"]],
         num_epochs = 1000, eta = 1e-3, batch_size = 100))), dsets)

# extract history of training
history <- lapply(trained, function(dset_res)
  do.call(rbind, lapply(names(methods), function(method)
    data.frame(mse = unlist(dset_res[[method]]$training_history), method = method, iter = 1:length(dset_res[[method]]$training_history)))))

# generate plots
ggplot(data = history[["square"]], aes(x = iter, y = mse, color = method)) +
  geom_line()

ggplot(data = history[["steps"]], aes(x = iter, y = mse, color = method)) +
  geom_line()

ggplot(data = history[["multimodal"]], aes(x = iter, y = mse, color = method)) +
  geom_line()
