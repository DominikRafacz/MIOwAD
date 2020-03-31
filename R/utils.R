match_fun_to_name <- function(fun_vec) {
  list(sigmoid = sigmoid,
       linear = linear,
       softmax = softmax,
       mse = mse,
       crossentropy = crossentropy)[fun_vec]
}

match_deriv_to_name <- function(fun_vec) {
  list(sigmoid = sigmoid_deriv,
       linear = linear_deriv,
       softmax = softmax_deriv,
       mse = mse_deriv,
       crossentropy = crossentropy_deriv)[fun_vec]
}


split_data <- function(X, y, batch_size, batch_num) {
  inds <- ceiling(sample(nrow(X)) / batch_size)
  lapply(1:batch_num, function(ind) list(X_batch = X[inds == ind, , drop = FALSE],
                                         y_batch = y[inds == ind, , drop = FALSE]))
}


build <- function(nn_proto) {
  network <- list(layers = length(nn_proto$sizes) - 1,
                  sizes = nn_proto$sizes)
  network$weights <- lapply(1:network$layers,
                    function(ind) matrix(0, nrow = network$sizes[ind] + 1,
                                         ncol = network$sizes[ind + 1]))
  network$activations <- match_fun_to_name(nn_proto$activations)
  network$activation_names <- nn_proto$activations
  class(network) <- "neural_network"
  network
}

sigmoid <- function(x) 1 / (1 + exp(-x))

linear <- function(x) x

softmax <- function(x) exp(x) / matrix(rep(apply(exp(x), 1, sum), ncol(x)), nrow(x))

sigmoid_deriv <- function(x) exp(-x) / (1 + exp(-x))^2

linear_deriv <- function(x) matrix(1, nrow(x), ncol(x))

softmax_deriv <- function(x) softmax(x) * (1 - softmax(x))

mse_deriv <- function(y_real, y_pred) {
  (y_pred - y_real) / 2
}

crossentropy_deriv <- function(y_real, y_pred) {
  - y_real / (y_pred + 0.005)
}
