match_fun_to_name <- function(fun_vec) {
  list(sigmoid = sigmoid,
       linear = linear,
       softmax = softmax,
       relu = relu,
       tanh = hyperbolic_tan,
       mse = mse,
       crossentropy = crossentropy)[fun_vec]
}

match_deriv_to_name <- function(fun_vec) {
  list(sigmoid = sigmoid_deriv,
       linear = linear_deriv,
       softmax = softmax_deriv,
       relu = relu_deriv,
       tanh = hyperbolic_tan_deriv,
       mse = mse_deriv,
       crossentropy = crossentropy_deriv)[fun_vec]
}


split_data <- function(X, y, batch_size, batch_num) {
  inds <- ceiling(sample(nrow(X)) / batch_size)
  lapply(1:batch_num, function(ind) list(X_batch = X[inds == ind, , drop = FALSE],
                                         y_batch = y[inds == ind, , drop = FALSE]))
}

#' @export
randomize_weights <- function(network) {
  network$weights <- lapply(network$weights, function(mat) {
    inp <- nrow(mat)
    out <- ncol(mat)
    matrix(rnorm(inp * out), inp, out)
  })
  network
}
