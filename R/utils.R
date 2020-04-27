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
randomize_weights_rnorm <- function(network, mu = 0, sigma = 1) {
  network$weights <- lapply(network$weights, function(mat) {
    inp <- nrow(mat)
    out <- ncol(mat)
    matrix(rnorm(inp * out, mean = mu, sd = sigma), inp, out)
  })
  network
}

#' @export
randomize_weights_runif <- function(network, min = 0, max = 1) {
  network$weights <- lapply(network$weights, function(mat) {
    inp <- nrow(mat)
    out <- ncol(mat)
    matrix(runif(inp * out, min = min, max = max), inp, out)
  })
  network
}

#' @export
randomize_weights_xavier <- function(network) {
  network$weights <- lapply(network$weights, function(mat) {
    inp <- nrow(mat)
    out <- ncol(mat)
    s <- sqrt(6 / (inp + out - 1))
    matrix(runif(inp * out, min = -s, max = s), inp, out)
  })
  network
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

scale_color_legendary <- function() scale_colour_manual(values = c("#e55934", "#5bc0eb", "#9bc53d", "#fde74c", "#CE8D66", "#963484"))
scale_fill_legendary <- function() scale_fill_manual(values = c("#e55934", "#5bc0eb", "#9bc53d", "#fde74c", "#CE8D66", "#963484"))
