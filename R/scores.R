#' @export
mse <- function(network, X, y) {
  sum((feed_network(network, X) - y)^2 / nrow(X))
}

#' @export
crossentropy <- function(network, X, y) {
  # crossentropy
  -sum(log(feed_network(network, X) + 0.0001) * y) / nrow(X)
}

mse_deriv <- function(y_real, y_pred) {
  (y_pred - y_real) / 2
}

crossentropy_deriv <- function(y_real, y_pred) {
  - y_real / (y_pred + 0.005)
}
