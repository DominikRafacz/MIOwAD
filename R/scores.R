#' @export
mse <- function(network, X, y) {
  sum((feed_network(network, X) - y)^2 / nrow(X))
}

#' @export
crossentropy <- function(network, X, y) {
  # crossentropy
  -sum(log(feed_network(network, X) + 0.0001) * y) / nrow(X)
}

#' @export
accuracy <- function(network, X, y) {
  sum(select_max(feed_network(network, X)) == y) / nrow(y)
}

mse_deriv <- function(y_real, y_pred) {
  (y_pred - y_real) / 2
}

crossentropy_deriv <- function(y_real, y_pred) {
  (y_pred - y_real) / (y_pred * (1 - y_pred) + 0.005)
}
