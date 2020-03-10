#' @export
feed_network <- function(network, X) {
  act <- X

  for (i in 1:network$layers) {
    act <- cbind(1, act)
    act <- network$activations[[i]](act %*% network$weights[[i]])
  }

  act
}
