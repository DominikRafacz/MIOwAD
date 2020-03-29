#' @export
feed_network <- function(network, X) {
  act <- X

  for (i in 1:network$layers) {
    act <- cbind(1, act)
    act <- network$activations[[i]](act %*% network$weights[[i]])
  }

  act
}

#' @export
select_max <- function(probs) {
  res <- apply(probs, 1, function(x) ((1:ncol(probs) - 1)[max(x) == x])[1])
  matrix(res, length(res))
}
