#' @export
train_network <- function(network, X, y, batch_size = NULL, eta = 1e-3, num_epochs = 10) {
  n <- nrow(X)
  if (is.null(batch_size)) batch_size <- n
  batch_num <- ceiling(n / batch_size)

  derivatives <- match_deriv_to_name(network$activation_names)
  a <- list()
  e <- list()

  for (num in 1:num_epochs) {
    batches <- split_data(X, y, batch_size, batch_num)
    for (batch in batches) {
      X_batch <- batch$X_batch
      y_batch <- batch$y_batch

      del_w <- lapply(network$weights, function(mat) matrix(0, nrow(mat), ncol(mat)))

      for (i in 1:nrow(X_batch)) {
        a[[1]] <- cbind(1, X_batch[i, , drop = FALSE]) %*% network$weights[[1]]
        y_real <- y_batch[i, , drop = FALSE]

        for (k in 2:(network$layers)) {
          a[[k]] <- cbind(1, network$activations[[k - 1]](a[[k - 1]])) %*% network$weights[[k]]
        }

        y_pred <- network$activations[[network$layers]](a[[network$layers]])

        e[[network$layers]] <- derivatives[[network$layers]](a[[network$layers]]) * (y_pred - y_real)

        for (k in (network$layers - 1):1) {
          e[[k]] <- derivatives[[k]](a[[k]]) * (e[[k + 1]] %*% t(network$weights[[k + 1]][-1, , drop = FALSE]))
        }

        for (k in (network$layers):2) {
          del_w[[k]] <- del_w[[k]] + rbind(1, network$activations[[k]](t(a[[k - 1]]))) %*% e[[k]]
        }

        del_w[[1]] <- del_w[[1]] + rbind(1, network$activations[[1]](t(X_batch[i, , drop = FALSE]))) %*% e[[1]]

      }

      network$weights <- lapply(1:network$layers, function(ind) network$weights[[ind]] - eta * del_w[[ind]])
    }
    cat("Epoch: ", num, "; MSE: ", mse(network, X, y), " \n")
  }

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

#' @export
mse <- function(network, X, y) {
  sum((feed_network(network, X) - y)^2 / nrow(X))
}
