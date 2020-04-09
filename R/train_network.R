calc_gradients <- function(network, X_batch, y_batch, loss, lambda, n) {
  # get derivatives of activation functions
  derivatives <- match_deriv_to_name(network$activation_names)
  # get derivative of loss function
  cost_derivative <- match_deriv_to_name(loss)[[1]]
  # initialize errors
  a <- list()
  e <- list()
  # initialize gradients
  del_w <- lapply(network$weights, function(mat) matrix(0, nrow(mat), ncol(mat)))

  # values on the input layer
  a[[1]] <- cbind(1, X_batch) %*% network$weights[[1]]

  # feed forward
  for (k in 2:(network$layers)) {
    a[[k]] <- cbind(1, network$activations[[k - 1]](a[[k - 1]])) %*% network$weights[[k]]
  }

  #calculate predictions
  y_pred <- network$activations[[network$layers]](a[[network$layers]])

  # errors on the last layer
  e[[network$layers]] <- derivatives[[network$layers]](a[[network$layers]]) * cost_derivative(y_batch, y_pred)

  # backpropagation
  for (k in (network$layers - 1):1) {
    e[[k]] <- derivatives[[k]](a[[k]]) * (e[[k + 1]] %*% t(network$weights[[k + 1]][-1, , drop = FALSE]))
  }

  # calculate gradients
  for (k in (network$layers):2) {
    del_w[[k]] <- del_w[[k]] + rbind(1, network$activations[[k]](t(a[[k - 1]]))) %*% e[[k]] +
      lambda / n * rbind(0, network$weights[[k]][-1, , drop = FALSE]) # L2 regularization
  }

  del_w[[1]] <- del_w[[1]] + rbind(1, network$activations[[1]](t(X_batch))) %*% e[[1]] +
    lambda / n * rbind(0, network$weights[[1]][-1, , drop = FALSE]) # L2 regularization

  del_w
}

#' @export
train_network_sgd <- function(network, X, y, batch_size = NULL, eta = 1e-3, num_epochs = 10, loss = "mse",
                              lambda = 0, dropout_rate = 0, X_validation = NULL, y_validation = NULL) {
  # initialize values
  n <- nrow(X)
  if (is.null(batch_size)) batch_size <- n
  batch_num <- ceiling(n / batch_size)
  network$training_history <- list()

  # get loss function by name
  loss_fun <- match_fun_to_name(loss)[[1]]


  for (num in 1:num_epochs) {
    # split into batches
    batches <- split_data(X, y, batch_size, batch_num)
    for (batch in batches) {
      # calculate gradients of cost in accordance to layers
      del_w <- calc_gradients(network, batch$X_batch, batch$y_batch, loss, lambda, n)

      #update weights
      network$weights <- lapply(1:network$layers, function(ind) network$weights[[ind]] - eta * del_w[[ind]])
    }
    m <- loss_fun(network, X, y)
    cat("Epoch: ", num, ", ", loss, ": ", m, "\n", sep = "")
    network$training_history[[num]] <- m
  }

  network
}


#' @export
train_network_momentum <- function(network, X, y, batch_size = NULL, eta = 1e-3,
                                  gamma = 0.9, num_epochs = 10, loss = "mse",
                                  lambda = 0, dropout_rate = 0, X_validation = NULL, y_validation = NULL) {
  n <- nrow(X)
  if (is.null(batch_size)) batch_size <- n
  batch_num <- ceiling(n / batch_size)

  network$training_history <- list()
  loss_fun <- match_fun_to_name(loss)[[1]]

  momentum <- lapply(network$weights, function(mat) matrix(0, nrow(mat), ncol(mat)))
  for (num in 1:num_epochs) {
    batches <- split_data(X, y, batch_size, batch_num)
    for (batch in batches) {
      del_w <- calc_gradients(network, batch$X_batch, batch$y_batch, loss, lambda, n)
      momentum <- lapply(1:network$layers, function(ind) momentum[[ind]] * gamma + eta * del_w[[ind]])
      network$weights <- lapply(1:network$layers, function(ind) network$weights[[ind]] - momentum[[ind]])
    }
    m <- loss_fun(network, X, y)
    cat("Epoch: ", num, ", ", loss, ": ", m, "\n", sep = "")
    network$training_history[[num]] <- m
  }

  network
}

#' @export
train_network_rmsprop <- function(network, X, y, batch_size = NULL, eta = 1e-3, beta = 0.9, num_epochs = 10,
                                  loss = "mse",
                                  lambda = 0, dropout_rate = 0, X_validation = NULL, y_validation = NULL) {
  n <- nrow(X)
  if (is.null(batch_size)) batch_size <- n
  batch_num <- ceiling(n / batch_size)

  network$training_history <- list()
  loss_fun <- match_fun_to_name(loss)[[1]]

  eg2 <- lapply(network$weights, function(mat) matrix(0, nrow(mat), ncol(mat)))

  for (num in 1:num_epochs) {
    batches <- split_data(X, y, batch_size, batch_num)
    for (batch in batches) {
      del_w <- calc_gradients(network, batch$X_batch, batch$y_batch, loss, lambda, n)
      eg2 <- lapply(1:network$layers, function(ind) beta * eg2[[ind]] + (1 - beta) * del_w[[ind]]^2)
      network$weights <- lapply(1:network$layers,
                                function(ind) network$weights[[ind]] - eta * del_w[[ind]] / (sqrt(eg2[[ind]]) + 1e-5))

    }
    m <- loss_fun(network, X, y)
    cat("Epoch: ", num, ", ", loss, ": ", m, "\n", sep = "")
    network$training_history[[num]] <- m
  }

  network
}

