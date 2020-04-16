calc_gradients <- function(network, X_batch, y_batch, loss, lambda, n, dropout_rate) {
  if (dropout_rate > 0)
    calc_gradients_dropout(network, X_batch, y_batch, loss, lambda, n, dropout_rate)
  else
    calc_gradients_wo_dropout(network, X_batch, y_batch, loss, lambda, n)
}

calc_gradients_wo_dropout <- function(network, X_batch, y_batch, loss, lambda, n) {
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

calc_gradients_dropout <- function(network, X_batch, y_batch, loss, lambda, n, dropout_rate) {
  reduced_sizes <- c(network$sizes[1],
                     sapply(network$sizes[2:network$layers],
                            function(size) ceiling(size * (1 - dropout_rate))),
                     network$sizes[network$layers + 1])

  active_indices <- c(list(1:network$sizes[1]),
                      lapply(2:(network$layers),
                             function(ind) sample(1:network$sizes[ind], reduced_sizes[ind])),
                      list(1:network$sizes[network$layers + 1]))

  reduced_weigths <- list()
  for (k in 1:network$layers) {
    reduced_weigths[[k]] <- network$weights[[k]][c(1, active_indices[[k]] + 1),
                                                 active_indices[[k + 1]], drop = FALSE]
  }

  network_cp <- list(layers = network$layers,
                     sizes = reduced_sizes,
                     weights = reduced_weigths,
                     activations = network$activations,
                     activation_names = network$activation_names)

  del_tmp <- calc_gradients_wo_dropout(network_cp, X_batch, y_batch, loss, lambda, n)

  del_w <- lapply(network$weights, function(mat) matrix(0, nrow(mat), ncol(mat)))

  for (k in (network$layers):1) {
    del_w[[k]][c(1, active_indices[[k]] + 1), active_indices[[k + 1]]] <- del_tmp[[k]]
  }

  del_w
}

training_setuper <- function(X, batch_size, loss, lambda) {
  n <- nrow(X)
  if (is.null(batch_size)) batch_size <- n
  batch_num <- ceiling(n / batch_size)

  # get loss function by name
  loss_fun <- match_fun_to_name(loss)[[1]]
  loss_msg <- if (lambda > 0) paste0(loss, " + L2") else loss
  list(n = n, batch_size = batch_size, batch_num = batch_num,
       loss_fun = loss_fun, loss_msg = loss_msg)
}

epoch_finisher <- function(network, X, y, num, verbose, loss_fun, loss_msg, X_validation,
                           y_validation, validation_threshold) {
  m <- loss_fun(network, X, y)
  if (verbose) cat("Epoch: ", num, ", Cost: ", loss_msg, " = ", m, "\n", sep = "")
  network$training_history$training <- c(network$training_history$training, m)
  # if cost on training set drop, but on validation rises - it's time to stop
  stop_flag <- FALSE
  if (!is.null(X_validation)) {
    m_validation <- loss_fun(network, X_validation, y_validation)
    network$training_history$validation <- c(network$training_history$validation, m_validation)
    stop_flag <- (num > 1 &&
        network$training_history$validation[num - 1] * validation_threshold < m_validation &&
        network$training_history$training[num - 1] < network$training_history$training[num])
  }
  list(network = network, stop_flag = stop_flag)
}


#' @export
train_network_sgd <- function(network, X, y, batch_size = NULL, eta = 1e-3, num_epochs = 10, loss = "mse",
                              lambda = 0, dropout_rate = 0, X_validation = NULL, y_validation = NULL,
                              validation_threshold = 1.01, verbose = TRUE) {
  setup <- training_setuper(X, batch_size, loss, lambda)

  for (num in 1:num_epochs) {
    # split into batches
    batches <- split_data(X, y, setup$batch_size, setup$batch_num)
    for (batch in batches) {
      # calculate gradients of cost in accordance to layers
      del_w <- calc_gradients(network, batch$X_batch, batch$y_batch, loss, lambda, setup$n, dropout_rate)

      #update weights
      network$weights <- lapply(1:network$layers, function(ind) network$weights[[ind]] - eta * del_w[[ind]])
    }

    fnsh <- epoch_finisher(network, X, y, num, verbose, setup$loss_fun, setup$loss_msg,
                           X_validation, y_validation, validation_threshold)
    network <- fnsh$network
    if (fnsh$stop_flag) break
  }

  network
}


#' @export
train_network_momentum <- function(network, X, y, batch_size = NULL, eta = 1e-3,
                                  gamma = 0.9, num_epochs = 10, loss = "mse",
                                  lambda = 0, dropout_rate = 0, X_validation = NULL, y_validation = NULL,
                                  validation_threshold = 1.01, verbose = TRUE) {
  setup <- training_setuper(X, batch_size, loss, lambda)

  momentum <- lapply(network$weights, function(mat) matrix(0, nrow(mat), ncol(mat)))
  for (num in 1:num_epochs) {
    batches <- split_data(X, y, setup$batch_size, setup$batch_num)
    for (batch in batches) {
      del_w <- calc_gradients(network, batch$X_batch, batch$y_batch, loss, lambda, setup$n, dropout_rate)
      momentum <- lapply(1:network$layers, function(ind) momentum[[ind]] * gamma + eta * del_w[[ind]])
      network$weights <- lapply(1:network$layers, function(ind) network$weights[[ind]] - momentum[[ind]])
    }

    fnsh <- epoch_finisher(network, X, y, num, verbose, setup$loss_fun, setup$loss_msg,
                           X_validation, y_validation, validation_threshold)
    network <- fnsh$network
    if (fnsh$stop_flag) break
  }

  network
}

#' @export
train_network_rmsprop <- function(network, X, y, batch_size = NULL, eta = 1e-3, beta = 0.9, num_epochs = 10,
                                  loss = "mse",
                                  lambda = 0, dropout_rate = 0, X_validation = NULL, y_validation = NULL,
                                  validation_threshold = 1.01, verbose = TRUE) {
  setup <- training_setuper(X, batch_size, loss, lambda)

  eg2 <- lapply(network$weights, function(mat) matrix(0, nrow(mat), ncol(mat)))

  for (num in 1:num_epochs) {
    batches <- split_data(X, y, setup$batch_size, setup$batch_num)
    for (batch in batches) {
      del_w <- calc_gradients(network, batch$X_batch, batch$y_batch, loss, lambda, setup$n, dropout_rate)
      eg2 <- lapply(1:network$layers, function(ind) beta * eg2[[ind]] + (1 - beta) * del_w[[ind]]^2)
      network$weights <- lapply(1:network$layers,
                                function(ind) network$weights[[ind]] - eta * del_w[[ind]] / (sqrt(eg2[[ind]]) + 1e-5))

    }

    fnsh <- epoch_finisher(network, X, y, num, verbose, setup$loss_fun, setup$loss_msg,
                           X_validation, y_validation, validation_threshold)
    network <- fnsh$network
    if (fnsh$stop_flag) break
  }

  network
}

