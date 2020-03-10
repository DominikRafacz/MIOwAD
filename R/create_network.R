#' @export
neural_network <- function(input_size = 1) {
  structure(list(sizes = input_size), class = "nn_proto")
}

#' @export
hidden_layer <- function(size = 1, activation = "linear") {
  structure(list(size = size, activation = activation), class = "nn_hidden_layer")
}

#' @export
output_layer <- function(size = 1, activation = "linear") {
  structure(list(size = size, activation = activation), class = "nn_output_layer")
}

#' @export
`+.nn_proto` <- function(e1, e2) {
  if (class(e2) == "nn_hidden_layer") {
    e1$sizes <- c(e1$sizes, e2$size)
    e1$activations <- c(e1$activations, e2$activation)
    e1
  } else if (class(e2) == "nn_output_layer") {
    e1$sizes <- c(e1$sizes, e2$size)
    e1$activations <- c(e1$activations, e2$activation)
    build(e1)
  }
}
