neural_network <- function(input_size = 1) {
  structure(list(sizes = input_size), class = "nn_proto")
}

hidden_layer <- function(size = 1, activation = "linear") {
  structure(list(size = size, activation = activation), class = "nn_hidden_layer")
}

output_layer <- function(size = 1, activation = "linear") {
  structure(list(size = size, activation = activation), class = "nn_output_layer")
}

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

build <- function(nn_proto) {
  network <- list(layers = length(nn_proto$sizes),
                  sizes = nn_proto$sizes)
  network$weights <- lapply(1:(network$layers - 1), 
                    function(ind) matrix(0, nrow = network$sizes[ind] + 1,
                                         ncol = network$sizes[ind + 1]))
  network$activations <- lapply(nn_proto$activations, match_fun_to_name)
  network$activation_names <- nn_proto$activations
  class(network) <- "neural_network"
  network
}


sigmoid <- function(x) 1 / (1 + exp(-x))

linear <- function(x) x