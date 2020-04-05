sigmoid <- function(x) 1 / (1 + exp(-x))

linear <- function(x) x

softmax <- function(x) exp(x) / matrix(rep(apply(exp(x), 1, sum), ncol(x)), nrow(x))

hyperbolic_tan <- function(x) tanh(x)

relu <- function(x) ifelse(x < 0, 0, x)


sigmoid_deriv <- function(x) exp(-x) / (1 + exp(-x))^2

linear_deriv <- function(x) matrix(1, nrow(x), ncol(x))

softmax_deriv <- function(x) softmax(x) * (1 - softmax(x))

hyperbolic_tan_deriv <- function(x) 1 - tanh(x)^2

relu_deriv <- function(x) ifelse(x < 0, 0, 1)
