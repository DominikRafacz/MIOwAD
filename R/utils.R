match_fun_to_name <- function(fun_vec) {
  list(sigmoid = sigmoid,
       linear = linear)[fun_vec]
}
