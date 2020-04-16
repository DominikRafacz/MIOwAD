#' @importFrom ggplot2 ggplot aes geom_point geom_line scale_color_gradient labs ggtitle theme_void theme
#' @importFrom dplyr left_join
#' @export
plot_weigths <- function(net) {
  sizes <- net$sizes
  layers <- net$layers

  neurons <- as.data.frame(
    do.call(
      rbind,
      c(lapply(1:layers,
               function(i) cbind(layer = i,
                                 neuron_num = 0:sizes[i],
                                 neuron_pos = 0:sizes[i] - sizes[i] / 2 )),
        list(cbind(layer = layers + 1,
                   neuron_num = 1:sizes[layers + 1],
                   neuron_pos = 1:sizes[layers + 1] - sizes[layers + 1] / 2 - 0.5)))))

  edges <- do.call(rbind, lapply(1:layers, function(i)
    data.frame(value = as.vector(net$weights[[i]]),
               input_num = rep(0:sizes[i], sizes[i + 1]),
               output_num = rep(1:sizes[i + 1], each = sizes[i] + 1), layer = i)))

  edges <- left_join(edges, neurons, by = c("layer" = "layer", "input_num" = "neuron_num"))
  neurons$layer_n <- neurons$layer - 1
  edges <- left_join(edges, neurons, by = c("layer" = "layer_n", "output_num" = "neuron_num"))

  ggplot() +
    geom_point(data = neurons, aes(x = layer, y = neuron_pos, shape = neuron_num == 0), size = 7) +
    geom_segment(data = edges, mapping = aes(x = layer, xend = layer.y,
                                             y = neuron_pos.x, yend = neuron_pos.y,
                                             color = value),
                 size = 2) +
    scale_color_gradient(low = "#DF1207", high = "#5AFF4C") +
    labs(shape = "bias neuron", color = "weight") +
    ggtitle("neural net weights") +
    theme_void() +
    theme(legend.position = "bottom")
}
