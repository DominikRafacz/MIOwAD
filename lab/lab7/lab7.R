X_2d <- read.csv("data/kohonen/hexagon.csv")

net_2d <- kohonen_network(X_2d[, 1:2], 20, 20, lambda = 50)

plot_kohonen(generate_net_plot_data(net_2d), X_2d)

X_3d <- read.csv("data/kohonen/cube.csv")

net_3d <- kohonen_network(X[, 1:3], 10, 10, lambda = 50)

plot_kohonen(generate_net_plot_data(net_3d[1:2,,]), X_3d[, c(1,2,4)])
