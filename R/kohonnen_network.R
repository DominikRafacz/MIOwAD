# Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł.
# Dominik Rafacz

#' @export
gauss <- function(x, scale) {
  exp(-(x * scale)^2)
}

#' @export
mexican_hat <- function(x, scale) {
  (2 - 4 * (x * scale)^2) * exp(-(x * scale)^2)
}

#' @importFrom dplyr `%>%` mutate filter pull
#' @export
kohonen_network <- function(input_X, N, M, lambda = 100, ngh_fun = gauss,
                            scale = 1, topology = "square", verbose = FALSE) {
  num_obs <- nrow(input_X)
  inp_dims <- ncol(input_X)

  # inicjalizujemy losowo tensor wag - jest to tablica trojwymiarowa, gdzie pierwszy wymiar
  # tabicy odpowiada przestrzeni wejsciowej, a pozostałe dwa wymiary odpowiadaja rozmieszczeniu
  # neuronow - czyli weights[k, i, j] oznacza wage neuronu [i, j] na elemencie k-tym wektora
  # wejsciowego; taka kolejnosc upraszcza obliczenia macierzowe
  init_data <- rnorm(inp_dims * N * M)
  weights <- array(init_data, c(inp_dims, N, M))

  # wybieramy funkcje odleglosci pomiedzy neuronami na podstawie topologii sieci
  dist_fun <- c(square = square_metric, hex = hex_metric)[[topology]]

  # liczymy macierz odleglosci pomiedzy neuronami w siatce - mozemy to zrobic juz na poczatku,
  # gdyz ta odleglosc sie nie zmienia pomiedzy iteracjami algorytmu; ramka danych sklada sie z
  # N^2 * M^2 wierszy, gdzie kazdy wiersz zawiera w kolumnie dist odleglosc pomiedzy dwoma neuronami
  # o wspolrzednych (x1, y1) i (x2, y2); odleglosc liczymy przez przeksztalcenie odleglosci z metryki
  # miejskiej przez podana na wejsciu funkcje ngh_fun, np. funkcje gaussa
  ngh <-
    mutate(data.frame(
      x1 = rep(rep(1:N, M), M * N),
      y1 = rep(rep(1:M, each = N), M * N),
      x2 = rep(rep(1:N, M), each = M * N),
      y2 = rep(rep(1:M, each = N), each = M * N)
    ),
    dist = ngh_fun(dist_fun(x1, y1, x2, y2), scale))

  if (verbose) width <- getOption("width")

  # dla kazdej iteracji algorytmu od 1 do lambda
  for (iter in 1:lambda) {
    if (verbose) {
      txt <- paste0("Iteration ", iter , " / ", lambda, " : ")
      cat(txt)
      prog <- 0
      stars <- 0
    }

    # tworzymy permutajcje indeksow obserwacji macierzy wejscioej
    perm <- sample(1:num_obs, num_obs)

    # dla kazdej obserwacji tablicy wejsciowej w nowej permutacji
    for (ind in perm) {
      # liczymy macierz odleglosci obserwacji od neuronow; robimy to, odejmujac od wag kazdego
      # neuronu obserwacje i nastepnie dla kazdego neuronu liczac sume wartosci kwadratow wspolrzednych;
      # uzyskujemy w ten sposob kwadraty odleglosci euklidesowej, ale z liniowosci kwadratu one tez nam
      # odpowiadaja do znajdowania minimum
      dists <- apply(weights - as.numeric(input_X[ind, ]), 2:3, function(neuron_dist) sum(neuron_dist^2))

      # wyznaczamy indeksy neurona z najmniejsza odlegloscia
      mininds <- which(dists == min(dists), arr.ind = TRUE)[1, ]

      # wybieramy z macierzy odleglosci tylko te rekordy, ktore odpowiadaja wyznaczonemu neuronowi
      n_factor <- filter(ngh,
             x1 == mininds[1],
             y1 == mininds[2])[["dist"]]

      # aktualizujemy wagi, dla kazdego neuronu dodajac do jego wagi roznice wagi neuronu i wektora
      # obserwacji wejsciowej, przemnozone przez wage odleglosci neuronu oraz przez wspolczynnik
      # wygaszania
      weights <- weights +
        (-weights + as.numeric(input_X[ind, ])) *
        rep(n_factor, each = inp_dims) *
        exp(-iter/lambda)

      if (verbose) {
        prog <- prog + 1
        if (floor(prog * (width - nchar(txt)) / num_obs) > stars) {
          stars <- stars + 1
          cat("*")
        }
      }
    }
    if (verbose) cat("\n")
  }

  # zwracamy sieć
  list(weights = weights, N = N, M = M, topology = topology)
}


# w topologii siatki kwadratowej, odleglosc liczymy w metryce miejskiej
square_metric <- function(x1, y1, x2, y2) abs(x1 - x2) + abs(y1 - y2)

# w topologii sieci heksagonalnej, niektore neurony nie uznajemy za polaczone - wyrazamy
# to za pomoca inaczej liczonej odleglosci miedzy nimi
hex_metric <- function(x1, y1, x2, y2) {
  y_diff <- abs(y1 - y2)
  x_diff <- abs(x1 - x2)
  y_diff + x_diff + 2 * (ifelse(y_diff > x_diff, floor((y_diff - x_diff) / 2), 0) +
                   ifelse((y1 - x1) %% 2 == 0, y1 > y2, y2 > y1))
}

#' @importFrom dplyr `%>%` filter left_join rename
#' @importFrom stringi stri_endswith_fixed stri_replace_first_fixed
#' @export
generate_net_plot_data <- function(net) {
  N <- net$N
  M <- net$M
  # generujemy indeksy sieci neuronow na podstawie rozmiaru sieci
  inds <- data.frame(
    x_ind = rep(1:N, M),
    y_ind = rep(1:M, each = N)
  )

  # tworzymy ramke danych opisujaca punkty
  net_points <- as.data.frame(
    do.call(rbind,lapply(1:M, function(i) t(net$weights[,,i]))))

  net_points <- cbind(setNames(
    net_points, paste0("inp_dim_", 1:ncol(net_points))), inds)

  net_lines <- do.call(rbind, lapply(1:N, function(x_ind)
    do.call(rbind, lapply(1:M, function(y_ind) {
      data.frame(x_ind = c(x_ind, x_ind),
                 y_ind = c(y_ind, y_ind),
                 x_ind_next = c(x_ind, x_ind + 1),
                 y_ind_next = c(y_ind + 1, y_ind))
    }))
  )) %>%
    filter(x_ind_next <= N,
           y_ind_next <= M)


  if (net$topology == "hex") {
    net_lines <- net_lines %>%
      filter((x_ind - y_ind) %% 2 == 0 |
               x_ind_next == x_ind + 1)
  }

  net_lines <- net_lines %>%
    left_join(net_points, by = c("x_ind" = "x_ind",
                                 "y_ind" = "y_ind")) %>%
    left_join(net_points, by = c("x_ind_next" = "x_ind",
                                 "y_ind_next" = "y_ind"))
  colnames(net_lines) <- sapply(colnames(net_lines), function(name) {
    if (stri_endswith_fixed(name, ".x"))
      stri_replace_first_fixed(name, ".x", "")
    else if (stri_endswith_fixed(name, ".y"))
      stri_replace_first_fixed(name, ".y", "_next")
    else name
  })

  list(net_points = net_points, net_lines = net_lines)
}

#' @importFrom ggplot2 ggplot aes geom_point geom_segment xlab ylab labs ggtitle theme_minimal scale_x_continuous scale_y_continuous
#' @export
plot_kohonen <- function(net_plot_data, input_X,
                         inp_dims_plot = c(1, 2),
                         inp_class = 3) {
  d1 <- inp_dims_plot[1]
  d2 <- inp_dims_plot[2]
  ggplot() +
    geom_point(data = input_X,
               mapping = aes(x = input_X[, d1],
                             y = input_X[, d2],
                             color = as.factor(input_X[, inp_class])),
               size = 1.5) +
    geom_segment(data = net_plot_data$net_lines,
                 mapping = aes(x = net_plot_data$net_lines[[paste0("inp_dim_", d1)]],
                               y = net_plot_data$net_lines[[paste0("inp_dim_", d2)]],
                               xend = net_plot_data$net_lines[[paste0("inp_dim_", d1, "_next")]],
                               yend = net_plot_data$net_lines[[paste0("inp_dim_", d2, "_next")]])) +
    geom_point(data = net_plot_data$net_points,
               mapping = aes(x = net_plot_data$net_points[[paste0("inp_dim_", d1)]],
                             y = net_plot_data$net_points[[paste0("inp_dim_", d2)]]),
               shape = 23,
               size = 2,
               color = "black",
               fill = "white") +
    scale_color_legendary() +
    #scale_x_continuous(limits = c(min(input_X[d1]), c(max(input_X[d1])))) +
    #scale_y_continuous(limits = c(min(input_X[d2]), c(max(input_X[d2])))) +
    xlab(paste0("input dimension ", d1, ": ", colnames(input_X)[d1])) +
    ylab(paste0("input dimension ", d2, ": ", colnames(input_X)[d2])) +
    labs(color = "class") +
    ggtitle(paste0("Kohonen plot ", colnames(input_X)[d1], " vs. ", colnames(input_X)[d2])) +
    theme_minimal()
}

#' @export
plot_kohonen_w_pca <- function(net, input_X, inp_class = ncol(input_X)) {
  N <- net$N
  M <- net$M

  #pca <- prcomp(scale(input_X[, -inp_class]), center = FALSE, rank. = 2)
  #inp <- scale(pca$x)

  # tworzymy ramke danych opisujaca punkty
  net_points <-
    do.call(rbind,lapply(1:M, function(i) t(net$weights[,,i])))

  pca <- prcomp(rbind(input_X[, -inp_class], net_points), scale. = TRUE, rank. = 2)

  net_points <- pca$x[(nrow(input_X) + 1):nrow(pca$x), ] #scale(
    #scale(net_points, center = pca$center, scale = pca$scale)
  inp <- pca$x[1:nrow(input_X), ]
  #net_points %*%
   # pca$rotation #, center = attr(inp, "scaled:center"), scale = attr(inp, "scaled:scale"))

  # generujemy indeksy sieci neuronow na podstawie rozmiaru sieci
  inds <- data.frame(
    x_ind = rep(1:N, M),
    y_ind = rep(1:M, each = N)
  )

  net_points <- cbind(net_points, inds)

  net_lines <- do.call(rbind, lapply(1:N, function(x_ind)
    do.call(rbind, lapply(1:M, function(y_ind) {
      data.frame(x_ind = c(x_ind, x_ind),
                 y_ind = c(y_ind, y_ind),
                 x_ind_next = c(x_ind, x_ind + 1),
                 y_ind_next = c(y_ind + 1, y_ind))
    }))
  )) %>%
    filter(x_ind_next <= N,
           y_ind_next <= M)


  if (net$topology == "hex") {
    net_lines <- net_lines %>%
      filter((x_ind - y_ind) %% 2 == 0 |
               x_ind_next == x_ind + 1)
  }

  net_lines <- net_lines %>%
    left_join(net_points, by = c("x_ind" = "x_ind",
                                 "y_ind" = "y_ind")) %>%
    left_join(net_points, by = c("x_ind_next" = "x_ind",
                                 "y_ind_next" = "y_ind"))

  colnames(net_lines) <- sapply(colnames(net_lines), function(name) {
    if (stri_endswith_fixed(name, ".x"))
      stri_replace_first_fixed(name, ".x", "")
    else if (stri_endswith_fixed(name, ".y"))
      stri_replace_first_fixed(name, ".y", "_next")
    else name
  })

  ggplot() +
    geom_point(data = as.data.frame(inp),
               mapping = aes(x = inp[, 1],
                             y = inp[, 2],
                             color = as.factor(input_X[, inp_class])),
               size = 1.5) +
    geom_segment(data = net_lines,
                 mapping = aes(x = PC1,
                               y = PC2,
                               xend = PC1_next,
                               yend = PC2_next)) +
    geom_point(data = net_points,
               mapping = aes(x = PC1,
                             y = PC2),
               shape = 23,
               size = 2,
               color = "black",
               fill = "white") +
    scale_color_legendary() +
    #scale_x_continuous(limits = c(min(inp[, 1]), c(max(inp[, 1])))) +
    #scale_y_continuous(limits = c(min(inp[, 2]), c(max(inp[, 2])))) +
    xlab("PCA1") +
    ylab("PCA2") +
    labs(color = "class") +
    ggtitle("Kohonen plot PCA") +
    theme_minimal()
}

