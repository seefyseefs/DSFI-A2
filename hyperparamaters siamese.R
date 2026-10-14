# =============================================================================
# Simple hyperparameter tuning for the Siamese network
# 
# We try 3 embedding sizes x 3 learning rates = 9 models.
# We compare them with AUC on the validation set (session 4).
# =============================================================================
library(reticulate)
py_require(python_version = "3.12")
library(keras3)
use_backend("tensorflow")
set.seed(25)
library(pROC)
# -----------------------------------------------------------------------------
hw   <- readRDS("handwriting.rds")
x    <- hw$images / 255                 # scale to [0, 1], keeps [n, 28, 28, 1] shape
meta <- hw$metadata

person  <- meta$person_id               # the "class label" for the Siamese task
session <- meta$session
digit   <- meta$digit

# Indices of images in each split (global row numbers)
train_ids <- which(session %in% 1:3)    # 15,000 images
valid_ids <- which(session == 4)        #  5,000 images
test_ids  <- which(session == 5)        #  5,000 images

# -----------------------------------------------------------------------------
# 2. Build pairs (slide 26), with labels = writer ID instead of digit
# -----------------------------------------------------------------------------
make_pairs <- function(ids, n, class_labels = person) {
  a <- sample(ids, n, replace = TRUE)
  b <- vapply(a, function(i) {
    same <- runif(1) < .5
    pool <- ids[class_labels[ids] == class_labels[i]]
    if (!same) pool <- ids[class_labels[ids] != class_labels[i]]
    pool <- setdiff(pool, i)
    pool[sample.int(length(pool), 1)]   
  }, integer(1))
  cbind(a, b)
}

# Slide 27:
make_batch <- function(p, images = x, class_labels = person) {
  y <- matrix(as.integer(class_labels[p[, 1]] == class_labels[p[, 2]]), ncol = 1)
  list(
    left  = images[p[, 1], , , , drop = FALSE],
    right = images[p[, 2], , , , drop = FALSE],
    y     = y
  )
}

pairs_train <- make_pairs(train_ids, 20000)
pairs_valid <- make_pairs(valid_ids, 5000)
pairs_test  <- make_pairs(test_ids,  5000)

train <- make_batch(pairs_train)
valid <- make_batch(pairs_valid)
test  <- make_batch(pairs_test)

mean(train$y); mean(valid$y); mean(test$y)   # should all be about 0.5

# -----------------------------------------------------------------------------
# 1. A function that builds and trains one  Siamese model.
#    only two things change:
#    emb_dim (size of the embedding) and lr (learning rate).
# -----------------------------------------------------------------------------
train_one_model <- function(emb_dim, lr) {
  
  # Encoder
  image <- layer_input(shape = c(28, 28, 1))
  z <- image |>
    layer_conv_2d(32, 3, activation = "relu", padding = "same") |>
    layer_max_pooling_2d(2) |>
    layer_conv_2d(64, 3, activation = "relu", padding = "same") |>
    layer_max_pooling_2d(2) |>
    layer_conv_2d(64, 3, activation = "relu", padding = "same") |>
    layer_global_average_pooling_2d() |>
    layer_dense(emb_dim)                     # <- hyperparameter 1
  encoder <- keras_model(image, z)
  
  # Siamese model 
  a <- layer_input(shape = c(28, 28, 1))
  b <- layer_input(shape = c(28, 28, 1))
  distance <- list(encoder(a), encoder(b)) |>
    layer_lambda(f = function(z) {
      op_sqrt(op_sum(op_square(z[[1]] - z[[2]]), axis = 2,
                     keepdims = TRUE) + 1e-7)
    })
  model <- keras_model(list(a, b), distance)
  
  contrastive_loss <- function(y, d) {
    op_mean(y * op_square(d) +
              (1 - y) * op_square(op_maximum(1 - d, 0)))
  }
  
  
  
  # Train (slide 31)
  model |> compile(
    optimizer = optimizer_adam(lr),          # <- hyperparameter 2
    loss = contrastive_loss
  )
  model |> fit(
    list(train$left, train$right), train$y,
    epochs = 10,
    batch_size = 64,
    verbose = 0                              
  )
  
  model
}

# -----------------------------------------------------------------------------
# 2. A function that gives the AUC of a model on a set of pairs.
# -----------------------------------------------------------------------------
get_auc <- function(model, pairs) {
  d <- as.numeric(predict(model, list(pairs$left, pairs$right), verbose = 0))
  y <- as.numeric(pairs$y)
  as.numeric(auc(roc(y, -d, direction = "<", quiet = TRUE)))
}

# -----------------------------------------------------------------------------
# 3.  every combination of the two lists
# -----------------------------------------------------------------------------
settings <- expand.grid(
  emb_dim = c(16, 32, 64),
  lr      = c(0.0001, 0.001, 0.01)
)
settings   
# -----------------------------------------------------------------------------
# 4. Train one model per row and record its AUC
# -----------------------------------------------------------------------------
settings$valid_auc <- NA
settings$test_auc  <- NA

for (i in 1:nrow(settings)) {
  cat("Model", i, "of", nrow(settings), "\n")
  set_random_seed(25)                       
  model <- train_one_model(settings$emb_dim[i], settings$lr[i])
  settings$valid_auc[i] <- get_auc(model, valid)
  settings$test_auc[i]  <- get_auc(model, test)
}

settings

# -----------------------------------------------------------------------------
# 5. Pick the best model using VALIDATION AUC, then report its TEST AUC
# -----------------------------------------------------------------------------
best <- settings[which.max(settings$valid_auc), ]
best

# -----------------------------------------------------------------------------
# 6. Plot
# -----------------------------------------------------------------------------
interaction.plot(
  x.factor     = settings$lr,
  trace.factor = settings$emb_dim,
  response     = settings$valid_auc,
  xlab = "Learning rate", ylab = "Validation AUC",
  trace.label = "Embedding size", type = "b", pch = 19
)
