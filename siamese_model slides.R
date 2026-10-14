# =============================================================================

# Split by session: train = sessions 1-3, validation = session 4, test = session 5
# =============================================================================
library(caret)
library(pROC)
library(reticulate)
py_require(python_version = "3.12")
library(keras3)
use_backend("tensorflow")
set.seed(25)

# -----------------------------------------------------------------------------
# 1. Load data
# handwriting.rds is a list with:
#   $images   : integer array [25000, 28, 28, 1], values 0-255
#   $metadata : data.frame with image_id, person_id, digit, session, replicate
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
# 3. Shared encoder (slide 28)
# -----------------------------------------------------------------------------
image <- layer_input(shape = c(28, 28, 1))
z <- image |>
  layer_conv_2d(32, 3, activation = "relu", padding = "same") |>
  layer_max_pooling_2d(2) |>
  layer_conv_2d(64, 3, activation = "relu", padding = "same") |>
  layer_max_pooling_2d(2) |>
  layer_conv_2d(64, 3, activation = "relu", padding = "same") |>
  layer_global_average_pooling_2d() |>
  layer_dense(32)
encoder <- keras_model(image, z)

# -----------------------------------------------------------------------------
# 4. Siamese model: both images go through the SAME encoder (slide 29)
# -----------------------------------------------------------------------------
a <- layer_input(shape = c(28, 28, 1))
b <- layer_input(shape = c(28, 28, 1))
distance <- list(encoder(a), encoder(b)) |>
  layer_lambda(f = function(z) {
    op_sqrt(op_sum(op_square(z[[1]] - z[[2]]), axis = 2,
                   keepdims = TRUE) + 1e-7)
  })
model <- keras_model(list(a, b), distance)

# -----------------------------------------------------------------------------
# 5. Contrastive loss, margin m = 1 (slide 30)
#   L = y * D^2 + (1 - y) * max(0, m - D)^2
# -----------------------------------------------------------------------------
contrastive_loss <- function(y, d) {
  op_mean(y * op_square(d) +
            (1 - y) * op_square(op_maximum(1 - d, 0)))
}

# -----------------------------------------------------------------------------
# 6. Train (slide 31) + early stopping on session-4 validation loss
# -----------------------------------------------------------------------------
model |> compile(
  optimizer = optimizer_adam(1e-3),
  loss = contrastive_loss
)

history <- model |> fit(
  list(train$left, train$right), train$y,
  validation_data = list(
    list(valid$left, valid$right), valid$y
  ),
  epochs = 20,
  batch_size = 64,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 3,
                            restore_best_weights = TRUE)
  )
)
plot(history)

# -----------------------------------------------------------------------------
# 7. Evaluate matching performance

# -----------------------------------------------------------------------------

# choosing the threshold

d_valid <- as.numeric(predict(model, list(valid$left, valid$right)))

valid_same <- valid_same <- as.logical(valid$y[, 1])

balanced_accuracy_at <- function(cut) {
  predicts_same <- d_valid < cut
  same_rate <- mean(predicts_same[valid_same])
  different_rate <- mean(!predicts_same[!valid_same])
  (same_rate + different_rate) / 2
}

example_thresholds <- unique(as.numeric(
  quantile(d_valid, c(.25, .5, .75))
))
data.frame(
  threshold = example_thresholds,
  balanced_accuracy = vapply(example_thresholds, balanced_accuracy_at, numeric(1))
)

threshold_candidates <- sort(unique(d_valid))
threshold_scores <- vapply(threshold_candidates, balanced_accuracy_at, numeric(1))
best_index <- which.max(threshold_scores)
threshold <- threshold_candidates[best_index]
data.frame(threshold = threshold, balanced_accuracy = threshold_scores[best_index])


d_test  <- as.numeric(predict(model, list(test$left,  test$right)))

y_valid <- as.numeric(valid$y)
y_test  <- as.numeric(test$y)

# 7b. Test-set metrics threshold
pred_same <- as.integer(d_test < threshold)
conf <- table(Predicted = factor(pred_same, levels = 0:1, labels = c("different", "same")),
              Actual    = factor(y_test,    levels = 0:1, labels = c("different", "same")))
print(conf)


acc <- mean(pred_same == y_test) #accuracy
tpr <- conf["same", "same"] / sum(conf[, "same"])                # true match rate (Sensitivity/recall)
fpr <- conf["same", "different"] / sum(conf[, "different"])        # false match rate
fnr <- 1 - tpr                                                     # false non-match rate
f1 <- 2 * (precision * tpr) / (precision + tpr)

# AUC 
auc_from_dist <- function(d, y) {
  s <- -d                                  # smaller distance = more "same"
  r <- rank(s)
  n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
auc <- auc_from_dist(d_test, y_test)
auc

# 7c. ROC curve
roc_pts <- t(sapply(sort(unique(c(0, d_test, Inf))), function(t) {
  c(fpr = mean(d_test[y_test == 0] < t), tpr = mean(d_test[y_test == 1] < t))
}))
plot(roc_pts[, "fpr"], roc_pts[, "tpr"], type = "l", lwd = 2,
     xlab = "False match rate", ylab = "True match rate",
     main = sprintf("Siamese ROC, test session 5 (AUC = %.3f)", auc))
abline(0, 1, lty = 2, col = "grey")

