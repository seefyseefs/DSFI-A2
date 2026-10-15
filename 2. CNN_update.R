
#reticulate::py_require(python_version = "3.11")
library(keras3)
use_backend("tensorflow", gpu = FALSE)
set.seed(2026)

# 1. Load and scale the images. They already have shape N x 28 x 28 x 1.
data <- readRDS("handwriting.rds")
x <- data$images / 255
writer_label <- data$metadata$person_id - 1L

# 2. Split by session; all digits and writers appear in each split.
train <- which(data$metadata$session %in% 1:3)
validation <- which(data$metadata$session == 4)
test <- which(data$metadata$session == 5)

x_train <- x[train, , , , drop = FALSE]
x_validation <- x[validation, , , , drop = FALSE]
x_test <- x[test, , , , drop = FALSE]

# 3. One-hot encode the writer labels, following the lecturer's approach.
y_train <- to_categorical(writer_label[train], 20)
y_validation <- to_categorical(writer_label[validation], 20)
actual_writer <- data$metadata$person_id[test]

# 4. Define the lecturer's CNN structure with 20 output classes.

input <- layer_input(shape = c(28,28,1))

output <- input |>
  layer_conv_2d(32, 3, padding = "same") |>
  layer_activation("relu") |>
  layer_max_pooling_2d(2) |>
  
  layer_conv_2d(64, 3, padding = "same") |>
  layer_activation("relu") |>
  layer_max_pooling_2d(2) |>
  
  layer_flatten() |>
  layer_dense(128, activation = "relu") |>
  layer_dropout(0.5) |>
  layer_dense(20, activation = "softmax")
model <- keras_model(inputs = input, outputs = output)

# 5. Compile: categorical cross-entropy matches the one-hot labels.
compile(model,
        loss = "categorical_crossentropy",
        optimizer = optimizer_adam(learning_rate = 0.0001),
        metrics = "accuracy",
        jit_compile = FALSE)

# 6. Train; use session 4 explicitly for validation.
early_stop <- callback_early_stopping(
  monitor = "val_loss",
  patience = 3,
  restore_best_weights = TRUE
)

history <- fit(model,
               x_train, y_train,
               epochs = 10, batch_size = 64,
               validation_data = list(x_validation, y_validation),
               shuffle = TRUE, verbose = 2,
               callbacks = list(early_stop),
               view_metrics = FALSE)

plot(history)

# 7. Test writer identification accuracy.
probabilities <- predict(model, x_test, verbose = 0)

# Columns 1-20 map to the original writer IDs 1-20.
predicted_writer <- max.col(probabilities, ties.method = "first")
correct <- predicted_writer == actual_writer
cat(sprintf("Overall test writer accuracy: %.2f%%\n", 100 * mean(correct)))

# 8. Compare accuracy by the digit in the image, using the same model.
digit_test <- data$metadata$digit[test]
results_by_digit <- data.frame(digit = 0:4, writer_accuracy_percent = NA_real_)
for (d in 0:4) {
  rows <- digit_test == d
  results_by_digit$writer_accuracy_percent[d + 1] <- 100 * mean(correct[rows])
}
print(results_by_digit)

# 9. Confusion matrix
library(caret)
cm <- confusionMatrix(
  factor(predicted_writer),
  factor(actual_writer)
)

cm

