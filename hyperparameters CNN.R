
#reticulate::py_require(python_version = "3.11")
library(ggplot2)
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

# -----------------------------------------------------------------------------
# 1. A function that builds and trains one CNN model.
#    only two things change:
#    Dense layer size (similar to embedding size in Siamese) and lr (learning rate).
# -----------------------------------------------------------------------------

train_one_cnn <- function(dense_units, lr) {
  
  # Encoder
  input <- layer_input(shape = c(28, 28, 1))
  
  output <- input |>
    layer_conv_2d(32, 3, padding = "same") |>
    layer_activation("relu") |>
    layer_max_pooling_2d(2) |>
    
    layer_conv_2d(64, 3, padding = "same") |>
    layer_activation("relu") |>
    layer_max_pooling_2d(2) |>
    
    layer_flatten() |>
    
    layer_dense(
      units = dense_units,      # Hyperparameter 1
      activation = "relu") |>
    
    layer_dropout(0.5) |>
    
    layer_dense(units = 20,
                activation = "softmax")
  
  model <- keras_model(input, output)
  
  # Train
  model |>
    compile(loss = "categorical_crossentropy",
            optimizer = optimizer_adam(learning_rate = lr),      # Hyperparameter 2
            metrics = "accuracy")
  
  early_stop <- callback_early_stopping(
    monitor = "val_loss",
    patience = 3,
    restore_best_weights = TRUE)
  
  history <- model |>
    fit(x_train,
        y_train,
        epochs = 10,
        batch_size = 64,
        validation_data = list(x_validation, y_validation),
        callbacks = list(early_stop),
        verbose = 0)
  
  list(model = model, history = history)
}

#Function to calculate accuracy

get_accuracy <- function(model) {
  
  probs <- predict(model, x_test, verbose = 0)
  
  pred_writer <- max.col(probs,
                         ties.method = "first")
  
  mean(pred_writer == actual_writer)
}

#Hyperparameter grid

settings <- expand.grid(dense_units = c(64, 128, 256),
                        lr = c(0.0001, 0.0005, 0.001))
settings

#train every combination

settings$val_accuracy <- NA
settings$test_accuracy <- NA

for(i in 1:nrow(settings)) {
  
  cat("Model", i, "of", nrow(settings), "\n")
  
  set.seed(2026)
  
  fit_obj <- train_one_cnn(dense_units = settings$dense_units[i],
    lr = settings$lr[i])
  
  model <- fit_obj$model
  history <- fit_obj$history
  
  settings$val_accuracy[i] <- max(history$metrics$val_accuracy)
  
  settings$test_accuracy[i] <- get_accuracy(model)
}

settings

#plot settings
ggplot(settings,
       aes(x = factor(lr),
           y = val_accuracy,
           fill = factor(dense_units))) +
  geom_col(position = "dodge") +
  labs(
    x = "Learning Rate",
    y = "Validation Accuracy",
    fill = "Dense Units"
  )

#pick the best model

best <- settings[which.max(settings$val_accuracy),]
best

#plot

interaction.plot(x.factor = settings$lr,
                 trace.factor = settings$dense_units,
                 response = settings$val_accuracy,
                 xlab = "Learning Rate",
                 ylab = "Validation Accuracy",
                 trace.label = "Dense Units",
                 type = "b",
                 pch = 19)