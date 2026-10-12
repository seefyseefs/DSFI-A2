# Load the file and assign it to a variable named 'my_data'
handwriting <- readRDS("handwriting.rds")
dim(handwriting$images)


library(keras3)
use_backend("tensorflow")
x <- handwriting$images / 255
y_person <- handwriting$metadata$person_id - 1L

# View the imported data frame
View(handwriting)

# 1. Check what kind of object it is (usually a list or data frame)
class(handwriting)

# 2. See a high-level summary of the structure and column names
str(handwriting)



# 1. Check total number of images using the metadata rows
nrow(handwriting$metadata)

# 2. View the first few rows of labels to see the column names
head(handwriting$metadata)

# 3. Check samples per person (should print 1250 for each)
table(handwriting$metadata$person_id)

# 4. Check samples per digit (should print 5000 for each)
table(handwriting$metadata$digit)






library(keras3)
library(tidyverse)
use_backend("tensorflow")

# 1. Load the dataset
data <- readRDS("handwriting.rds")

# 2. Add an explicit row index to metadata for easy image slicing
data$metadata <- data$metadata %>% 
  mutate(img_row_idx = row_number())

# 3. Create a clean visualization function using base R graphics
# This extracts an image index, flips/rotates it to account for R's coordinate mapping, 
# and displays it cleanly inside RStudio.
display_handwriting <- function(index, title_text = "") {
  # Extract the 28x28 2D slice from the 4D Keras tensor
  img_matrix <- data$images[index, , , 1]
  
  # R maps matrices from bottom-left to top-right. 
  # This transposes and reverses rows to orient the text normally.
  rotated_matrix <- t(apply(img_matrix, 2, rev))
  
  # Plot using an inverse grayscale palette (ink is high value/dark, background is light)
  image(rotated_matrix, col = gray.colors(256, rev = TRUE), axes = FALSE, main = title_text)
}


# Find the first image of Digit '0' in Session '1' for the first 4 writers
people_samples <- data$metadata %>%
  filter(digit == 0, session == 1, replicate == 1) %>%
  slice(1:4)

# Set grid layout to 1 row, 4 columns
par(mfrow = c(1, 4))
for(i in 1:nrow(people_samples)) {
  display_handwriting(
    index = people_samples$img_row_idx[i], 
    title_text = paste("Person", people_samples$person_id[i])
  )
}


# Find the first image of each digit (0 through 4) for Person '1' in Session '1'
digit_samples <- data$metadata %>%
  filter(person_id == 1, session == 1, replicate == 1) %>%
  arrange(digit)

par(mfrow = c(1, 5))
for(i in 1:nrow(digit_samples)) {
  display_handwriting(
    index = digit_samples$img_row_idx[i], 
    title_text = paste("Digit", digit_samples$digit[i])
  )
}

# Find the first replicate of Digit '2' by Person '1' across all 5 sessions
session_samples <- data$metadata %>%
  filter(person_id == 1, digit == 2, replicate == 1) %>%
  arrange(session)

par(mfrow = c(1, 5))
for(i in 1:nrow(session_samples)) {
  display_handwriting(
    index = session_samples$img_row_idx[i], 
    title_text = paste("Session", session_samples$session[i])
  )
}


library(keras3)
library(tidyverse)
use_backend("tensorflow")

# ==============================================================================
# 1. LOAD AND SPLIT DATA BY SESSION (FOR THE CNN)
# ==============================================================================
data <- readRDS("handwriting.rds")

# Scale pixel values immediately to [0, 1]
x_all <- data$images / 255
y_all <- data$metadata$person_id - 1  # 0-indexed for Keras

# Identify rows belonging to each session
train_idx <- which(data$metadata$session %in% c(1, 2, 3))
val_idx   <- which(data$metadata$session == 4)
test_idx  <- which(data$metadata$session == 5)

# CNN Inputs
x_train_cnn <- x_all[train_idx, , , , drop = FALSE]
x_val_cnn   <- x_all[val_idx, , , , drop = FALSE]
x_test_cnn  <- x_all[test_idx, , , , drop = FALSE]

y_train_cnn <- y_all[train_idx]
y_val_cnn   <- y_all[val_idx]
y_test_cnn  <- y_all[test_idx]


# ==============================================================================
# 2. CREATE PAIRS FOR THE SIAMESE NETWORK
# ==============================================================================
# Simple function to build pairs without heavy loops
make_simple_pairs <- function(images, metadata) {
  set.seed(42)
  n <- nrow(metadata)
  
  # Step A: Shift indices by a random number to easily pair images
  shift <- sample(1:(n-1), 1)
  idx1  <- 1:n
  idx2  <- c((shift + 1):n, 1:shift) # Shifts the rows over
  
  # Step B: Create Left and Right image tensors
  x1 <- images[idx1, , , , drop = FALSE]
  x2 <- images[idx2, , , , drop = FALSE]
  
  # Step C: Check matching conditions (1 if same person AND same digit, else 0)
  same_person <- (metadata$person_id[idx1] == metadata$person_id[idx2])
  same_digit  <- (metadata$digit[idx1] == metadata$digit[idx2])
  
  # Binary label: 1 for matching style, 0 for different style
  y <- as.integer(same_person & same_digit)
  
  return(list(x1 = x1, x2 = x2, y = as.array(y)))
}

# Generate clean Siamese datasets
cat("Generating Siamese pair inputs...\n")
pairs_train <- make_simple_pairs(x_train_cnn, data$metadata[train_idx, ])
pairs_val   <- make_simple_pairs(x_val_cnn, data$metadata[val_idx, ])
pairs_test  <- make_simple_pairs(x_test_cnn, data$metadata[test_idx, ])





library(tidyverse)

# Create a data frame summarizing your split strategy
split_data <- data.frame(
  Split = c("Train (60%)", "Validation (20%)", "Test (20%)"),
  Session_1 = c(5000, 0, 0),
  Session_2 = c(5000, 0, 0),
  Session_3 = c(5000, 0, 0),
  Session_4 = c(0, 5000, 0),
  Session_5 = c(0, 0, 5000)
) |> 
  pivot_longer(cols = starts_with("Session"), names_to = "Session", values_to = "Count") |> 
  filter(Count > 0) |> 
  mutate(Session = str_replace(Session, "_", " "))

# Plot the split distribution
ggplot(split_data, aes(y = Split, x = Count, fill = Session)) +
  geom_col(color = "white", width = 0.6) +
  scale_fill_brewer(palette = "Blues") +
  scale_x_continuous(labels = scales::comma, expand = c(0, 0), limits = c(0, 16000)) +
  theme_minimal(base_size = 14) +
  labs(
    title = "Chronological Session Split Allocation Strategy",
    subtitle = "Ensuring strict temporal isolation across modeling stages",
    x = "Number of Images",
    y = NULL,
    fill = "Data Source"
  ) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor.x = element_blank(),
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )


# Target a specific digit and recording context
target_digit <- 0
target_session <- 1
target_replicate <- 1

# Pick 4 specific people to compare
selected_people <- c(1, 2, 3, 4)

# Filter metadata to find the corresponding image matrix rows
sampled_rows <- data$metadata |> 
  mutate(original_row = row_number()) |> 
  filter(
    digit == target_digit, 
    session == target_session, 
    replicate == target_replicate,
    person_id %in% selected_people
  )

# Extract and reconstruct the pixel data into a long-form data frame for ggplot
pixel_list <- lapply(1:nrow(sampled_rows), function(i) {
  row_info <- sampled_rows[i, ]
  img_matrix <- data$images[row_info$original_row, , , 1]
  
  # Convert the 2D matrix into long coordinates (X, Y, Intensity)
  as.data.frame(as.table(img_matrix)) |> 
    rename(Y = Var1, X = Var2, Intensity = Freq) |> 
    mutate(
      X = as.numeric(X),
      # Flip the Y axis because R grids render from bottom-up natively
      Y = 29 - as.numeric(Y), 
      person_id = paste("Writer", row_info$person_id)
    )
})

ggplot_pixel_df <- bind_rows(pixel_list)

# Generate a high-contrast tile plot showing style variations
ggplot(ggplot_pixel_df, aes(x = X, y = Y, fill = Intensity)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "gray10") +
  facet_wrap(~ person_id, nrow = 1) +
  theme_void(base_size = 14) +
  labs(
    title = paste("Handwriting Variation for Digit", target_digit),
    subtitle = "Comparing stroke thickness, width, and slant across different writers"
  ) +
  theme(
    strip.text = element_text(face = "bold", margin = margin(b = 5)),
    legend.position = "none",
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, size = 11, color = "gray40")
  )


