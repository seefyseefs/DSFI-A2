# Load the file and assign it to a variable named 'my_data'
handwriting <- readRDS("handwriting.rds")

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


# 1. Print the image size (e.g., 28 means 28x28 pixels)
img_size <- handwriting$image_scale
print(img_size)

# 2. Extract the very first image row from the matrix
first_img_vector <- handwriting$images[1, ]

# 3. Reshape it into a square matrix using the scale
img_matrix <- matrix(first_img_vector, nrow = img_size, ncol = img_size)

# 4. Rotate it so it displays right-side up in R
img_matrix <- t(img_matrix[, ncol(img_matrix):1])

# 5. Plot the digit!
image(img_matrix, col = gray.colors(256), axes = FALSE)
