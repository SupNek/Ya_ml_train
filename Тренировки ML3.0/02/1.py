# do not change the code in the block below
# __________start of block__________
import numpy as np
def compute_sobel_gradients_two_loops(image):
    # Get image dimensions
    height, width = image.shape

    # Initialize output gradients
    gradient_x = np.zeros_like(image, dtype=np.float64)
    gradient_y = np.zeros_like(image, dtype=np.float64)

    # Pad the image with zeros to handle borders
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0)
# __________end of block__________

    # Define the Sobel kernels for X and Y gradients
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Apply Sobel filter for X and Y gradients using convolution
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            gradient_x[i-1][j-1] = (padded_image[np.ix_([i-1, i, i+1], [j-1, j, j+1])] * sobel_x).sum()
            gradient_y[i-1][j-1] = (padded_image[np.ix_([i-1, i, i+1], [j-1, j, j+1])] * sobel_y).sum()
    return gradient_x, gradient_y

import numpy as np # for your convenience when you copy the code to the contest
def compute_gradient_magnitude(sobel_x, sobel_y):
    '''
    Compute the magnitude of the gradient given the x and y gradients.

    Inputs:
        sobel_x: numpy array of the x gradient.
        sobel_y: numpy array of the y gradient.

    Returns:
        magnitude: numpy array of the same shape as the input [0] with the magnitude of the gradient.
    '''
    return np.sqrt(sobel_x ** 2 + sobel_y ** 2)


def compute_gradient_direction(sobel_x, sobel_y):
    '''
    Compute the direction of the gradient given the x and y gradients. Angle must be in degrees in the range (-180; 180].
    Use arctan2 function to compute the angle.

    Inputs:
        sobel_x: numpy array of the x gradient.
        sobel_y: numpy array of the y gradient.

    Returns:
        gradient_direction: numpy array of the same shape as the input [0] with the direction of the gradient.
    '''
    return np.rad2deg(np.arctan2(sobel_y, sobel_x))

cell_size = 7
def compute_hog(image, pixels_per_cell=(cell_size, cell_size), bins=9):
    # 1. Convert the image to grayscale if it's not already (assuming the image is in RGB or BGR)
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)  # Simple averaging to convert to grayscale
    
    # 2. Compute gradients with Sobel filter
    gradient_x, gradient_y = compute_sobel_gradients_two_loops(image)

    # 3. Compute gradient magnitude and direction
    magnitude = compute_gradient_magnitude(gradient_x, gradient_y)
    direction = compute_gradient_direction(gradient_x, gradient_y)
    # 4. Create histograms of gradient directions for each cell
    cell_height, cell_width = pixels_per_cell
    n_cells_x = image.shape[1] // cell_width
    n_cells_y = image.shape[0] // cell_height

    histograms = np.zeros((n_cells_y, n_cells_x, bins))
    for i in range(n_cells_y):
      for j in range(n_cells_x):
        # for x in range(j * cell_width, (j+1) * cell_width):
        #   for y in range(i * cell_height, (i+1) * cell_height):
        #     histograms[i][j][int(direction[x][y] + 179.999999) // (360 // bins)] += magnitude[x][y]
        histograms[i][j] = np.histogram(direction[i * cell_height : (i+1) * cell_height, j * cell_width : (j+1) * cell_width], 
                                        bins=np.linspace(-180, 180, bins+1),
                                        # bins=bins,
                                        weights=magnitude[i * cell_height : (i+1) * cell_height, j * cell_width : (j+1) * cell_width]
                                        )[0]

        if histograms[i][j].sum():
            histograms[i][j] /= histograms[i][j].sum()

    return histograms