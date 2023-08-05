import numpy as np

def patches_generator(image):
    """
    Divide the input image in patches to be used during convolution.
    Yields the tuples containing the patches and their coordinates.
    """
    for i in range(image.shape[0]//2):
        for j in range(image.shape[1]//2):
            patch = image[i*2:(i+1)*2,j*2:(j+1)*2]
            yield patch, i, j

def process(image):
    output = np.ndarray((image.shape[0]//2,image.shape[1]//2))
    for patch, i, j in patches_generator(image):
        output[i,j] = np.sum(patch) / patch.size
    return output
