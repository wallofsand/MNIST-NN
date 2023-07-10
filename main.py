import numpy as np
from matplotlib import pyplot as plt
import struct
from utils import *

def dataset():
    # MNIST digit set:
    FILE_DIGIT_TRAIN_IMG = 'MNIST/train-images-idx3-ubyte'
    FILE_DIGIT_TRAIN_LBL = 'MNIST/train-labels-idx1-ubyte'
    FILE_DIGIT_TEST_IMG = 'MNIST/t10k-images-idx3-ubyte'
    FILE_DIGIT_TEST_LBL = 'MNIST/t10k-labels-idx1-ubyte'
    FILE_TRAIN_IMG = FILE_DIGIT_TRAIN_IMG
    FILE_TRAIN_LBL = FILE_DIGIT_TRAIN_LBL
    FILE_TEST_IMG = FILE_DIGIT_TEST_IMG
    FILE_TEST_LBL = FILE_DIGIT_TEST_LBL
    # open the training file
    with open(FILE_TRAIN_IMG, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = np.reshape(train_data, (size, nrows, ncols))
    # open the label file for validation
    with open(FILE_TRAIN_LBL,'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    # open the testing file
    with open(FILE_TEST_IMG, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = np.reshape(test_data, (size, nrows, ncols))
    # open the testing label file for validation
    with open(FILE_TEST_LBL,'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    return train_data, train_labels, test_data, test_labels

def main():
    train_data, train_labels, test_data, test_labels = dataset()
    CNN = CNN_training(train_data, train_labels, display=False)
    CNN.test(test_data, test_labels)
    idx = 0
    current_image = test_data[0]/255.
    CL_output = CNN.layers[0].forward_prop(current_image)
    MPL_output = CNN.layers[1].forward_prop(CL_output)
    prediction = CNN.layers[2].forward_prop(MPL_output)
    fig, ax = plt.subplots(16, 4)
    # Image
    ax[0][0].imshow(current_image, cmap='gray', interpolation='nearest')
    # Convolutions
    for i in range(CL_output.shape[2]):
        ax[i][1].imshow(CL_output[:,:,i], cmap='gray', interpolation='nearest')
    # Max Pooling
    for i in range(MPL_output.shape[2]):
        ax[i][2].imshow(MPL_output[:,:,i], cmap='gray', interpolation='nearest')
    # Softmax
    ax[0][3].imshow(prediction[:,np.newaxis], cmap='gray', interpolation='nearest')
    plt.show()

if __name__ == "__main__":
    main()
