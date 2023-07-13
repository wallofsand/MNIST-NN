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

    CvL1 = ConvolutionalLayer(4, 3, 1) # layer with 4 3x3 filters, output (26,26,4)
    MPL1 = MaxPoolingLayer(2) # pooling layer 2x2, output (13,13,4)
    SmL1 = SoftmaxLayer(13*13*4, 10) # softmax layer with 5*5*4 inputs and 10 outputs
    CNN = Network([CvL1,MPL1,SmL1])

    # CvL1 = ConvolutionalLayer(4, 3, 1) # layer with 4 3x3x1 filters, output (26,26,4)
    # MPL1 = MaxPoolingLayer(2) # pooling layer 2x2, output (13,13,4)
    # CvL2 = ConvolutionalLayer(4, 2, 4) # layer with 4 2x2x4 filters, output (12,12,4)
    # MPL2 = MaxPoolingLayer(2) # pooling layer 2x2, output (6,6,4)
    # SmL1 = SoftmaxLayer(6*6*4, 10) # softmax layer with 5*5*4 inputs and 10 outputs
    # CNN = Network([CvL1,MPL1,CvL2,MPL2,SmL1]) # 71.67% test accuracy

    # train_data, train_labels, test_data, test_labels = dataset() # input (28,28)
    CNN.CNN_training(*dataset())

    # idx = 1
    # current_image = test_data[idx]/255.
    # CvL1_output = CNN.layers[0].forward_prop(current_image[:,:,np.newaxis])
    # MPL1_output = CNN.layers[1].forward_prop(CvL1_output)
    # CvL2_output = CNN.layers[2].forward_prop(MPL1_output)
    # MPL2_output = CNN.layers[3].forward_prop(CvL2_output)
    # # prediction = CNN.layers[2].forward_prop(MPL_output)
    # fig, ax = plt.subplots(4, 4)
    # # Image
    # # ax[0][0].imshow(current_image, cmap='gray', interpolation='nearest')
    # # Convolutions
    # for i in range(CvL1_output.shape[2]):
    #     ax[i][0].imshow(CvL1_output[:,:,i], cmap='gray', interpolation='nearest')
    # # Max Pooling
    # for i in range(MPL1_output.shape[2]):
    #     ax[i][1].imshow(MPL1_output[:,:,i], cmap='gray', interpolation='nearest')
    # # Convolutions
    # for i in range(CvL2_output.shape[2]):
    #     ax[i][2].imshow(CvL2_output[:,:,i], cmap='gray', interpolation='nearest')
    # # Max Pooling
    # for i in range(MPL2_output.shape[2]):
    #     ax[i][3].imshow(MPL2_output[:,:,i], cmap='gray', interpolation='nearest')
    # # Softmax
    # # ax[0][3].imshow(prediction[:,np.newaxis], cmap='gray', interpolation='nearest')
    # plt.show()

if __name__ == "__main__":
    main()

    # i = np.zeros((3,3,2))
    # i[0,1] = [1,10]
    # i[0,2] = [2,20]
    # i[1,0] = [3,30]
    # i[1,1] = [4,40]
    # i[1,2] = [5,50]
    # i[2,0] = [6,60]
    # i[2,1] = [7,70]
    # i[2,2] = [8,80]

    # print(i)
    # print(np.sum(i, axis=2))

    # print('AX0', np.sum(i, axis=0))
    # print(i[0,0]+i[1,0]+i[2,0])
    # print(i[0,1]+i[1,1]+i[2,1])
    # print(i[0,2]+i[1,2]+i[2,2])
    # print('AX1', np.sum(i, axis=1))
    # print('AX2', np.sum(i, axis=2))
    # print(i[0,0]+i[0,1]+i[0,2])
    # print(i[1,0]+i[1,1]+i[1,2])
    # print(i[2,0]+i[2,1]+i[2,2])