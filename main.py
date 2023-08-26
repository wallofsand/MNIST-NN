from utils import *
import draw

def main():
    # CvL1 = ConvolutionalLayer(8, 3, 1) # layer with 8 3x3x1 filters, output (26,26,8)
    # MPL1 = MaxPoolingLayer(2) # pooling layer 2x2, output (13,13,8)
    # SmL1 = SoftmaxLayer(13*13*8, 10) # softmax layer with 13*13*8 inputs and 10 outputs
    # CNN = Network([CvL1,MPL1,SmL1])

    # CvL1 = ConvolutionalLayer(8, 3, 1) # layer with 8 3x3x1 filters, output (26,26,8)
    # MPL1 = MaxPoolingLayer(2) # pooling layer 2x2, output (13,13,8)
    # ReLu = ReLuLayer()
    # CvL2 = ConvolutionalLayer(8, 4, 8) # layer with 8 4x4x8 filters, output (10,10,8)
    # MPL2 = MaxPoolingLayer(2) # pooling layer 2x2, output (5,5,8)
    # SmL1 = SoftmaxLayer(5*5*8, 10) # softmax layer with 5*5*8=200 inputs and 10 outputs
    # CNN = Network([CvL1,MPL1,ReLu,CvL2,MPL2,SmL1])

    train_data, train_labels, test_data, test_labels = dataset() # input (28,28)
    # CNN.CNN_training(train_data, train_labels)

    # CNN.save('network')
    ld = Network([])
    ld.load('network.npz')

    # ld.test(test_data, test_labels)

    draw.build_buttons(ld)
    draw.loop()

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
