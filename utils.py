import numpy as np
from matplotlib import pyplot as plt
import struct

def test_prediction(network, index, X, Y):
    """
    Show a specific input image and output value.
    """
    current_image = X[:, index, None]
    prediction = network.forward_pass(current_image)
    label = Y[index]
    print("Prediction:", prediction)
    print("Label:", label)
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

class Layer:
    def forward_prop(self, image):
        pass
    def backward_prop(self, dE_dY, alpha=0.05):
        pass

class ConvolutionalLayer(Layer):
    def __init__(self, kernel_num, kernel_size):
        rng = np.random.default_rng()
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.kernels = rng.standard_normal((kernel_num, kernel_size, kernel_size))/(kernel_size**2)

    def patches_generator(self, image):
        """
        Divide the input image in patches to be used during convolution.
        Yields the tuples containing the patches and their coordinates.
        """
        # Extract image height and width
        image_h, image_w = image.shape
        # The number of patches, given an fxf filter is h-f+1 for height and w-f+1 for width
        # For our 3x3 filter and our 28x28 image there are
        # 28 - 3 + 1 = 26 for height and width so 26 * 26 = 676 patches
        for h in range(image_h - self.kernel_size + 1):
            for w in range(image_w - self.kernel_size + 1):
                patch = image[h:(h + self.kernel_size), w:(w + self.kernel_size)]
                yield patch, h, w

    def forward_prop(self, image):
        """
        Perform forward propagation for the convolutional layer.
        """
        self.CL_in = image
        # Extract image height and width
        image_h, image_w = self.CL_in.shape
        # Initialize the convolution output volume of the correct size
        cl_output = np.zeros((image_h-self.kernel_size+1, image_w-self.kernel_size+1, self.kernel_num))
        # Unpack the generator
        for patch, h, w in self.patches_generator(self.CL_in):
            # Perform convolution for each patch
            cl_output[h,w] = np.sum(patch*self.kernels, axis=(1,2))
        return cl_output

    def backward_prop(self, dE_dY, alpha):
        """
        Takes the gradient of the loss function with respect to the output and
        computes the gradients of the loss function with respect to the kernels' weights.
        dE_dY comes from the following layer, typically max pooling layer.
        It updates the kernels' weights
        """
        # Initialize gradient of the loss function with respect to the kernel weights
        dE_dk = np.zeros(self.kernels.shape)
        for patch, h, w in self.patches_generator(self.CL_in):
            for f in range(self.kernel_num):
                dE_dk[f] += patch * dE_dY[h, w, f]
        # Update the parameters
        self.kernels -= alpha*dE_dk
        return dE_dk

class MaxPoolingLayer(Layer):
    def __init__(self, kernel_size):
        """
        Constructor takes as input kernel size (n of an nxn kernel)
        """
        self.n = kernel_size

    def patches_generator(self, image):
        """
        Divide the input image in patches to be used during convolution.
        Yields the tuples containing the patches and their coordinates.
        """
        for h in range(image.shape[0]//self.n):
            for w in range(image.shape[1]//self.n):
                patch = image[h*self.n:(h+1)*self.n-1, w*self.n:(w+1)*self.n-1]
                yield patch, h, w

    def forward_prop(self, image):
        # Store for backprop
        self.MPL_in = image
        image_h, image_w, num_k = self.MPL_in.shape
        output = np.zeros((image_h//self.n, image_w//self.n, num_k))
        for patch, h, w in self.patches_generator(self.MPL_in):
            output[h, w] = np.amax(patch, axis=(0, 1))
        return output

    def backward_prop(self, dE_dY, alpha):
        """
        Takes the gradient of the loss function with respect to the output and
        computes the gradients of the loss function with respect to the kernels' weights.
        dE_dY comes from the following layer, typically softmax.
        There are no weights to update; output is used to update the weights of the previous layer.
        """
        dE_dk = np.zeros(self.MPL_in.shape)
        for patch, h, w in self.patches_generator(self.MPL_in):
            image_h, image_w, num_k = patch.shape
            max_val = np.amax(patch, axis=(0,1))
            for idx_h in range(image_h):
                for idx_w in range(image_w):
                    for idx_k in range(num_k):
                        if patch[idx_h, idx_w, idx_k] == max_val[idx_k]:
                            dE_dk[h*self.n+idx_h, w*self.n+idx_w, idx_k] = dE_dY[h, w, idx_k]
            return dE_dk

class SoftmaxLayer(Layer):
    def __init__(self, input_units, output_units):
        """
        Initiallize weights and biases.
        """
        rng = np.random.default_rng()
        self.w = rng.standard_normal((input_units, output_units))/input_units
        self.b = np.zeros(output_units)

    def forward_prop(self, image):
        # Store for backprop
        self.input_shape = image.shape
        self.input_flat = image.flatten()
        # W * x + b
        self.output = self.input_flat.dot(self.w) + self.b
        softmax_activation = np.exp(self.output) / np.sum(np.exp(self.output), axis=0)
        return softmax_activation

    def backward_prop(self, dE_dY, alpha=0.05):
        # loss (dE_dY) will be 0 for most cases but...
        for i, gradient in enumerate(dE_dY):
            # ... will be some float when i is the label index
            if gradient == 0:
                continue
            # Compute exponential of output ie. partial softmax activation
            trans_eq = np.exp(self.output)
            S_total = np.sum(trans_eq)
            # Compute gradients with respect to output (Z)
            dY_dZ = -trans_eq[i]*trans_eq / (S_total**2)
            dY_dZ[i] = trans_eq[i]*(S_total - trans_eq[i]) / (S_total**2)
            # Compute gradients of Z with respect to weights, bias, input
            dZ_dw = self.input_flat
            dZ_db = 1
            dZ_dX = self.w
            # Gradient of loss with respect to output...
            dE_dZ = gradient * dY_dZ
            dE_dw = dZ_dw[np.newaxis].T @ dE_dZ[np.newaxis]
            dE_db = dE_dZ * dZ_db
            dE_dX = dZ_dX @ dE_dZ
            # Update parameters
            self.w -= dE_dw * alpha
            self.b -= dE_db * alpha
            return dE_dX.reshape(self.input_shape)

class Network:
    def __init__(self, layers:list):
        self.layers = layers

    def forward_pass(self, image, label):
        """
        Pass an image through the model without training.
        """
        output = image/255.
        for layer in self.layers:
            output = layer.forward_prop(output)
        # Compute loss and accuracy
        loss = -np.log(output[label])
        accuracy = 1 if np.argmax(output) == label else 0
        return output, loss, accuracy

    def backprop(self, gradient, alpha=0.05):
        gradient_back = gradient
        # print(gradient_back)
        for layer in reversed(self.layers):
            gradient_back = layer.backward_prop(gradient_back, alpha)
            # print(gradient_back)
        return gradient_back

    def train(self, image, label, alpha=0.05):
        output, loss, accuracy = self.forward_pass(image, label)
        gradient = np.zeros(10)
        gradient[label] = -1/output[label]
        gradient_back = self.backprop(gradient, alpha)
        return loss, accuracy

    def test(self, X, Y):
        """
        Test the model on a set of data.
        Doesn't train the model.
        """
        loss = 0
        accuracy = 0
        print("Begin test . . .")
        for i, (image, label) in enumerate(zip(X, Y)):
            output, loss_1, accuracy_1 = self.forward_pass(image, label)
            loss += loss_1
            accuracy += accuracy_1
        print('Test accuracy {}'.format(100*accuracy/len(Y)))

def CNN_training(X, Y, display=False):
    num_filters = 16
    CL = ConvolutionalLayer(num_filters, 3) # layer with 2 3x3 filters, output (26,26,2)
    MPL = MaxPoolingLayer(2) # pooling layer 2x2, output (13,13,2)
    SmL = SoftmaxLayer(13*13*num_filters, 10) # softmax layer with 13*13*2 inputs and 10 outputs
    CNN = Network(layers=[CL,MPL,SmL])

    for epoch in range(4):
        print('Epoch {} ->'.format(epoch+1))
        # Shuffle training data
        permutation = np.random.permutation(1500)
        X = X[permutation]
        Y = Y[permutation]
        loss = 0
        accuracy = 0
        for i, (image, label) in enumerate(zip(X, Y)):
            # print a snapshot of the training
            if i % 100 == 0:
                print("Step {}. For the last 100 steps: average loss {}, accuracy {}".format(i+1, loss/100, accuracy))
                loss = 0
                accuracy = 0
            if display == True and i % 1000 == 0:
                current_image = image/255.
                CL_output = CL.forward_prop(current_image)
                MPL_output = MPL.forward_prop(CL_output)
                prediction = SmL.forward_prop(MPL_output)
                fig, ax = plt.subplots(num_filters, 4)
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
            loss_1, accuracy_1 = CNN.train(image, label)
            loss += loss_1
            accuracy += accuracy_1
    return CNN
