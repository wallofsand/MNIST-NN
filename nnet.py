import numpy as np

class Layer():
    def forward_prop(self, X):
        pass
    def backward_prop(self, dE_dY, alpha=0.05):
        pass

class Convolutional(Layer):
    def __init__(self, num_kernels, kernel_size, input_channels, padding):
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.kernel_channels = input_channels
        self.padding = padding
        self.kernels = np.random.default_rng().standard_normal((num_kernels, kernel_size, kernel_size, input_channels))
    def patches_generator(self, image):
        for h in range(image.shape[0] - self.kernel_size + 1):
            for w in range(image.shape[1] - self.kernel_size + 1):
                patch = image[h*self.kernel_size:(h+1)*self.kernel_size,w*self.kernel_size:(w+1)*self.kernel_size]
                yield patch, h, w
    def forward_prop(self, X):
        return super().forward_prop(X)
    def backward_prop(self, dE_dY, alpha=0.05):
        return super().backward_prop(dE_dY, alpha)

class MaxPooling(Layer):
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
    def patches_generator(self, image):
        yield image
    def forward_prop(self, X):
        return super().forward_prop(X)
    def backward_prop(self, dE_dY, alpha=0.05):
        return super().backward_prop(dE_dY, alpha)

class FullyConnected(Layer):
    def __init__(self, num_inputs, num_outputs):
        self.weights = np.random.default_rng().standard_normal((num_inputs, num_outputs))/num_inputs
        self.biases = np.zeros(num_outputs)
    def forward_prop(self, X):
        return super().forward_prop(X)
    def backward_prop(self, dE_dY, alpha=0.05):
        return super().backward_prop(dE_dY, alpha)
