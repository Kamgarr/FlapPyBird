import numpy as np


class conv2d:
    def __init__(self, input_shape, kernel, filters, stride, activation=None):
        self.in_shape = input_shape
        self.kernel = kernel
        self.stride = stride
        self.filters = filters
        self.activation = activation
        self.out_shape = [input_shape[0] // stride, input_shape[1] // stride, filters]
        if len(input_shape) < 3:
            self.weight_size = kernel * kernel * filters
        else:
            self.weight_size = kernel * kernel * filters * input_shape[2]

    def __call__(self, input, weights):
        output = np.ndarray(shape=(self.out_shape[0], self.out_shape[1], self.filters), dtype=float)
        pos = 0

        for f in range(0, self.filters):
            filter = weights[pos:pos + self.kernel * self.kernel]
            for i in range(0, self.out_shape[0]):
                for j in range(0, self.out_shape[1]):
                    input_sub = input[i * self.stride:i * self.stride + self.kernel,
                                           j * self.stride:j * self.stride + self.kernel].flatten()
                    if len(input_sub) == len(filter):
                        output[i][j][f] = np.dot(input_sub, filter)
            pos += self.kernel * self.kernel
        return output


class maxPool2d:
    def __init__(self, input_shape, kernel, stride):
        self.in_shape = input_shape
        self.kernel = kernel
        self.stride = stride
        self.weight_size = 0
        self.out_shape = [input_shape[0] // stride, input_shape[1] // stride]

    def __call__(self, input, weights):
        output = np.ndarray(shape=(self.out_shape[0], self.out_shape[1]), dtype=float)
        for i in range(0, self.out_shape[0]):
            for j in range(0, self.out_shape[1]):
                output[i][j] = np.amax(input[i * self.stride:i * self.stride + self.kernel, j * self.stride:j * self.stride + self.kernel])
        return output

class flatten:
    def __init__(self, input_shape):
        self.weight_size = 0
        self.out_shape = [np.prod(input_shape)]

    def __call__(self, input, weights):
        return input.flatten()


class dense:
    def __init__(self, input_size, size, activation=None):
        self.weight_size = input_size * size
        self.input_s = input_size
        self.out_shape = [size]
        self.activation = activation

    def __call__(self, input, weights):
        pos = 0
        output = []
        for i in range(0, self.out_shape[0]):
            if self.activation:
                output.append(self.activation(np.dot(input, weights[pos:pos + self.input_s])))
            else:
                output.append(np.dot(input, weights[pos:pos + self.input_s]))
            pos += self.input_s
        return np.array(output)


def relu(x):
    np.maximum(x, 0)
    return x


class network:
    # parameters is a strin with comma separated layers and dash separated layer parameters
    def __init__(self, input_shape, parameters):
        layer_args = [x.split('-') for x in parameters.split(',')]
        self.weight_size = 0
        last_shape = input_shape

        self.layers = []
        for i in range(0, len(layer_args)):
            if layer_args[i][0] == "C":
                layer = conv2d(last_shape, int(layer_args[i][1]), int(layer_args[i][2]), int(layer_args[i][3]))
            elif layer_args[i][0] == "CR":
                layer = conv2d(last_shape, int(layer_args[i][1]), int(layer_args[i][2]), int(layer_args[i][3]), relu)
            elif layer_args[i][0] == 'P':
                layer = maxPool2d(last_shape, int(layer_args[i][1]), int(layer_args[i][2]))
            elif layer_args[i][0] == 'D':
                layer = dense(last_shape[0], int(layer_args[i][1]))
            elif layer_args[i][0] == 'DR':
                layer = dense(last_shape[0], int(layer_args[i][1]), relu)
            elif layer_args[i][0] == 'F':
                layer = flatten(last_shape)

            self.weight_size += layer.weight_size
            last_shape = layer.out_shape
            self.layers.append(layer)

    def __call__(self, input, weights):
        weight_i = 0
        for layer in self.layers:
            input = layer(input, weights[weight_i:weight_i+layer.weight_size])
            weight_i += layer.weight_size
        return input[0] < 0
