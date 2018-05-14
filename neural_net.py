import numpy as np


class conv2d:
    def __init__(self, input_shape, kernel, filers, stride):
        None

    def __call__(self, input, weights):
        None


class maxPool2d:
    def __init__(self, input_shape, kernel, stride):
        None

    def __call__(self, input, weights):
        None


class relu:
    def __init__(self):
        None

    def __call__(self, input):
        None


class flatten:
    def __init__(self):
        None

    def __call__(self, input):
        None


class dense:
    def __init__(self, input_size, size):
        self.weight_size = input_size * size
        self.input_s = input_size
        self.output_s = size

    def __call__(self, input, weights):
        pos = 0
        output = []
        for i in range(0, self.output_s):
            output.append(np.dot(input, weights[pos:pos + self.input_s]))
            pos += self.input_s
        return np.array(output)


class network:
    def __init__(self, input_shape, parameters):
        self.layers = []
        tmp = dense(5, 2)
        self.layers.append([tmp, tmp.weight_size])
        #todo create network and calculate size of weight vecotor
        self.weight_size = 10

    def __call__(self, input, weights):
        return np.random.uniform(0, 100) < 5
