import torch
import numpy as np

device = torch.device('cuda:0')


class conv2d:
    def __init__(self, input_shape, kernel, filters, stride):
        self.filter_size = kernel * kernel
        self.weight_size = self.filter_size * filters

        self.conv2d = torch.nn.Conv2d(input_shape[1], filters, kernel, stride=stride, padding=0, bias=False).cuda()
        self.out_shape = self.conv2d(torch.zeros(input_shape, device=device)).shape
        self.weight_size = np.prod(self.conv2d.weight.data.shape)
        print("Conv2d\n"
              "\tfrom ", input_shape,
              "\n\tto ", self.out_shape,
              "\n\tkernel ", kernel,
              "\n\tstride ", stride,
              "\n\tfilters ", filters, "\n")

    def __call__(self, input, weights):
        self.conv2d.weight.data = weights.view(self.conv2d.weight.data.shape).cuda()
        return self.conv2d(input)


class maxPool2d:
    def __init__(self, input_shape, kernel, stride):
        self.weight_size = 0
        self.pool = torch.nn.MaxPool2d(kernel, stride=stride, padding=1, ceil_mode=True)
        self.out_shape = self.pool(torch.zeros(input_shape, device=device)).shape
        print("maxPool2d\n"
              "\tfrom ", input_shape,
              "\n\tto ", self.out_shape,
              "\n\tkernel ", kernel,
              "\n\tstride ", stride, "\n")

    def __call__(self, input, weights):
        return self.pool(input).cuda()


class flatten:
    def __init__(self, input_shape):
        self.weight_size = 0
        self.out_shape = [input_shape[0], np.prod(input_shape) // input_shape[0]]
        print("Flatten\n"
              "\tfrom ", input_shape,
              "\n\tto ", self.out_shape, "\n")

    def __call__(self, input, weights):
        return input.view(input.size()[0], -1).cuda()


class dense:
    def __init__(self, input_shape, size):
        self.weight_size = np.prod(input_shape) * size
        self.input_s = input_shape
        self.out_shape = [input_shape[0], size]
        self.dense = torch.nn.Linear(input_shape[1], self.out_shape[1], bias=False).cuda()

        print("Dense\n"
              "\tfrom ", input_shape,
              "\n\tto ", self.out_shape, "\n")

    def __call__(self, input, weights):
        self.dense.weight.data = weights.view(self.dense.weight.data.shape).cuda()
        return self.dense(input)



class relu:
    def __init__(self, input_shape):
        self.weight_size = 0
        self.out_shape = input_shape
        print("Relu\non", input_shape,"\n")

    def __call__(self, input, _):
        return torch.nn.functional.relu(input).cuda()

class tanh:
    def __init__(self, input_shape):
        self.weight_size = 0
        self.out_shape = input_shape
        print("Tanh\non", input_shape,"\n")

    def __call__(self, input, _):
        return torch.nn.functional.tanh(input).cuda()

def argmax(input):
    values, indices = input.max(1)
    return indices.cpu().numpy()[0]


class network:
    # parameters is a strin with comma separated layers and dash separated layer parameters
    def __init__(self, input_shape, parameters):
        layer_args = [x.split('-') for x in parameters.split(',')]
        self.weight_size = 0
        self.dense_weight_size = 0
        last_shape = input_shape

        self.layers = []
        for i in range(0, len(layer_args)):
            if layer_args[i][0] == "C":
                layer = conv2d(last_shape, int(layer_args[i][1]), int(layer_args[i][2]), int(layer_args[i][3]))
            elif layer_args[i][0] == 'P':
                layer = maxPool2d(last_shape, int(layer_args[i][1]), int(layer_args[i][2]))  # kernel, stride
            elif layer_args[i][0] == 'D':
                layer = dense(last_shape, int(layer_args[i][1]))
                self.dense_weight_size += layer.weight_size
            elif layer_args[i][0] == 'R':
                layer = relu(last_shape)
            elif layer_args[i][0] == 'T':
                layer = tanh(last_shape)
            elif layer_args[i][0] == 'F':
                layer = flatten(last_shape)

            self.weight_size += layer.weight_size
            last_shape = layer.out_shape
            self.layers.append(layer)

    def __call__(self, input, weights):
        weight_i = 0
        input = torch.tensor(input, dtype=torch.double, device=device).unsqueeze(0)
        for layer in self.layers:
            input = layer(input, torch.from_numpy(weights[weight_i:weight_i + layer.weight_size]))
            weight_i += layer.weight_size
        return argmax(input)
