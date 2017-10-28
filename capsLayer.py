import tensorflow as tf


class CapsConv(object):
    def __init__(self, num_units):
        ''' Capsule convolution.
        Args:
            num_units: the length of the output vector of a capsule.
        '''
        self.num_units = num_units

    def __call__(self, input, num_outputs, kernel_size, strides):
        self.num_outputs = num_outputs
        self.kernel_size = kernel_size
        self.strides = strides


class Routing(object):
    def __init__(self, cfg):
        self.cfg = cfg
