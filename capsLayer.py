import tensorflow as tf

from config import cfg


class CapsConv(object):
    def __init__(self, num_units, with_routing=True):
        ''' Capsule convolution.
        Args:
            num_units: the length of the output vector of a capsule.
        '''
        self.num_units = num_units
        self.with_routing = with_routing

    def __call__(self, input, num_outputs, kernel_size=None, strides=None):
        self.num_outputs = num_outputs
        self.kernel_size = kernel_size
        self.strides = strides

        if self.with_routing:
            # the DigitCaps layer
            # Return a list with 10 capsule output, each capsule is a tensor
            # with shape [batch_size, 1, 16, 1]

            # Reshape the input into shape [batch_size, 1152, 8, 1]
            self.input = tf.reshape(input, shape=(cfg.batch_size, 1152, 8, 1))
            capsules = [Capsule() for i in range(self.num_outputs)]
            capsules = [capsules[i](input) for i in range(self.num_outputs)]
            return(capsules)
        else:
            # the PrimaryCaps layer
            pass


class Capsule(object):
    ''' The routing algorithm.
    Args:
        input: A Tensor with shape [batch_size, num_caps, length(u_i), 1]

    Returns:
        A Tensor of shape [batch_size, 1, length(v_j), 1] representing the
        vector output `v_j` of capsule j
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''

    def __init__(self):
        # [batch_size, 32x6x6, 8]
        with tf.variable_scope('routing'):
            self.W_ij = tf.get_variable('w_ij', shape=(1, 1152, 16, 8))
            self.b_ij = tf.get_variable('b_ij', shape=(1, 1152, 1, 1))

    def __call__(self, input):
        '''
        Args:
            input: shape [batch_size, 1152, 8, 1]
        Returns:
            shape [batch_size, 1, 16, 1]
        '''
        self.input = input
        for r_iter in cfg.iter_routing:
            # line 4:
            # [1, 1152, 1, 1]
            c_i = tf.nn.softmax(self.b_ij, dim=1)

            # line 5:
            # [16, 8] x [8, 1] => [batch_size, 1152, 16, 1]
            u_hat = tf.matmul(self.W_ij.T, self.input)
            # weighting u_hat with c_i in the third dim,
            # then sum in the second dim, resulting in [batch_size, 1, 16, 1]
            s_j = tf.reduce_sum(tf.multipy(c_i, u_hat))

            # line 6:
            # squash using Eq.1, resulting in [batch_size, 1, 16, 1]
            s_abs = tf.abs(s_j)
            scalar_factor = tf.square(s_abs) / (1 + tf.square(s_abs))
            v_j = scalar_factor * tf.divide(s_j, s_abs)
            # line 7:
            # [1152, 16] x [16, 1] => [1152, 1], then reduce mean in the
            # batch_size dim, resulting in [1, 1152, 1, 1]
            u_produce_v = tf.matmul(u_hat, v_j)
            self.b_ij = self.b_ij + u_produce_v

        return(v_j)
