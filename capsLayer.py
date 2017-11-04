import numpy as np
import tensorflow as tf

from config import cfg


epsilon = 1e-9


class CapsLayer(object):
    ''' Capsule layer.
    Args:
        input: A 4-D tensor.
        num_outputs: the number of capsule in this layer.
        vec_len: integer, the length of the output vector of a capsule.
        layer_type: string, one of 'FC' or "CONV", the type of this layer,
            fully connected or convolution, for the future expansion capability
        with_routing: boolean, this capsule is routing with the
                      lower-level layer capsule.

    Returns:
        A 4-D tensor.
    '''
    def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type

    def __call__(self, input, kernel_size=None, stride=None):
        '''
        The parameters 'kernel_size' and 'stride' will be used while 'layer_type' equal 'CONV'
        '''
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride

            if not self.with_routing:
                # the PrimaryCaps layer, a convolutional layer
                # input: [batch_size, 20, 20, 256]
                assert input.get_shape() == [cfg.batch_size, 20, 20, 256]

                capsules = []
                for i in range(self.vec_len):
                    # each capsule i: [batch_size, 6, 6, 32]
                    with tf.variable_scope('ConvUnit_' + str(i)):
                        caps_i = tf.contrib.layers.conv2d(input, self.num_outputs,
                                                          self.kernel_size, self.stride,
                                                          padding="VALID")
                        caps_i = tf.reshape(caps_i, shape=(cfg.batch_size, -1, 1, 1))
                        capsules.append(caps_i)

                assert capsules[0].get_shape() == [cfg.batch_size, 1152, 1, 1]

                # [batch_size, 1152, 8, 1]
                capsules = tf.concat(capsules, axis=2)
                capsules = squash(capsules)
                assert capsules.get_shape() == [cfg.batch_size, 1152, 8, 1]
                return(capsules)

        if self.layer_type == 'FC':
            if self.with_routing:
                # the DigitCaps layer, a fully connected layer
                # Reshape the input into [batch_size, 1152, 1, 8, 1]
                self.input = tf.reshape(input, shape=(cfg.batch_size, -1, 1, input.shape[-2].value, 1))

                with tf.variable_scope('routing'):
                    # b_IJ: [1, num_caps_l, num_caps_l_plus_1, 1 , 1]
                    b_IJ = tf.constant(np.zeros([1, input.shape[1].value, self.num_outputs,1,1], dtype=np.float32))
                    capsules = routing(self.input, b_IJ)
                    capsules = tf.squeeze(capsules, axis=1)

            return(capsules)


def routing(input, b_IJ):
    ''' The routing algorithm.

    Args:
        input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
               shape, num_caps_l meaning the number of capsule in the layer l.
    Returns:
        A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
        representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''

    # W: [num_caps_j, num_caps_i, len_u_i, len_v_j]
    W = tf.get_variable('Weight', shape=(1, 1152, 10, 8, 16), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=cfg.stddev))

    # Eq.2, calc u_hat
    # do tiling for input and W before matmul
    # input => [batch_size, 1152, 10, 8, 1]
    # W => [batch_size, 1152, 10, 8, 16]
    input = tf.tile(input, [1, 1, 10, 1, 1])
    W = tf.tile(W, [cfg.batch_size, 1, 1, 1, 1])
    assert input.get_shape() == [cfg.batch_size, 1152, 10, 8, 1]

    # in last 2 dims:
    # [8, 16].T x [8, 1] => [16, 1] => [batch_size, 1152, 10, 16, 1]
    u_hat = tf.matmul(W, input, transpose_a=True)
    assert u_hat.get_shape() == [cfg.batch_size, 1152, 10, 16, 1]

    # line 3,for r iterations do
    for r_iter in range(cfg.iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [1, 1, 1152, 10, 1]
            c_IJ = tf.nn.softmax(b_IJ, dim=3)
            c_IJ = tf.tile(c_IJ, [cfg.batch_size, 1, 1, 1, 1])
            assert c_IJ.get_shape() == [cfg.batch_size, 1152, 10, 1, 1]

            # line 5:
            # weighting u_hat with c_IJ, element-wise in the last two dim
            # => [batch_size, 1152, 10, 16, 1]
            s_J = tf.multiply(c_IJ, u_hat)
            # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
            s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
            assert s_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]

            # line 6:
            # squash using Eq.1,
            v_J = squash(s_J)
            assert v_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]

            # line 7:
            # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 10, 1152, 16, 1]
            # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
            # batch_size dim, resulting in [1, 1152, 10, 1, 1]
            v_J_tiled = tf.tile(v_J, [1, 1152, 1, 1, 1])
            u_produce_v = tf.matmul(u_hat, v_J_tiled, transpose_a=True)
            assert u_produce_v.get_shape() == [cfg.batch_size, 1152, 10, 1, 1]
            b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)

    return(v_J)


def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A 5-D tensor with shape [batch_size, 1, num_caps, vec_len, 1],
    Returns:
        A 5-D tensor with the same shape as vector but squashed in 4rd and 5th dimensions.
    '''
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)


# TODO: 1. Test the `fully_connected` and `conv2d` function;
#       2. Update  doc about these two function.
def fully_connected(inputs,
                    num_outputs,
                    vec_len,
                    with_routing=True,
                    weights_initializers=tf.initializers.xavier_initializer(),
                    reuse=None,
                    variable_collections=None,
                    scope=None):
    '''A capsule fully connected layer.(Notes: not tested yet)
    Args:
        inputs: A tensor of as least rank 3, i.e. `[batch_size, num_inputs, vec_len]`,
                `[batch_size, num_inputs, vec_len, 1]`.
        num_outputs: ...
    Returns:
        ...
    Raise:
        ...
    '''
    layer = CapsLayer(num_outputs=num_outputs,
                      vec_len=vec_len,
                      with_routing=with_routing,
                      layer_type='FC')
    return layer.apply(inputs)


def conv2d(inputs,
           filters,
           vec_len,
           kernel_size,
           strides=(1, 1),
           with_routing=False,
           reuse=None):
    '''A capsule convolutional layer.(Notes: not tested yet)
    Args:
        inputs: A tensor.
    Returns:
        ...
    Raises:
        ...
    '''
    layer = CapsLayer(num_outputs=filters,
                      vec_len=vec_len,
                      with_routing=with_routing,
                      layer_type='CONV')
    return(layer(inputs, kernel_size=kernel_size, stride=strides))
