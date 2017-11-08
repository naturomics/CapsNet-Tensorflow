import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import cfg
import numpy as np

def build_arch(input, is_train=False):
    initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    biasInitializer = tf.constant_initializer(0.0)

    with slim.arg_scope([slim.conv2d], trainable=is_train, weights_initializer=initializer, biases_initializer=biasInitializer):
        with tf.variable_scope('Conv1_layer') as scope:
            output = slim.conv2d(input, num_outputs=256, kernel_size=[9, 9], stride=1, padding='VALID', scope=scope)
            assert output.get_shape() == [cfg.batch_size, 20, 20, 256]

        with tf.variable_scope('PrimaryCaps_layer') as scope:
            output = slim.conv2d(output, num_outputs=32*8, kernel_size=[9, 9], stride=2, padding='VALID', scope=scope)
            output = tf.reshape(output, [cfg.batch_size, -1, 1, 8])
            assert output.get_shape() == [cfg.batch_size, 1152, 1, 8]

        with tf.variable_scope('DigitCaps_layer') as scope:
            u_hats = []
            input_groups = tf.split(axis=1, num_or_size_splits=1152, value=output)
            for i in range(1152):
                u_hat = slim.conv2d(input_groups[i], num_outputs=16*10, kernel_size=[1, 1], stride=1, padding='VALID', scope='DigitCaps_layer_w_'+str(i))
                u_hat = tf.reshape(u_hat, [cfg.batch_size, 1, 10, 16])
                u_hats.append(u_hat)

            output = tf.concat(u_hats, axis=1)
            assert output.get_shape() == [cfg.batch_size, 1152, 10, 16]

            b_ijs = tf.constant(np.zeros([1152, 10], dtype=np.float32))
            v_js = []
            for r_iter in range(cfg.iter_routing):
                with tf.variable_scope('iter_'+str(r_iter)):
                    c_ijs = tf.nn.softmax(b_ijs, dim=1)

                    c_ij_groups = tf.split(axis=1, num_or_size_splits=10, value=c_ijs)
                    b_ij_groups = tf.split(axis=1, num_or_size_splits=10, value=b_ijs)
                    input_groups = tf.split(axis=2, num_or_size_splits=10, value=output)

                    s_js = []

                    for i in range(10):
                        c_ij = tf.reshape(tf.tile(c_ij_groups[i], [1, 16]), [1152, 1, 16, 1])
                        s_j = tf.nn.depthwise_conv2d(input_groups[i], c_ij, strides=[1, 1, 1, 1], padding='VALID')
                        assert s_j.get_shape() == [cfg.batch_size, 1, 1, 16]

                        s_j = tf.reshape(s_j, [cfg.batch_size, 16])
                        s_j_norm_square = tf.reduce_mean(tf.square(s_j), axis=1, keep_dims=True)
                        v_j = s_j_norm_square*s_j/((1+s_j_norm_square)*tf.sqrt(s_j_norm_square+1e-9))
                        assert v_j.get_shape() == [cfg.batch_size, 16]

                        tmp = tf.reduce_sum(tf.matmul(tf.reshape(input_groups[i], [cfg.batch_size, 1152, 16]), tf.reshape(v_j, [cfg.batch_size, 16, 1])), axis=0)
                        b_ij_groups[i] = b_ij_groups[i]+tmp

                        if r_iter == cfg.iter_routing-1:
                            v_js.append(tf.reshape(v_j, [cfg.batch_size, 1, 16]))

                    b_ijs = tf.concat(b_ij_groups, axis=1)

            output = tf.concat(v_js, axis=2)

    return output

















