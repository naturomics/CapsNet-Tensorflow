import tensorflow as tf

from config import cfg
from utils import get_batch_data
from capsLayer import CapsConv


class CapsNet(object):
    def __init__(self, is_training=True):
        if is_training:
            self.X, self.Y = get_batch_data()

            self.build_arch()
            self.loss()

            t_vars = tf.trainable_variables()
            self.optimizer = tf.train.AdamOptimizer()
            self.train_op = self.optimizer.minimize(self.total_loss,
                                                    var_list=t_vars)
        else:
            self.X = tf.placeholder(tf.float32,
                                    shape=(cfg.batch_size, 28, 28, 1))

        tf.logging.info('Seting up the main structure')

    def build_arch(self):
        # Conv1, [batch_size, 20, 20, 256]
        conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=256,
                                         kernel_size=9, stride=1,
                                         padding='VALID')
        assert conv1.get_shape() == [cfg.batch_size, 20, 20, 256]

        # TODO: Rewrite the 'CapsConv' class as a function, the capsLay
        # function should be encapsulated into tow function, one like conv2d
        # and another is fully_connected in Tensorflow.
        # Primary Capsules, [batch_size, 1152, 8, 1]
        primaryCaps = CapsConv(num_units=8, with_routing=False)
        caps1 = primaryCaps(conv1, num_outputs=32, kernel_size=9, stride=2)
        assert caps1.get_shape() == [cfg.batch_size, 1152, 8, 1]

        # DigitCaps layer, [batch_size, 10, 16, 1]
        digitCaps = CapsConv(num_units=16, with_routing=True)
        self.caps2 = digitCaps(caps1, num_outputs=10)

        # Decoder structure in Fig. 2
        # TODO: before reconstruction the input caps2 should do masking to pick
        # out the activity vector of the correct digit capsule.
        fc1 = tf.contrib.layers.fully_connected(self.caps2, num_outputs=512)
        fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
        self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784)

    def loss(self):
        # 1. The margin loss

        # calc ||v_c||
        # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
        v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True))
        assert v_length.get_shape() == [cfg.batch_size, 10, 1, 1]

        # [batch_size, 10, 1, 1]
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., cfg.m_plus - v_length))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., v_length - cfg.m_minus))
        assert max_l.get_shape() == [cfg.batch_size, 10, 1, 1]

        # TODO:calc T_c [batch_size, 10, 1, 1]
        T_c = ''
        # [batch_size, 10, 1, 1, 1]
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 2. The reconstruction loss
        orgin = tf.reshape(self.input, shape=(cfg.batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(squared)

        # 3. Total loss
        self.total_loss = self.margin_loss + 0.0005 * self.reconstruction_err
