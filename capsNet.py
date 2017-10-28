import tensorflow as tf

from config import cfg
from capsLayer import CapsConv


class CapsNet(object):
    def __init__(self):
        self.input = tf.placeholder(tf.float32, shape=(cfg.batch_size, 28, 28))

    def build_arch(self):
        # Conv1, [batch_size, 20, 20, 256]
        conv1 = tf.contrib.layers.conv2d(self.input, num_outputs=256,
                                         kernel_size=9, strides=1)

        # Primary Capsules, [batch_size, 1152, 8, 1]
        primaryCaps = CapsConv(num_units=8, with_routing=False)
        caps1 = primaryCaps(conv1, num_outputs=32, kernel_size=9, strides=2)

        # DigitCaps layer, [batch_size, 10, 8, 1]
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

        # [batch_size, 10, 1, 1]
        v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2))

        # [batch_size, 10, 1, 1]
        max_l = tf.square(tf.maximum(0, cfg.m_plus - v_length))
        max_r = tf.square(tf.maximum(0, v_length - cfg.m_minus))
        # TODO:calc T_c
        T_c = ''
        # [batch_size, 10, 1, 1, 1]
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 2. The reconstruction loss
        orgin = tf.reshape(self.input, shape=(cfg.batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(squared)

        # 3. Total loss
        self.tatol_loss = self.margin_loss + 0.0005 * self.reconstruction_err
