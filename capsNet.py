import tensorflow as tf

from config import cfg
from capsLayer import CapsConv


class CapsNet(object):
    def __init__(self):
        self.input = tf.placeholder(tf.float32, shape=(cfg.batch_size, 28, 28))

    def build_arch(self):
        # Conv1
        conv1 = tf.contrib.layers.conv2d(self.input, num_outputs=256,
                                         kernel_size=9, strides=1)

        # Primary Capsules
        primaryCaps = CapsConv(num_units=8, with_routing=False)
        caps1 = primaryCaps(conv1, num_outputs=32, kernel_size=9, strides=2)

        # DigitCaps layer
        digitCaps = CapsConv(num_units=16, with_routing=True)
        caps2 = digitCaps(caps1, num_outputs=10)

        # Decoder structure in Fig. 2
        # TODO: before reconstruction the input caps2 should do masking to pick
        # out the activity vector of the correct digit capsule.
        fc1 = tf.contrib.layers.fully_connected(caps2, num_outputs=512)
        fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
        self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784)

    def loss(self):
        # The margin loss

        # The reconstruction loss
        orgin = tf.reshape(self.input, shape=(cfg.batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(squared)
