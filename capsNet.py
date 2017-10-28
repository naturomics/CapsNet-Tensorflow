import tensorflow as tf

from config import cfg
import capsLayer


class CapsNet(object):
    def __init__(self):
        self.input = tf.placeholder(tf.float32, shape=(cfg.batch_size, 28, 28))

    def build_arch(self):
        # Conv1
        conv1 = tf.contrib.layers.conv2d(self.input, num_outputs=256,
                                         kernel_size=9, strides=1)

        # Primary Capsules
        primaryCaps = capsLayer(conv1)

        # DigitCaps layer

        # Decoder structure in Fig. 2

    def loss(self):
        pass
