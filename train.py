import tensorflow as tf

from config import cfg
from utils import load_mnist
from capsNet import CapsNet


if __name__ == "__main__":
    trainX, trainY, testX, testY = load_mnist(cfg.dataset)

    capsNet = CapsNet(is_training=cfg.is_training)
    tf.logging.info('Graph loaded')
    sv = tf.train.Supervisor()

    with sv.managed_session() as sess:
        for epoch in range(cfg.epoch):
            if sv.should_stop():
                break
            sess.run(capsNet.train_op)

    tf.logging.info('Training done')
