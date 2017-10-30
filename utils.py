import os
import numpy as np
import tensorflow as tf

from config import cfg


def load_mnist(path):
    fd = open(os.path.join(cfg.dataset, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(cfg.dataset, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(cfg.dataset, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(cfg.dataset, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    # normalization and convert to a tensor [60000, 28, 28, 1]
    trX = tf.convert_to_tensor(trX / 255., tf.float32)
    teX = tf.convert_to_tensor(teX / 255., tf.float32)  # [10000, 28, 28, 1]

    # => [num_samples, 10]
    trY = tf.one_hot(trY, depth=10, axis=1, dtype=tf.float32)
    teY = tf.one_hot(teY, depth=10, axis=1, dtype=tf.float32)

    return trX, trY, teX, teY


def get_batch_data():
    trX, trY, teX, teY = load_mnist(cfg.dataset)

    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=cfg.num_threads,
                                  batch_size=cfg.batch_size,
                                  capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32,
                                  allow_smaller_final_batch=False)

    return(X, Y)


if __name__ == '__main__':
    trX, trY, teX, teY = load_mnist(cfg.dataset)
    print(teY.get_shape())
    print(teY.dtype)
