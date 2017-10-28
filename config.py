import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

# For separate margin loss
flags.DeFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DeFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DeFINE_float('lambda', 0.5, 'down weight of the loss for absent digit classes')


############################
#   environment setting    #
############################
flags.DEFINE_string('dataset', 'data/mnist', 'the path for dataset')


cfg = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)
