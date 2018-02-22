from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

class Class_config:

    NUM_CLASS         = 5
    BASE_NUM_FILTERS  = 16
    MAIN_INPUT_CONFIG = 'config/input_train2d.cfg'
    TEST_INPUT_CONFIG = ''
    ANA_OUTPUT_CONFIG = ''
    LOGDIR            = 'class_train_log'
    SAVE_FILE         = 'class_checkpoint/capsnet'
    LOAD_FILE         = ''
    AVOID_LOAD_PARAMS = []
    LEARNING_RATE     = -1
    MINIBATCH_SIZE    = 10
    NUM_MINIBATCHES   = 5
    TEST_BATCH_SIZE   = 10
    ITERATIONS        = 100000
    TF_RANDOM_SEED    = 1234
    TRAIN             = True
    DEBUG             = False
    USE_WEIGHTS       = False
    REPORT_STEPS      = 200
    SUMMARY_STEPS     = 20
    CHECKPOINT_STEPS  = 200
    CHECKPOINT_NMAX   = 10
    CHECKPOINT_NHOUR  = 0.4
    KEYWORD_DATA      = 'image'
    KEYWORD_LABEL     = 'label'
    KEYWORD_WEIGHT    = ''
    KEYWORD_TEST_DATA   = ''
    KEYWORD_TEST_LABEL  = ''
    KEYWORD_TEST_WEIGHT = ''
    def __init__(self):
        pass

if __name__ == '__main__':
    c = Class_config()
    if not os.path.exists(c.MAIN_INPUT_CONFIG):
        print("wrong main input config")
        
