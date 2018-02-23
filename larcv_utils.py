from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys,time
import numpy as np

import tensorflow as tf

from larcv import larcv
from larcv.dataloader2 import larcv_threadio
from larcv_config import Class_config

class LArCVDataset(object):
    def __init__(self):
        self._cfg        = Class_config()
        self._input_main = None
        self._input_test = None
        self._output     = None

    def initialize(self):
    # Instantiate and configure
        if not self._cfg.MAIN_INPUT_CONFIG:
            print('Must provide larcv data filler configuration file!')
            return 
        #
        # Data IO configuration
        #
        # Main input stream
        self._input_main = larcv_threadio()
        filler_cfg = {'filler_name' : 'ThreadProcessor',
                      'verbosity'   : 0, 
                      'filler_cfg'  : self._cfg.MAIN_INPUT_CONFIG}
        self._input_main.configure(filler_cfg)
        self._input_main.start_manager(self._cfg.MINIBATCH_SIZE)

        # Test input stream (optional)
        if self._cfg.TEST_INPUT_CONFIG:
            self._input_test = larcv_threadio()
            filler_cfg = {'filler_name' : 'TestIO',
                          'verbosity'   : 0,
                          'filler_cfg'  : self._cfg.TEST_INPUT_CONFIG}
            self._input_test.configure(filler_cfg)
            self._input_test.start_manager(self._cfg.TEST_BATCH_SIZE)

        # Output stream (optional)
        if self._cfg.ANA_OUTPUT_CONFIG:
            self._output = larcv.IOManager(self._cfg.ANA_OUTPUT_CONFIG)
            self._output.initialize()

        # Retrieve image/label dimensions
        self._input_main.next(store_entries   = (not self._cfg.TRAIN),
                              store_event_ids = (not self._cfg.TRAIN))
        dim_data = self._input_main.fetch_data(self._cfg.KEYWORD_DATA).dim()
        
        return dim_data

if __name__ == '__main__':
    k = LArCVDataset()
    dim_data = k.initialize()
    print(dim_data)
