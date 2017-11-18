# CapsNet-Tensorflow Distributed Version

A distributed implementation of CapsNet for training and inference. Some optimization of network structure is also added for acceleration.

> **Status:**
> 1. The implementation of distributed training is finished.
> 2. Finish the speed test on single GPU.

> **Daily task**
> 1. Validation performance with multi-gpus
> 2. Implement the test and inference part

- Python
- NumPy
- [Tensorflow](https://github.com/tensorflow/tensorflow) 1.2.0+

> **Speed test report**
With single GPU GTX 1080 and CPU i7-5820K CPU @ 3.30GHz.
1 epoch for training on MNIST costs 157.4s, approximately 0.34s/iteration.
100 times inferences costs 4.2s, approximately 0.04s for inference.
