# CapsNet-Tensorflow

[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=plastic)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=plastic)](https://opensource.org/licenses/Apache-2.0)
![completion](https://img.shields.io/badge/completion%20state-80%25-blue.svg?style=plastic)

A Tensorflow implementation of CapsNet in Hinton's paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

**The code is not complete yet, I'm trying to finish it as quickly as possible. [Here](https://zhihu.com/question/67287444/answer/251460831) is my understanding of the section 4 of the paper (the core part of CapsNet), it might be helpful for understanding the code. Thanks for your focus**

Give me one day more, and I will move the earth. ---- Archimedes :blush:

## Requirements
- Python
- NumPy
- [Tensorflow](https://github.com/tensorflow/tensorflow) (I'm using 1.3.0, others should work, too)

## Usage

### Training
**Step 1.** 
Clone this repository with ``git``.

```
$ git clone https://github.com/naturomics/CapsNet-Tensorflow.git
$ cd CapsNet-Tensorflow
```

**Step 2.** 
Download [MNIST dataset](http://yann.lecun.com/exdb/mnist/), ``mv`` and extract them into ``data/mnist`` directory.(Be careful the backslash appeared around the curly braces when you copy the ``wget `` command to your terminal, remove it)

```
$ mkdir -p data/mnist
$ wget -c -P data/mnist http://yann.lecun.com/exdb/mnist/{train-images-idx3-ubyte.gz,train-labels-idx1-ubyte.gz,t10k-images-idx3-ubyte.gz,t10k-labels-idx1-ubyte.gz}
$ gunzip data/mnist/*.gz
```

**Step 3.** 
Start training with command line:
```
$ python train.py
```

### Evaluation
```
$ python eval.py
```


## Results
