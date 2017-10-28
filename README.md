# CapsNet-Tensorflow
A Tensorflow implementation of CapsNet in Hinton's paper ['Dynamic Routing Between Capsules'](https://arxiv.org/abs/1710.09829)

## Requirements
- Python
- [Tensorflow](https://github.com/tensorflow/tensorflow)

## Usage

### Training
** Step 1. ** 
Clone this repository with ``git``.

```
$ git clone https://github.com/naturomics/CapsNet-Tensorflow.git
$ cd CapsNet-Tensorflow
```

** Step 2. ** 
Download [MNIST dataset](http://yann.lecun.com/exdb/mnist/), ``mv`` and extract them into ``data/mnist`` directory.

```
$ mkdir -p data/mnist
$ wget -c -P data/mnist http://yann.lecun.com/exdb/mnist/{train-images-idx3-ubyte.gz,train-labels-idx1-ubyte.gz,t10k-images-idx3-ubyte.gz,t10k-labels-idx1-ubyte.gz}
$ gunzip data/mnist/*.gz
```

** Step 3. ** 
Start training with command line:
```
$ python train.py
```

### Evaluation
```
python eval.py
```


## Results
