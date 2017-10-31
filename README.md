# CapsNet-Tensorflow

[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=plastic)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=plastic)](https://opensource.org/licenses/Apache-2.0)
![completion](https://img.shields.io/badge/completion%20state-90%25-blue.svg?style=plastic)

A Tensorflow implementation of CapsNet in Hinton's paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

- **Note:**

**The routing algorithm has been found not implemented as the same as the one in the paper!(#8, thanks very much, it's my carelessness) Although it works, but not effective as the results of the paper, I've paste out some results of the 'wrong' version(It's really interesting). I'll analyze the reasons of this interesting thing**

[Here](https://zhihu.com/question/67287444/answer/251460831) is my understanding of the section 4 of the paper (the core part of CapsNet), it might be helpful for understanding the code. Thanks for your focus

if you find out any problems, please let me know. I will try my best to 'kill' it as quickly as possible.

In the day of waiting, be patient: Merry days will come, believe. ---- Alexander PuskinIf :blush:

## Chat group:
WeChat: ![wechat](imgs/wechat_group.png)
Gitter: [![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg?style=plastic)](https://gitter.im/CapsNet-Tensorflow/Lobby)

We have a lot of interesting discussion in the WeChat group, welcome to join us. We will release the discussion results in the name of this group, pointing out the contribution of any contributors

## Requirements
- Python
- NumPy
- [Tensorflow](https://github.com/tensorflow/tensorflow) (I'm using 1.3.0, others should work, too)
- tqdm (for showing training progress info)

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
$ pip install tqdm  # install it if you haven't installed yes
$ python train.py
```

the tqdm package is not necessary, just a tool for showing the training progress. If you don't want it, change the loop ``for in step ...`` to ``for step in range(num_batch)`` in ``train.py``

### Evaluation
```
$ python eval.py
```


## Results
Code is still running, here some results:

![total_loss](imgs/total_loss.png)

![margin_loss](imgs/margin_loss.png)
![reconstruction_loss](imgs/reconstruction_loss.png)

### TODO:
- Finish the MNIST version of capsNet (progress:90%)
- Do some different experiments for capsNet:
  * Using other datasets such as CIFAR
	* Adjusting model structure

- There is [another new paper](https://openreview.net/pdf?id=HJWLfGWRb) about capsules(submitted to ICLR 2018), follow-up.
