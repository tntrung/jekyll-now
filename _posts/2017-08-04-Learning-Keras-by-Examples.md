---
layout: post
title: Learning basic TensorFlow, Keras and Convolutional Neural Network (CNN) by examples!
---

In this tutorial, we learn TensorFlow, Keras by going step by step from simple thing to recent state-of-the-art neural network in computer vision. At the beginning of the tutorial, we learn how to implement convolutional neural networks by TensorFlow and more efficient tool Keras. We start with simple to get familiar with them. Towards the end of this tutorial, you can go advance to implement from the scratch state-of-the-art convolution neural networks: VGG, Inception V4, DenseNet, etc.

# What is Keras?

Keras is a high-level Python API, which is designed to quickly build and train neural networks using either `TensorFlow` or `Theano`. It is developed by François Chollet.

# Why Keras?

Keras is hailed to be the future of deep learning framework.

# Installing Keras on TensorFlow

# Learning Keras by examples

Let's start Keras excervise by the simple classification problem by using linear regression and `softmax`, which is often considered as the hello world lesson in machine learning.

## Example 1: Linear regression

``Data preparation``: Tensorflow provide function that we can easily download and prepare MNIST data by one line of Python code. To facilitate for our learning and implementation in the next lesson, we put the code in a single file: e.g. `datatool/datasets.py`.

```python
from tensorflow.examples.tutorials.mnist import input_data

def read_tf_mnist(data_dir):
	# Read MNIST data by TensorFlow
  # mnist.train.images [N 784]
  # mnist.train.labels [N 10]
	mnist = input_data.read_data_sets(data_dir, one_hot=True)
	return mnist
```

Import the TensorFlow, and `datatool` functions (ensure Python can read the functions from other directories by creating a file named `__init__.py` in those directories):

```python
import tensorflow as tf
from datatool.datasets import read_tf_mnist
```

Loading MNIST by using our `datatool`:

```python
mnist = read_tf_mnist("MNIST_data")
```

Creating placeholders to assign data at on. For our classification problem on MNIST, the data consists of raw images and their labels. Therefore, we need declare two placeholders:

```python
x  = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
```

`x` stores the image data ((which was vectorized in MNIST)) in type float `tf.float32`. `[None,784]` indicates data dimension of 784 with no constraint of image number can be put in the placeholder. Similarly for `_y` as the ground-truth label of 10 dimension.

Declare variables `W` for weights and `b` for bias to model the regression problem: `y = f(W*x+b)`, `f` is the softmax function:

```python
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
```

*To be continued*
