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

Let's start Keras excervise by the simple linear regression problem, which is often considered as the hello world lesson in machine learning.

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

First we go with the TensorFlow,  

```
```

```python
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# data generation
trX = np.linspace(-1,1,100)
trY = 3 * trX + np.random.randn(*trX.shape) * 0.33

# creat the linear model
model = Sequential()
model.add(Dense(input_dim=1,output_dim=1,init='uniform', activation='linear'))

# print initial weight
weights = model.layers[0].get_weights()
w_init = weights[0][0][0]
b_init = weights[1][0]

print('Linear regression model is initialized with weights w: %.2f, b: %.2f' % (w_init, b_init))
## Linear regression model is initialized with weight w: -0.03, b: 0.00

# choose optimizer
model.compile(optimizer='sgd', loss='mse')

# train the model
model.fit(trX,trY,nb_epoch=200,verbose=1)

# print the weights
weights = model.layers[0].get_weights()
w_final = weights[0][0][0]
b_final = weights[1][0]

print('Linear regression model is trained to have weights w: %.2f, b: %.2f' % (w_final, b_final))

model.save_weights('my_model.h5')

```
