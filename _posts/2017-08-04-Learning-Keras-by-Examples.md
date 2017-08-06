---
layout: post
title: Learning basic TensorFlow, Keras for CNN implementation by examples! - Part I
---

In this tutorial, we learn TensorFlow, Keras by going step by step from simple thing to recent state-of-the-art neural network in computer vision. At the beginning of the tutorial, we learn how to implement Convolutional Neural Networks (CNN) by TensorFlow and more efficient tool Keras. Towards the end of this tutorial, you can go advance to implement from the scratch state-of-the-art CNN, such as: VGG, Inception V4, DenseNet, etc. If you are not familiar with CNN, I recommend to take a look this tutorial first: [http://cs231n.github.io/convolutional-networks/](http://cs231n.github.io/convolutional-networks/)

# What is Keras?

Keras is a high-level Python API, which is designed to quickly build and train neural networks using either `TensorFlow` or `Theano`. It is developed by François Chollet.

# Why Keras?

Keras is hailed to be the future of deep learning framework.

# Installing Keras on TensorFlow

# Learning Keras by examples

Let's start Keras excervise by the simple classification problem by using linear regression and `softmax`, which is often considered as the hello world lesson in machine learning.

## Example 1: MNIST classification by linear model and softmax

### TensofFlow

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

`Define model`: creating placeholders to assign data at on. For our classification problem on MNIST, the data consists of raw images and their labels. Therefore, we need declare two placeholders:

```python
x  = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
```

`x` stores the image data ((which was vectorized in MNIST)) in type float `tf.float32`. `[None,784]` indicates data dimension of 784 with no constraint of image number can be put in the placeholder. Similarly for `_y` as the ground-truth label of 10 dimension.

Declare variables `W` for weights and `b` for bias to model the regression problem: `y = f(W*x+b)`, `f` is the softmax function.

```python
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
```

For similicity, we initialize `W` and `b` with zero matrices by `tf.zeros` which works with linear model. However, the zero weights are not applicable for deep neural network in the next examples. We then can define the model and define loss function of cross entropy:

```python
y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
```

`tf.nn.softmax` to compute softmax function. Because `_y` and `y` are matrices of `[None,10]`. `y_ * tf.log(y)` is the element-wise multiplication, and `tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1])` computes sum of elements of each row. To understand more about the function `tf.reduce_sum`, let's take a look this example:

```python
# 'x' is [[1, 1, 1]
#         [1, 1, 1]]
tf.reduce_sum(x) ==> 6
tf.reduce_sum(x, 0) ==> [2, 2, 2]
tf.reduce_sum(x, 1) ==> [3, 3]
tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
tf.reduce_sum(x, [0, 1]) ==> 6
```

ON another way, TF supports the softmax by simple function, which operates more efficient `tf.nn.softmax_cross_entropy_with_logits`. It is recommended to use instead of `tf.nn.softmax` because it covers numerically unstable corner cases in the mathematically right way.

```python
y = tf.matmul(x, W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
```
Choose the optimizer for training by using Gradient Descent with learning rate 0.5:

```python
train_step  = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

To encapsulate the environment in which `Operation` objects are executed and and Tensor objects are evaluated, we need a `Session`:

```python
with tf.Session() as sess:
  # init the variables
	sess.run(tf.global_variables_initializer())
  # training with 1000 iterations
	for _ in range(1000):
      # 100 sample per batch
	  	batch_xs, batch_ys = mnist.train.next_batch(100)
      # run training
	  	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  # evaluation
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

The model is fined above has no value until starting the session. `tf.global_variables_initializer()` initialize all parameters of the model. Then, the training is performed through 1000 iterations:

```python
sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

Each batch has 100 samples, and you need to specify the value of the learning phase as part of `feed_dict` corresponding with `placeholder` you defined at the beginning. Similarly for evaluation but conducted on test set of MNIST.

```python
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

The complete code is here `tf_mnist_softmax.py`:

```python
#!/usr/bin/python

import tensorflow as tf
from datatool.datasets import read_tf_mnist

mnist = read_tf_mnist("MNIST_data")

x  = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W) + b
# The loss function by cross entropy
cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step  = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
  # init the variables
	sess.run(tf.global_variables_initializer())
  # training with 1000 iterations
	for _ in range(1000):
      # 100 sample per batch
	  	batch_xs, batch_ys = mnist.train.next_batch(100)
      # run training
	  	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	# evaluation
	print("Accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

Run TF program:

```
>> python tf_mnist_softmax.py

>> ('Accuracy: ', 0.91909999)
```
The accuracy is about `92%`, which is not good on MNIST, because we used only one fully-connected layer. We'll try to improve the accuracy in next examples.

### Keras
In this session, we try to simplify TF by Keras, and minimize the change to keep the transition smooth:

Model definition from TF:

```python
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
```

To Keras which provide the layer `Dense` (fully-connected layer) can operate in similar, but shorter way:

```python
from keras.layers import Dense

y = Dense(10, activation='softmax')(x)
```
The loss function defined above of TF:

```python
# The loss function by cross entropy
cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
```

can be replaced by a built-in function of Keras:

```python
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy_eval

cross_entropy = tf.reduce_mean(categorical_crossentropy(y_, y))

# Test trained model
correct_prediction = accuracy_eval(y_, y)
```

The complete code with Keras is here `keras_mnist_softmax.py`:

```python
#!/usr/bin/python

import tensorflow as tf
from datatool.datasets import read_tf_mnist
import keras
from keras.layers import Dense
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy_eval

mnist = read_tf_mnist("MNIST_data")

x  = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])

y = Dense(10, activation='softmax')(x)  # fully-connected layer of output layer of 10 units with a softmax activation
cross_entropy = tf.reduce_mean(categorical_crossentropy(y_, y))

# Test trained model
correct_prediction = accuracy_eval(y_, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step  = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
  # init the variables
	sess.run(tf.global_variables_initializer())
  # training with 1000 iterations
	for _ in range(1000):
      # 100 sample per batch
	  	batch_xs, batch_ys = mnist.train.next_batch(100)
      # run training
	  	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  # evaluation
	print("Accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

Run new program and obtain the similar accuracy as above:

```
>> python keras_mnist_softmax.py

>> ('Accuracy: ', 0.91790003)
```

However, in this case we use Keras only as shortcut to map inputs to outputs, and built-in functions provided in Keras. The optimization is done via native TF optimizer rather than Keras optimizer. We do not even use Keras Model at all.

## Example 2: MNIST classification by LeNet

*To be continued*
