---
layout: post
title: Learning Keras by examples!
---

# What is Keras?

# Installing Keras on TensorFlow

# Learning Keras by examples

## Example 1: Linear regression

```
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
