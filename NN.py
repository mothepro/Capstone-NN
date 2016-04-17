#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import inp

####################
# Hyper Parameters #
####################

iterations = 100
batch = 128
input_layer = 784
hidden_layer = 625
output_layer = 10
learning_rate = 0.05

###################

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


mnist = inp("TensorFlow-Tutorials/MNIST_data/")
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, input_layer])
Y = tf.placeholder("float", [None, output_layer])

w_h = init_weights([input_layer, hidden_layer]) # create symbolic variables
w_o = init_weights([hidden_layer, output_layer])

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    #summary_writer = tf.train.SummaryWriter('./logs.3', sess.graph)

    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(iterations):
        for start, end in zip(range(0, len(trX), batch), range(128, len(trX), batch)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
            print("Train>> Iteration: {:d}\tBatch: {:d}\tStep: {:d}".format(i, start//batch, (i*(len(trX)//batch)+(start//batch)) ))
        print("Test>> Iteration: {:d}\tAccuracy: {:.7f}".format(i, np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX, Y: teY})) ))
