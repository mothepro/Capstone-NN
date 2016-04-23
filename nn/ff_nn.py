#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import time
DATA_SOURCE = './data/matrixEnron6.txt'


###################
# Of Machine & Men
#
# Maurice Prosper
# kace echo
# Cole Troutman
#
# CS 3359
#
###################

try:
    opts, args = getopt.getopt(sys.argv[1:],"d:i:h:b:",["data-set=","input-neurons=","hidden-neurons=","batch-size="])
except getopt.GetoptError:
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-i", "--input-neurons"):
        input_layer = int(arg)
    elif opt in ("-d", "--data-set"):
        dataSet = int(arg)
    elif opt in ("-h", "--hidden-neurons"):
        hidden_layer = int(arg)
    elif opt in ("-b", "--batch-size"):

#read input from features matrix and store in matrix data structures for NN processing
from random import shuffle

class EmailSet(object):
    def __init__(self,matrix_dir):
        self.matrix = self.read_matrix(matrix_dir)
        self.labels = self.create_label_matrix()
        self.matrix = self.remove_label_matrix(self.matrix)

    def read_matrix(self,matrix_dir):
        matrix = []

        with open(matrix_dir,"r") as matrix_file:
            i = 0
            for line in matrix_file:
                matrix.append([])

                line = line.strip().split(' ')

                #this could be taken out by modifying FEATURES project output format for labels
                line[ -1 ] = 1 if line[-1] == 'H' else 0
                line.append(1 if line[-1] == 0 else 0)

                matrix[i] = [int(entry) for entry in line if entry != ' ']

        shuffle(matrix)
        return matrix

    def create_label_matrix(self):
        matrix_label = [[entry[-2], entry[-1]] for entry in self.matrix]
        return matrix_label

    def remove_label_matrix(self,matrix):
        matrix = [entry[:-2] for entry in matrix]
        return matrix

##############
# Get Inputs #
##############

print("Reading Training data from '{:s}'.".format(DATA_SOURCE))

in_data = EmailSet(DATA_SOURCE);
size = int(len(in_data.matrix) * .8)
trX = in_data.matrix[:size]
trY = in_data.labels[:size]
teX = in_data.matrix[size:]
teY = in_data.labels[size:]

print("Using {:d} Training sets and {:d} Test sets".format(len(trX), len(teX)))

####################
# Hyper Parameters #
####################

iterations = 100
batch = 128
input_layer = len(trX[0])
hidden_layer = int(input_layer * 1.5)
output_layer = len(trY[0])
learning_rate = 0.05

########################
# Build Neural Network #
########################

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_o):
    h = tf.nn.tanh(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

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
    tf.initialize_all_variables().run()

    # Lets train over this set a few times
    for i in range(iterations):
        accuracy = []

        for start, end in zip(range(0, len(trX), batch), range(batch, len(trX), batch)):
            step = start//batch

            # Calculate accuracy and save it to accuracy list
            accuracy.append(np.mean(
                np.argmax(trY[start:end], axis=1) ==
                sess.run(predict_op, feed_dict={
                    X: trX[start:end],
                    Y: trY[start:end]
                })
            ))

            # Attempt this batch
            print("Test>> Iteration: {:d}\tBatch: {:d}\tStep: {:d}\tAccuracy: {:.7f}\tAggregate: {:.7f}".format(
                i,
                step,
                i*(len(trX)//batch)+step,
                accuracy[step],
                np.average(accuracy)
            ))

            # Then train on it
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

            # Log the train duration
            print("Train>> Iteration: {:d}\tBatch: {:d}\tStep: {:d}\tTimestamp: {:.6f}".format(
                i,
                step,
                i*(len(trX)//batch)+step,
                time.time()
            ))
