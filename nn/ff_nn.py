#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import time
from random import shuffle
import sys, getopt

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

########
# read input from features matrix
# store in matrix data structures for NN processing
########

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

# Read Command line args
# Overwrites the hyper parameters

try:
    opts, args = getopt.getopt(sys.argv[1:], "d:i:h:b:",
                               ["data-set=", "input-neurons=", "hidden-neurons=", "batch-size="])
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
        batch = int(arg)

##############
# Get Inputs #
##############

print("Reading Training data from '{:s}'.".format(DATA_SOURCE))

in_data = EmailSet(DATA_SOURCE)
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


def model(X, w_h, w_o, b_h, b_o):
    h = tf.nn.tanh(
        tf.add(
            tf.matmul(X, w_h),
            b_h
        )
    )

    # we dont take the softmax at the end because our cost fn does that for us
    predict = tf.add(
        tf.matmul(h, w_o),
        b_o
    )

    return predict

# I/O
X = tf.placeholder("float", [None, input_layer])
Y = tf.placeholder("float", [None, output_layer])

# Weights
w_h = init_weights([input_layer, hidden_layer])
w_o = init_weights([hidden_layer, output_layer])

# Biases
b_h = init_weights([1, hidden_layer])
b_o = init_weights([1, output_layer])

# Prediction
py_x = model(X, w_h, w_o, b_h, b_o)
predict_op = tf.argmax(py_x, 1) # Spam or Ham

# compute costs
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))

# construct an optimizer (Back Prop)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Launch the graph in a session
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    # Lets train over this set a few times
    for i in range(iterations):
        accuracy = []
        f1score = []

        for start, end in zip(range(0, len(trX), batch), range(batch, len(trX), batch)):
            step = start // batch

            # Calculate accuracy and save it to accuracy list
            predictionList = sess.run(predict_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
            accuracy.append(np.mean(
                np.argmax(trY[start:end], axis=1) == predictionList
            ))

            baseScore = [0, 0, 0, 0]  # tp,tn,fp,fn
            for i in range(start, end):
                if trY[i] == 1:  # email is legitimate
                    if predictionList[i - start] == 1:  # predicted legitimate as legitimate    (true positive)
                        baseScore[0] += 1
                    elif predictionList[i - start] == 0:  # predicted legitimate as spam          (false negative)
                        baseScore[3] += 1
                else:  # email is spam
                    if predictionList[i - start] == 1:  # predicted spam as legitimate          (false positive)
                        baseScore[2] += 1
                    else:  # predicted spam as spam                (true negative)
                        baseScore[1] += 1

            precision = baseScore[0] / (baseScore[0] + baseScore[2])  # might not cast automatically
            recall = baseScore[0] / (baseScore[0] + baseScore[3])
            fscore = 2 * ((precision * recall) / (precision + recall))

            f1score.append(fscore)

            # Attempt this batch
            print("Test>> Iteration: {:d}\tBatch: {:d}\tStep: {:d}\tAccuracy: {:.7f}\tAggregate: {:.7f}\tFScore: {:.7f}\tFScore Aggregate: {:.7f}".format(
                i,
                step, #batch
                i*(len(trX)//batch)+step, #step
                accuracy[step],
                np.average(accuracy),
                f1score[step],
                np.average(f1score)
            ))

            # Then train on it
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

            # Log the train duration
            print("Train>> Iteration: {:d}\tBatch: {:d}\tStep: {:d}\tTimestamp: {:.6f}".format(
                i,
                step,
                i * (len(trX) // batch) + step,
                time.time()
            ))
