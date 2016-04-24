#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import time
from random import shuffle
import argparse
import os

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
        self.all = self.read_matrix(matrix_dir)
        self.labels = [row[-2:] for row in self.all]
        self.matrix = [row[:-2] for row in self.all]

    def read_matrix(self,matrix_dir):
        matrix = []

        with open(matrix_dir,"r") as matrix_file:
            i = 0
            for line in matrix_file:
                matrix.append([])

                line = line.strip().split(' ')

                # this could be taken out by modifying FEATURES project output format for labels
                line[ -1 ] = 1 if line[-1] == 'S' else 0
                line.append(1 if line[-1] == 0 else 0)

                matrix[i] = [int(entry) for entry in line if entry != ' ']
                i += 1

        shuffle(matrix)
        return matrix

# Read Command line args
# Overwrites the hyper parameters

parser = argparse.ArgumentParser(description='Simple FeedForward Neural Network')
parser.add_argument('-f', '--input-matrix')
parser.add_argument('-i', '--input-neurons', type=int, default=0)
parser.add_argument('-n', '--hidden-neurons', type=int, default=0)
parser.add_argument('-b', '--batch-size', type=int, default=128)
parser.add_argument('-l', '--learning-rate', type=float, default=0.05)
parser.add_argument('--iterations', type=int, default=100)
args = parser.parse_args()

if(not args.input_matrix):
    print("Input matrix file not given.")
    parser.print_help()
    exit(1)

if(not os.path.isfile(args.input_matrix)):
    print("File {:s} doesn't exist.".format(args.input_matrix))
    exit(1)

##############
# Get Inputs #
##############

print("Reading Training data from '{:s}'.".format(args.input_matrix))

in_data = EmailSet(args.input_matrix)
size = int(len(in_data.matrix) * .8)
trX = in_data.matrix[:size]
trY = in_data.labels[:size]
teX = in_data.matrix[size:]
teY = in_data.labels[size:]

print("Using {:d} Training sets and {:d} Test sets".format(len(trX), len(teX)))
# print('train X', trX)
# print('train Y', trY)
# print('test X', teX)
# print('test Y', teY)
####################
# Hyper Parameters #
####################

if args.input_neurons: # Use only first X features
    trX = [row[:args.input_neurons] for row in trX]
    teX = [row[:args.input_neurons] for row in teX]

input_layer = len(trX[0])
hidden_layer = args.hidden_neurons or int(input_layer * 1.5)
output_layer = 2 # len(trY[0])

########################
# Build Neural Network #
########################

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_o): # , b_h, b_o):
    h = tf.matmul(X, w_h)
    # h = tf.nn.tanh(
    #     tf.add(
    #         tf.matmul(X, w_h),
    #         b_h
    #     )
    # )

    # we dont take the softmax at the end because our cost fn does that for us
    predict = tf.matmul(h, w_o)
    # predict = tf.add(
    #     tf.matmul(h, w_o),
    #     b_o
    # )

    return predict

# I/O
X = tf.placeholder("float", [None, input_layer])
Y = tf.placeholder("float", [None, output_layer])

# Weights
w_h = init_weights([input_layer, hidden_layer])
w_o = init_weights([hidden_layer, output_layer])

# Biases
# b_h = init_weights([1, hidden_layer])
# b_o = init_weights([1, output_layer])

# Prediction
py_x = model(X, w_h, w_o) # , b_h, b_o)
predict_op = tf.argmax(py_x, 1) # Spam or Ham

# compute costs
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))

# construct an optimizer (Back Prop)
train_op = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(cost)

# Launch the graph in a session
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    # Lets train over this set a few times
    for itera in range(args.iterations):
        accuracy = []
        f1Scores = []

        for start, end in zip(range(0, len(trX), args.batch_size), range(args.batch_size, len(trX), args.batch_size)):
            batchNum = start // args.batch_size

            # Lets try to predict the test now
            predictionList = sess.run(predict_op, feed_dict={X: teX, Y: teY})
            print(predictionList, np.argmax(teY, axis=1))

            # Calculate accuracy and save it to accuracy list
            accuracy.append(np.mean(np.argmax(teY, axis=1) == predictionList))

            # Calc f score and save to list
            baseScore = [0, 0, 0, 0]  # tp,tn,fp,fn
            fscore = 0

            for i in range(0, len(teY)):
                if np.argmax(teY[i]) == 1:  # email is legitimate
                    if predictionList[i] == 1:  # predicted legitimate as legitimate    (true positive)
                        baseScore[0] += 1
                    else:  # predicted legitimate as spam          (false negative)
                        baseScore[3] += 1
                else:  # email is spam
                    if predictionList[i] == 1:  # predicted spam as legitimate          (false positive)
                        baseScore[2] += 1
                    else:  # predicted spam as spam                (true negative)
                        baseScore[1] += 1
            print(baseScore)
            if baseScore[0]:
                precision = baseScore[0] / (baseScore[0] + baseScore[2])  # might not cast automatically
                recall = baseScore[0] / (baseScore[0] + baseScore[3])
                fscore = 2 * ((precision * recall) / (precision + recall))

            f1Scores.append(fscore)

            # Attempt this args.batch_size
            print("Test>> Iteration: {:d}\tBatch: {:d}\tStep: {:d}\tAccuracy: {:.7f}\tAccuracy Aggregate: {:.7f}\tFScore: {:.7f}\tFScore Aggregate: {:.7f}".format(
                itera,
                batchNum,
                itera * (len(trX)//args.batch_size) + batchNum,
                accuracy[-1],
                np.average(accuracy),
                f1Scores[-1],
                np.average(f1Scores)
            ))

            # Then train on it
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

            # Log the train duration
            print("Train>> Iteration: {:d}\tBatch: {:d}\tStep: {:d}\tTimestamp: {:.6f}".format(
                itera,
                batchNum,
                itera * (len(trX) // args.batch_size) + batchNum,
                time.time()
            ))
