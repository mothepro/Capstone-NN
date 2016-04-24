#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import time
import argparse
import os
import pickle

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

# Read Command line args
# Overwrites the hyper parameters

parser = argparse.ArgumentParser(description='Simple FeedForward Neural Network')
parser.add_argument('-f', '--input')
parser.add_argument('-i', '--input-neurons', type=int, default=0)
parser.add_argument('-n', '--hidden-neurons', type=int, default=0)
parser.add_argument('-b', '--batch-size', type=int, default=100)
parser.add_argument('-l', '--learning-rate', type=float, default=0.05)
parser.add_argument('-z', '--iterations', type=int, default=100)
parser.add_argument('-t', '--train', type=bool, default=True)
parser.add_argument('-x', '--tolerance', type=float, default=0)
parser.add_argument('-s', '--save-point')
# parser.add_argument('--tolerance', type=float, default=0.001)
args = parser.parse_args()

if(not args.input):
    print("Input matrix file not given.")
    parser.print_help()
    exit(1)

if(not os.path.isfile(args.input)):
    print("File {:s} doesn't exist.".format(args.input))
    exit(1)

##############
# Get Inputs #
##############

print("Reading Training data from '{:s}'.".format(args.input))

with open(args.input, 'rb') as pickle_file:
    trX, trY, teX, teY = pickle.load(pickle_file)

print("Using {:d} Training sets and {:d} Test sets".format(len(trX), len(teX)))

####################
# Hyper Parameters #
####################

if args.input_neurons: # Use only first X features
    trX = [row[:args.input_neurons] for row in trX]
    teX = [row[:args.input_neurons] for row in teX]

input_layer = len(trX[0])
hidden_layer = args.hidden_neurons or int(input_layer * 1.5)
output_layer = 2 # len(trY[0])

print("Using first {:d} input neurons, {:d} hidden neurons, {:d} output neurons".format(input_layer, hidden_layer, output_layer))

########################
# Build Neural Network #
########################

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_o, b_h, b_o):
    # h = tf.matmul(X, w_h)
    h = tf.nn.tanh(
        tf.add(
            tf.matmul(X, w_h),
            b_h
        )
    )

    # we dont take the softmax at the end because our cost fn does that for us
    # predict = tf.matmul(h, w_o)
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
py_x = model(X, w_h, w_o , b_h, b_o)
predict_op = tf.argmax(py_x, 1) # Spam or Ham

# compute costs
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))

# construct an optimizer (Back Prop)
train_op = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(cost)

# Launch the graph in a session
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    step = 0
    cost_values = [0]

    accuracy = []
    f1Scores = []

    saver = None
    if args.save_point:
        saver = tf.train.Saver()

    # Lets train over this set a few times
    for itera in range(args.iterations):

        for start, end in zip(range(0, len(trX), args.batch_size), range(args.batch_size, len(trX), args.batch_size)):
            batchNum = start // args.batch_size

            # Lets try to predict the test now
            predictionList = sess.run(predict_op, feed_dict={X: teX, Y: teY})

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

            if baseScore[0]:
                precision = baseScore[0] / (baseScore[0] + baseScore[2])  # might not cast automatically
                recall = baseScore[0] / (baseScore[0] + baseScore[3])
                fscore = 2 * ((precision * recall) / (precision + recall))

            f1Scores.append(fscore)

            # Attempt this args.batch_size
            print("Test>> Iteration: {:d}\tBatch: {:d}\tStep: {:d}\tAccuracy: {:.7f}\tAccuracy Aggregate: {:.7f}\tFScore: {:.7f}\tFScore Aggregate: {:.7f}".format(
                itera,
                batchNum,
                step,
                accuracy[-1],
                np.average(accuracy),
                f1Scores[-1],
                np.average(f1Scores)
            ))

            # Then train on it
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

            # Log the train duration
            # print("Train>> Iteration: {:d}\tBatch: {:d}\tStep: {:d}\tTimestamp: {:.6f}".format(
            #     itera,
            #     batchNum,
            #     step,
            #     time.time()
            # ))

            step += 1

        if saver: # Save the weights
            saver.save(sess, args.save_point, global_step=itera)

        if args.tolerance:
            # Add training cost to list
            cost_values.append(sess.run(cost, feed_dict={X: trX, Y: trY}))
            diff = abs(cost_values[-1] - cost_values[-2])

            if diff < args.tolerance: # Check if we should quit
                print("Cost Converging with difference of {:.7f}".format(diff))
                break

    sess.close()
