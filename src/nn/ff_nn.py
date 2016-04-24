#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import time
import argparse
import os
import pickle
import math

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
parser.add_argument('-x', '--tolerance', type=float, default=0)
parser.add_argument('-s', '--save-point')
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--no-train', dest='train', action='store_false')
parser.set_defaults(train=True)
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

if args.train:
    print("Reading Training data from '{:s}'.".format(args.input))

    with open(args.input, 'rb') as pickle_file:
        trX, trY, teX, teY = pickle.load(pickle_file)

    print("Using {:d} Training sets and {:d} Test sets".format(len(trX), len(teX)))
else:
    print("Reading Test data from '{:s}'.".format(args.input))

    with open(args.input, 'rb') as pickle_file:
        teX = pickle.load(pickle_file)

####################
# Hyper Parameters #
####################

if args.input_neurons: # Use only first X features
    if args.train:
        trX = [row[:args.input_neurons] for row in trX]
    teX = [row[:args.input_neurons] for row in teX]

input_layer = len(teX[0])
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
    mcclist = []

    saver = None
    if args.save_point:
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(args.save_point)
        if ckpt and ckpt.model_checkpoint_path:
            print("Using weights from", ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

    if args.train: # Lets train over this set a few times
        for itera in range(args.iterations):
            # Train in batches
            for start, end in zip(range(0, len(trX), args.batch_size), range(args.batch_size, len(trX), args.batch_size)):
                batchNum = start // args.batch_size

                # Lets try to predict the test now
                predictionList = sess.run(predict_op, feed_dict={X: teX, Y: teY})

                # Calculate accuracy and save it to accuracy list
                accuracy.append(np.mean(np.argmax(teY, axis=1) == predictionList))

                # Calc f score and save to list
                baselist = [0, 0, 0, 0]  # tp,tn,fp,fn
                mcc = 0

                for i in range(0, len(teY)):
                    if np.argmax(teY[i]) == 1:  # email is legitimate
                        if predictionList[i] == 1:  # predicted legitimate as legitimate    (true positive)
                            baselist[0] += 1
                        else:  # predicted legitimate as spam          (false negative)
                            baselist[3] += 1
                    else:  # email is spam
                        if predictionList[i] == 1:  # predicted spam as legitimate          (false positive)
                            baselist[2] += 1
                        else:  # predicted spam as spam                (true negative)
                            baselist[1] += 1

                tp = baselist[0]
                tn = baselist[1]
                fp = baselist[2]
                fn = baselist[3]

                if tp:
                    # precision = baselist[0] / (baselist[0] + baselist[2])  # might not cast automatically
                    # recall = baselist[0] / (baselist[0] + baselist[3])
                    # fscore = 2 * ((precision * recall) / (precision + recall))
                    mcc = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

                mcclist.append(mcc)

                # Attempt this args.batch_size
                print("Test>> Iteration: {:d}\tBatch: {:d}\tStep: {:d}\tAccuracy: {:.7f}\tAccuracy Aggregate: {:.7f}\tMCC: {:.7f}\tMCC Aggregate: {:.7f}".format(
                    itera,
                    batchNum,
                    step,
                    accuracy[-1],
                    np.average(accuracy),
                    mcclist[-1],
                    np.average(mcclist)
                ))

                # Then train on it
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

                # Log the train duration
                print("Train>> Iteration: {:d}\tBatch: {:d}\tStep: {:d}\tTimestamp: {:.6f}".format(
                    itera,
                    batchNum,
                    step,
                    time.time()
                ))

                step += 1

            if saver and args.train: # Save the weights if training
                saver.save(sess, args.save_point, global_step=itera)

            if args.tolerance: # Check if cost is still decreasing
                # Add training cost to list
                cost_values.append(sess.run(cost, feed_dict={X: trX, Y: trY}))
                diff = abs(cost_values[-1] - cost_values[-2])

                if diff < args.tolerance: # Check if we should quit
                    print("Cost Converging with difference of {:.7f}".format(diff))
                    break

        print("Finished Training")
    else: # test only given set
        predictionList = sess.run(py_x, feed_dict={X: teX})
        for i, prob in enumerate(predictionList):
            print(i, np.argmax(prob), prob[1] - prob[0], prob)

    sess.close()
