#!/usr/bin/env python3

import pickle
import argparse
import os
from random import shuffle

###################
# Of Machine & Men
#
# Maurice Prosper
# kace echo
# Cole Troutman
#
# CS 3359
#
# Save a feature set as a pickled dump
#
###################

parser = argparse.ArgumentParser(description='Turn feature list into pickle dump')
parser.add_argument('-i', '--input')
parser.add_argument('-o', '--output', default='')
parser.add_argument('-t', '--train', type=float, default=0.0)
args = parser.parse_args()

if(not args.input):
    print("Input file file not given.")
    parser.print_help()
    exit(1)

if(not os.path.isfile(args.input)):
    print("File {:s} doesn't exist.".format(args.input))
    exit(1)


########
# read input from features matrix
# store in matrix data structures for NN processing
########

class EmailSet(object):
    def __init__(self,matrix_dir,labeled):
        all = self.read_matrix(matrix_dir,labeled)

        if labeled :
            self.labels = [row[-2:] for row in all]
            self.matrix = [row[:-2] for row in all]
        else:
            self.matrix = all

    def read_matrix(self,matrix_dir,labeled):
        matrix = []

        with open(matrix_dir,"r") as matrix_file:
            i = 0
            for line in matrix_file:
                matrix.append([])

                line = line.strip().split(' ')

                if labeled : # this could be taken out by modifying FEATURES project output format for labels
                    line[ -1 ] = 1 if line[-1] == 'S' else 0
                    line.append(1 if line[-1] == 0 else 0)

                matrix[i] = [int(entry) for entry in line if entry != ' ']
                i += 1

        shuffle(matrix)
        return matrix

def save_object(obj, filename):
    if args.output: # save to file
        with open(filename, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    else:
        print(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))

if args.train: # Training Data
    emails = EmailSet(args.input, True)
    size = int(len(emails.matrix) * args.train)
    save_object([
        emails.matrix[:size],  # training emails
        emails.labels[:size],  # training labels
        emails.matrix[size:],  # testing emails
        emails.labels[size:]  # testing labels
    ], args.output)
else:
    emails = EmailSet(args.input, False)
    save_object(emails.matrix, args.output)
