#read input from features matrix and store in matrix data structures for NN processing
from random import shuffle

class EmailSet(object):
    def __init__(self,matrix_dir):
        self.matrix = self.read_matrix(matrix_dir)
        self.matrix_label = self.create_label_matrix()
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

                matrix[i] = [int(entry) for entry in line if entry != ' ']

                i += 1
                #debug
                #if i % 100 == 0:
                #    print("Reading: {:d}".format(i))
                #if i > 1600:
                #    break
        shuffle(matrix)
        return matrix

    def create_label_matrix(self):
        matrix_label = [entry[-1] for entry in self.matrix]
        return matrix_label

    def remove_label_matrix(self,matrix):
        matrix = [entry[:-1] for entry in matrix]
        return matrix

#test = EmailSet("C:\\Users\\kace\\Desktop\\Enron6 Emails\\matrixEnron6.txt")

#print(len(test.matrix))
#print(test.matrix[1])

#for entry in test.matrix_label:
#    print(entry)
#print(len(test.matrix_label))