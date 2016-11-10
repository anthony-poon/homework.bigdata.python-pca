#!/usr/bin/python
import argparse
import numpy
import time
import math

numpy.set_printoptions(precision=4)
start_time = time.time()

# Parse argument
parser = argparse.ArgumentParser()
parser.add_argument("-d","--data", help="Input path of the non sparse train data")
parser.add_argument("-r", help="Row count")
parser.add_argument("-c", help="Column count")
parser.add_argument("-x", help="Eign value matrix path")
parser.add_argument("-v", help="Eign vector")
parser.add_argument("-t", "--test", help="Input path of test data")
parser.add_argument("-n", help="Number of Eigen Vector to keep", default=0)
#parser.add_argument("-e", "--exp", help="Restore from exp function.", default=False, action="store_true")
args = parser.parse_args()

print "T+", round(time.time() - start_time), " seconds: Loading data"

# Load data from csv
eigen_vector = numpy.loadtxt(args.v, delimiter=",")
eigen_value = numpy.loadtxt(args.x, delimiter=",")
eigen_vector_count = int(args.n)

print "T+", round(time.time() - start_time), " seconds: Sorting and removing unused eigen data"
# eigen_value[:, None] change array into column vector, numpy.concatenate glue the eigen_vector to the eigen_value
# The result become 
# [eigen_value_1, eigen_vector_1[0], eigen_vector_1[1]] 
# [eigen_value_2, eigen_vector_2[0], eigen_vector_2[1]] 
# [eigen_value_3, eigen_vector_3[0], eigen_vector_3[1]], etc
combined = numpy.concatenate((eigen_value[:, None], eigen_vector.T), axis=1)

# Sort descending by left most column, which is the eigen_value since numpy.linalg.eig do not guarantee order
combined = combined[combined[:,0].argsort()[::-1]]

# if a eigen count is specified, only get the first n row
if (eigen_vector_count > 0):
	combined = combined[0:eigen_vector_count]
	
# chop first column to be eigen_value, secound to the end column to be eigen vector, transpose back
eigen_value = combined[:, 0:1]
eigen_vector = combined[:, 1:].T

print "T+", round(time.time() - start_time), " seconds: Reconstructing"

data = numpy.loadtxt(args.data, delimiter=",")
score_matrix = data.dot(eigen_vector)
# reconstruct the matrix
reconstruct = score_matrix.dot(eigen_vector.T)

test_file = open(args.test, 'r')
output_file = open('score-' + str(eigen_vector_count) + ".txt", 'w')
for line in test_file:
	str_array = line.split('\t')
	# get i-1, j-1 cell sinces the input is 1 based and matrix is 0 based 
	output_file.write(str(numpy.round(reconstruct[int(str_array[0])-1][int(str_array[1])-1], 3))+"\n")
	
print "T+", round(time.time() - start_time), " seconds: Finished"

#if (args.exp) :
#	for cell in numpy.nditer(reconstruct, op_flags=['readwrite']):
#		value = - math.log((1/cell) - 1)
#		cell[...] = value
#for row in score_matrix:
#	print row.dot(eigen_vector.T)