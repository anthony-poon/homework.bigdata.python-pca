#!/usr/bin/python
import argparse
import pprint
import numpy
import scipy.sparse
import math
from math import exp
import time
from sklearn.decomposition import PCA

start_time = time.time()
# Parsing argument
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", help="Input path of the training data")
parser.add_argument("-r", "--row", help="Row count")
parser.add_argument("-c", "--column", help="Column count")
#parser.add_argument("-e", "--exp", help="Pass values to exponentiation function", default=False, action='store_true')
#parser.add_argument("-l", "--low", help="Low bound of value. default = 0", default=0)
#parser.add_argument("-u", "--up", help="Up bound of value. default = 1", default=1)
args = parser.parse_args()

#low_bound = int(args.low)
#up_bound = int(args.up)

print "T+", round(time.time() - start_time), " seconds: Loading from files"

# Load file into an matrix with size specified by argument
train_file = open(args.train, 'r')
train_matrix = numpy.zeros(shape=(int(args.row),int(args.column)))
for line in train_file:
	str_array = line.split('\t')
	train_matrix[int(str_array[0])-1][int(str_array[1])-1] = float(str_array[2])

# Get row average and fill into empty space
for row_index in range(train_matrix.shape[0]):
	sum = 0
	count = 0
	for col_index in range(train_matrix[row_index].shape[0]):
		# if value is 0 sum is unchanged
		sum = sum + train_matrix[row_index][col_index]
		# Accounting for float point acc. and check if cell is 0, if not zero count ++
		if abs(train_matrix[row_index][col_index]) > 0.0001:
			count = count + 1
			
	if (count > 0):
		average = sum / count
	else:
		average = 0
		
	for col_index in range(train_matrix[row_index].shape[0]):
		# Again, only do if cell is zero
		if abs(train_matrix[row_index][col_index]) <= 0.0001:
			train_matrix[row_index][col_index] = average
#if (args.exp):
#	for cell in numpy.nditer(train_matrix, op_flags=['readwrite']):
#		cell[...] = 1 / (1 + math.exp(-value))

# Saving original matrix to a csv
numpy.savetxt("non_sparse_data.csv", train_matrix.astype(float), fmt="%.4f", delimiter=',')
print "T+", round(time.time() - start_time), " seconds: Getting covar_matrix"

# Saving covariance matrix to a csv
covar_matrix = numpy.cov(train_matrix, rowvar=False)
numpy.savetxt("covar.csv", covar_matrix.astype(float), fmt="%.4f", delimiter=',')

# Get and save eigen data to a csv
print "T+", round(time.time() - start_time), " seconds: Getting eigen"
eigen_value, eigen_vector = numpy.linalg.eig(covar_matrix)
numpy.savetxt("eigen_value.csv", eigen_value.astype(float), fmt="%.4f", delimiter=',')
numpy.savetxt("eigen_vector.csv", eigen_vector.astype(float), fmt="%.4f", delimiter=',')
print "T+", round(time.time() - start_time), " seconds: Finish"