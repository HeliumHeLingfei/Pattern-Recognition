# -*- coding: UTF-8 -*-

import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import random

mu1 = np.array([[1, 1]])
Sigma1 = np.array([[12, 0], [0, 1]])
mu2 = np.array([[7, 7]])
Sigma2 = np.array([[8, 3], [3, 2]])
mu3 = np.array([[15, 1]])
Sigma3 = np.array([[2, 0], [0, 2]])
R1 = cholesky(Sigma1)
R2 = cholesky(Sigma2)
R3 = cholesky(Sigma3)


# 二维正态分布
def gen():
	n1 = 0
	n2 = 0
	n3 = 0

	for i in range(1000):
		randnum = random.randint(1, 9)
		if randnum <= 3:
			n1 += 1
		elif randnum >= 7:
			n3 += 1
		else:
			n2 += 1
	print(n1, n2, n3)
	plt.figure(1)
	plt.subplot(211)
	s1 = np.dot(np.random.randn(n1, 2), R1) + mu1
	plt.plot(s1[:, 0], s1[:, 1], '+', 'b')
	s2 = np.dot(np.random.randn(n2, 2), R2) + mu2
	plt.plot(s2[:, 0], s2[:, 1], '+', 'g')
	s3 = np.dot(np.random.randn(n3, 2), R3) + mu3
	plt.plot(s3[:, 0], s3[:, 1], '+', 'r')
	with open('task1.txt', 'w') as f:
		f.write(str(n1) + ' blue type1----------------------------------------------------\n')
		for ii in s1.tolist():
			f.write(str(ii) + '\n')
		f.write(str(n2) + ' green type2----------------------------------------------------\n')
		for ii in s2.tolist():
			f.write(str(ii) + '\n')
		f.write(str(n3) + ' red type3----------------------------------------------------\n')
		for ii in s3.tolist():
			f.write(str(ii) + '\n')

	_n1 = 0
	_n2 = 0
	_n3 = 0
	for i in range(1000):
		randnum = random.randint(1, 10)
		if randnum <= 6:
			_n1 += 1
		elif randnum == 10:
			_n3 += 1
		else:
			_n2 += 1
	print(_n1, _n2, _n3)
	plt.subplot(212)
	_s1 = np.dot(np.random.randn(_n1, 2), R1) + mu1
	plt.plot(_s1[:, 0], _s1[:, 1], '+', 'b')
	_s2 = np.dot(np.random.randn(_n2, 2), R2) + mu2
	plt.plot(_s2[:, 0], _s2[:, 1], '+', 'g')
	_s3 = np.dot(np.random.randn(_n3, 2), R3) + mu3
	plt.plot(_s3[:, 0], _s3[:, 1], '+', 'r')
	with open('task2.txt', 'w') as f:
		f.write(str(_n1) + ' blue type1----------------------------------------------------\n')
		for ii in _s1.tolist():
			f.write(str(ii) + '\n')
		f.write(str(_n2) + ' green type2----------------------------------------------------\n')
		for ii in _s2.tolist():
			f.write(str(ii) + '\n')
		f.write(str(_n3) + ' red type3----------------------------------------------------\n')
		for ii in _s3.tolist():
			f.write(str(ii) + '\n')
	plt.savefig("sample.png")
	plt.show()

	return s1, s2, s3


#
# x1 = numpy.round(numpy.random.normal(1, 0.447, 100), 2)
# y1 = numpy.round(numpy.random.normal(1, 0.447, 100), 2)
#
# x2 = numpy.round(numpy.random.normal(1.5, 0.447, 100), 2)
# y2 = numpy.round(numpy.random.normal(1.5, 0.447, 100), 2)
#
# plt.scatter(x1, y1, c='blue', alpha=0.5)
# plt.scatter(x2, y2, c='red', alpha=0.5)
#
# plt.show()
if __name__ == '__main__':
	gen()
