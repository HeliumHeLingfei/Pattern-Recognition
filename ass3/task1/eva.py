# -*- coding: UTF-8 -*-
from gen import gen

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


def MLE(s):
	mu = np.mean(s, axis=0)
	sigma = np.array([[0, 0], [0, 0]])
	for i in s:
		k = i - mu
		m = np.array([[k[0] * k[0], k[0] * k[1]], [k[1] * k[0], k[1] * k[1]]])
		sigma = sigma + m
	sigma1 = sigma / len(s)
	return mu, sigma1


def BE(s, sigma):
	mu = np.mean(s, axis=0)
	mu0 = np.array([1, 1])
	sigma0 = np.array([[1, 0], [0, 1]])
	n = len(s)
	tem = np.linalg.inv(sigma0 + sigma / n)
	mun = sigma0.dot(tem.dot(mu)) + sigma.dot(tem.dot(mu0)) / n
	sigman = sigma0.dot(tem.dot(sigma / n))
	return mun, sigman + sigma


def main():
	s1, s2, s3 = gen()
	mu, sigma = MLE(s1)
	print('MLE1:', mu)
	print(sigma)
	mu, sigma = MLE(s2)
	print('MLE2:', mu)
	print(sigma)
	mu, sigma = MLE(s3)
	print('MLE3:', mu)
	print(sigma)
	mu, sigma = BE(s1, Sigma1)
	print('BE1:', mu)
	print(sigma)
	mu, sigma = BE(s2, Sigma2)
	print('BE2:', mu)
	print(sigma)
	mu, sigma = BE(s3, Sigma3)
	print('BE3:', mu)
	print(sigma)

	print('\nsecond trial:')
	t1 = []
	t2 = []
	t3 = []
	for i in range(300):
		randnum = random.randint(1, 9)
		if randnum <= 3:
			while True:
				tem = random.randint(0, len(s1) - 1)
				if tem in t1:
					continue
				else:
					t1.append(tem)
					break
		elif randnum >= 7:
			while True:
				tem = random.randint(0, len(s2) - 1)
				if tem in t2:
					continue
				else:
					t2.append(tem)
					break
		else:
			while True:
				tem = random.randint(0, len(s3) - 1)
				if tem in t3:
					continue
				else:
					t3.append(tem)
					break
	print(len(t1), len(t2), len(t3))

	q1 = []
	q2 = []
	q3 = []
	for i in t1:
		q1.append(s1[i])
	for i in t2:
		q2.append(s2[i])
	for i in t3:
		q3.append(s3[i])
	mu, sigma = MLE(q1)
	print('MLE1:', mu)
	print(sigma)
	mu, sigma = MLE(q2)
	print('MLE2:', mu)
	print(sigma)
	mu, sigma = MLE(q3)
	print('MLE3:', mu)
	print(sigma)
	mu, sigma = BE(q1, Sigma1)
	print('BE1:', mu)
	print(sigma)
	mu, sigma = BE(q2, Sigma2)
	print('BE2:', mu)
	print(sigma)
	mu, sigma = BE(q3, Sigma3)
	print('BE3:', mu)
	print(sigma)


if __name__ == '__main__':
	main()
