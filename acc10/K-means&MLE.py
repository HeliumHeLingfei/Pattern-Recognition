# -*- coding: UTF-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

fignum = 0

def distance(A, B):
	sum = 0.0
	for i in range(len(A)):
		sum += np.square(A[i] - B[i])
	return np.sqrt(sum)


def gen():
	n0 = 100
	n = [1000, 600, 1600]
	mu = [[1, 1, 1], [3, 3, 3], [7, 8, 9]]
	sigma = np.sqrt([[1, 1, 1], [2, 3, 4], [6, 6, 9]])

	D1 = []
	D2 = [[], [], []]  # d.id
	for i in range(3):
		s = []
		for k in range(3):
			s.append(np.random.normal(mu[i][k], sigma[i][k], n0))
			D2[k].extend(np.random.normal(mu[i][k], sigma[i][k], n[i]))
		D1.append(copy.deepcopy(s))

	with open("D1.txt", 'w') as f:
		for i in range(3):
			f.write(str(i) + '\n')
			for id in range(len(D1[i][0])):
				f.write(str(D1[i][0][id]) + '\t' + str(D1[i][1][id]) + '\t' + str(D1[i][2][id]) + '\n')

	with open("D2.txt", 'w') as f:
		for id in range(len(D2[0])):
			f.write(str(D2[0][id]) + '\t' + str(D2[1][id]) + '\t' + str(D2[2][id]) + '\n')

	ax = plt.subplot(111, projection='3d')
	ax.scatter(D2[0][:1000], D2[1][:1000], D2[2][:1000], c='y')
	ax.scatter(D2[0][1000:1600], D2[1][1000:1600], D2[2][1000:1600], c='r')
	ax.scatter(D2[0][1600:], D2[1][1600:], D2[2][1600:], c='g')

	ax.set_zlabel('Z')
	ax.set_ylabel('Y')
	ax.set_xlabel('X')
	global fignum
	plt.savefig("fig" + str(fignum) + ".png")
	fignum+=1
	plt.show()
	D2 = np.array(D2).T
	for i in range(3):
		D1[i] = np.array(D1[i]).T
	D1 = np.array(D1)
	return D1, D2


# category.d.id
def k_means(c, D1, D2):
	n = len(D2)
	mu = []
	muid = np.random.choice(n, c, replace=False)
	points = []
	for i in range(c):
		mu.append(D2[muid[i]])
		points.append([])
	while True:
		for pid in range(n):
			point = D2[pid]
			ptype = 0
			dis = 100
			for ca in range(c):
				tem_dis = distance(mu[ca], point)
				if tem_dis < dis:
					dis = tem_dis
					ptype = ca
			points[ptype].append(point)
		dis = 0.0
		for i in range(c):
			oldmu = mu[i]
			mu[i] = np.mean(points[i], axis=0)
			if distance(oldmu, mu[i]) > dis:
				dis = distance(oldmu, mu[i])
		if dis < 0.00001:
			break
		else:
			for i in range(c):
				points[i] = []
	sigma2 = []
	p = []
	sum = 0
	for i in range(c):
		sigma2.append(np.var(points[i], axis=0))
		p.append(len(points[i]))
		sum += len(points[i])
	p = np.array(p) / sum
	ax = plt.subplot(111, projection='3d')
	color = ['y', 'r', 'g']
	for i in range(c):
		paint = np.array(points[i]).T
		if len(paint) < 1:
			continue
		ax.scatter(paint[0], paint[1], paint[2], c=color[i])

	ax.set_zlabel('Z')
	ax.set_ylabel('Y')
	ax.set_xlabel('X')
	global fignum
	plt.savefig("fig" + str(fignum) + ".png")
	fignum+=1
	plt.show()
	return p, np.array(mu), np.array(sigma2)


def MLE(c, D1, D2, p, mu, sigma):
	n = len(D2)
	if p == []:
		p = np.random.rand(1, c)
		p = p / np.sum(p)
		p = p[0]
		mu = np.random.randint(1, 10, size=(c, 3))
		sigma = np.random.randint(1, 10, size=(c, 3))

	pw = []
	for k in range(n):
		pw.append([])
		for i in range(c):
			pw[k].append(1.0 / c)
	pw = np.array(pw)

	def get_pw():
		for k in range(n):
			for i in range(c):
				det = sigma[i][0] * sigma[i][1] * sigma[i][2]
				exponent = -0.5 * np.dot((D2[k] - mu[i]) / sigma[i], D2[k] - mu[i])
				pw[k][i] = np.power(det, -0.5) * math.exp(exponent) * p[i]
			pw[k] = pw[k] / np.sum(pw[k])

	get_pw()

	while True:
		p = []
		mu = []
		sigma = []
		for i in range(c):
			p.append(0.0)
			mu.append([0.0, 0.0, 0.0])
			sigma.append([0.0, 0.0, 0.0])
		p = np.array(p)
		mu = np.array(mu)
		sigma = np.array(sigma)
		for k in range(n):
			p += pw[k]
			for i in range(c):
				mu[i] += pw[k][i] * D2[k]
		for i in range(c):
			mu[i] /= p[i]
		for k in range(n):
			for i in range(c):
				sigma[i] += pw[k][i] * np.power(D2[k] - mu[i], 2)
		for i in range(c):
			sigma[i] /= p[i]
		p /= n
		oldpw = copy.deepcopy(pw)
		get_pw()
		d = 0.0
		for k in range(n):
			if distance(oldpw[k], pw[k]) > d:
				d = distance(oldpw[k], pw[k])
		if d < 0.00001:
			break
	points = []
	for i in range(c):
		points.append([])
	for k in range(n):
		type = 0
		for i in range(1, c):
			if pw[k][type] < pw[k][i]:
				type = i
		points[type].append(D2[k])

	ax = plt.subplot(111, projection='3d')
	color = ['y', 'r', 'g']
	for i in range(c):
		paint = np.array(points[i]).T
		if len(paint) < 1:
			continue
		ax.scatter(paint[0], paint[1], paint[2], c=color[i])

	ax.set_zlabel('Z')
	ax.set_ylabel('Y')
	ax.set_xlabel('X')
	global fignum
	plt.savefig("fig" + str(fignum) + ".png")
	fignum+=1
	plt.show()
	return p, mu, sigma


def task5(D1, mu_4):
	dtype = [('data', 'float'), ('type', 'int')]
	mu_type = np.array([(mu_4[0][2], 0), (mu_4[1][2], 1), (mu_4[2][2], 2)], dtype=dtype)
	seq = np.sort(mu_type, order='data')
	acc = [0, 0, 0]
	rpoints = []
	wpoints = []
	for c in range(3):
		for i in range(100):
			type = 0
			dis = 100
			for j in range(3):
				tem_dis = distance(D1[c][i], mu_4[seq[j][1]])
				if tem_dis < dis:
					dis = tem_dis
					type = j
			if type == c:
				acc[c] += 1
				rpoints.append(D1[c][i])
			else:
				wpoints.append(D1[c][i])
	print(acc)
	ax = plt.subplot(111, projection='3d')
	rpaint = np.array(rpoints).T
	ax.scatter(rpaint[0], rpaint[1], rpaint[2], c='g')
	wpaint = np.array(wpoints).T
	ax.scatter(wpaint[0], wpaint[1], wpaint[2], c='r')
	ax.set_zlabel('Z')
	ax.set_ylabel('Y')
	ax.set_xlabel('X')
	global fignum
	plt.savefig("fig" + str(fignum) + ".png")
	fignum+=1
	plt.show()

	acc = [0, 0]
	rpoints = []
	wpoints = []
	for c in range(3):
		for i in range(100):
			type = 0
			dis = 100
			for j in range(3):
				tem_dis = distance(D1[c][i], mu_4[seq[j][1]])
				if tem_dis < dis:
					dis = tem_dis
					type = j
			if c == 0 or c == 1:
				if type == 0:
					acc[0] += 1
					rpoints.append(D1[c][i])
				else:
					wpoints.append(D1[c][i])
			else:
				if type != 0:
					acc[1] += 1
					rpoints.append(D1[c][i])
				else:
					wpoints.append(D1[c][i])

	print(acc)
	ax = plt.subplot(111, projection='3d')
	rpaint = np.array(rpoints).T
	ax.scatter(rpaint[0], rpaint[1], rpaint[2], c='g')
	wpaint = np.array(wpoints).T
	ax.scatter(wpaint[0], wpaint[1], wpaint[2], c='r')
	ax.set_zlabel('Z')
	ax.set_ylabel('Y')
	ax.set_xlabel('X')
	plt.savefig("fig" + str(fignum) + ".png")
	fignum+=1
	plt.show()
	print('_____________________________________')


def task6(D1, p_5, mu_5, sigma_5):
	dtype = [('data', 'float'), ('type', 'int')]
	mu_type = np.array([(mu_5[0][2], 0), (mu_5[1][2], 1), (mu_5[2][2], 2)], dtype=dtype)
	seq = np.sort(mu_type, order='data')
	acc = [0, 0, 0]
	rpoints = []
	wpoints = []
	for c in range(3):
		for i in range(100):
			type = 0
			dis = 0
			for j in range(3):
				det = np.sqrt(sigma_5[seq[j][1]][0] * sigma_5[seq[j][1]][1] * sigma_5[seq[j][1]][2])
				tem = 1 / (np.power(2 * np.pi, 1.5) * det)
				exponent = -0.5 * np.dot((D1[c][i] - mu_5[seq[j][1]]) / sigma_5[seq[j][1]], D1[c][i] - mu_5[seq[j][1]])
				tem_dis = p_5[seq[j][1]] * tem * np.exp(exponent)
				if tem_dis > dis:
					dis = tem_dis
					type = j
			if type == c:
				acc[c] += 1
				rpoints.append(D1[c][i])
			else:
				wpoints.append(D1[c][i])
	print(acc)
	ax = plt.subplot(111, projection='3d')
	rpaint = np.array(rpoints).T
	ax.scatter(rpaint[0], rpaint[1], rpaint[2], c='g')
	wpaint = np.array(wpoints).T
	ax.scatter(wpaint[0], wpaint[1], wpaint[2], c='r')
	ax.set_zlabel('Z')
	ax.set_ylabel('Y')
	ax.set_xlabel('X')
	global fignum
	plt.savefig("fig" + str(fignum) + ".png")
	fignum+=1
	plt.show()
	print('_____________________________________')


if __name__ == '__main__':
	D1, D2 = gen()
	p_1, mu_1, sigma_1 = k_means(2, D1, D2)
	print('1:\n', p_1, '\n', mu_1, '\n', sigma_1)
	print('_____________________________________')
	p_2, mu_2, sigma_2 = MLE(2, D1, D2, [], [], [])
	print('2:\n', p_2, '\n', mu_2, '\n', sigma_2)
	print('_____________________________________')
	p_3, mu_3, sigma_3 = MLE(2, D1, D2, p_1, mu_1, sigma_1)
	print('3:\n', p_3, '\n', mu_3, '\n', sigma_3)
	print('_____________________________________')
	p_4, mu_4, sigma_4 = k_means(3, D1, D2)
	print('4:\n', p_4, '\n', mu_4, '\n', sigma_4)
	task5(D1, mu_4)
	p_5, mu_5, sigma_5 = MLE(3, D1, D2, [], [], [])
	print('5:\n', p_5, '\n', mu_5, '\n', sigma_5)
	task6(D1, p_5, mu_5, sigma_5)
	p_6, mu_6, sigma_6 = MLE(3, D1, D2, p_4, mu_4, sigma_4)
	print('6:\n', p_6, '\n', mu_6, '\n', sigma_6)
	task6(D1, p_6, mu_6, sigma_6)
