# -*- coding: UTF-8 -*-

import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import copy
import heapq


def distance(A, B):
	sum = 0.0
	for i in range(3):
		sum += np.square(A[i] - B[i])
	return np.sqrt(sum)


# def Ho_Kashyap(D1, D2):
# 	w = [0, 1, 1, 2, 2, 0]
# 	b_min = 0.0001
# 	yita = 0.01
# 	b = []
# 	a = []
# 	P = 0
# 	for i in range(3):
# 		w0 = w[i * 2]
# 		w1 = w[i * 2 + 1]
# 		s0 = np.array([np.ones(len(D2[w0][0]))]).T
# 		s = np.array(D2[w0]).T
# 		Y0 = np.concatenate((s0, s), axis=1)
#
# 		t0 = np.array([-1 * np.ones(len(D2[w1][0]))]).T
# 		t = -1 * np.array(D2[w1]).T
# 		Y1 = np.concatenate((t0, t), axis=1)
# 		Y = np.concatenate((Y0, Y1), axis=0)
# 		_Y = np.dot(np.mat(np.dot(Y.T, Y)).I, Y.T)
# 		a0 = np.array([[1, 1, 1, 1]]).T
# 		b0 = np.array([np.ones(np.shape(Y)[0])]).T
# 		time = 0
# 		while True:
# 			time += 1
# 			e = np.array(np.dot(Y, a0) - b0)
# 			_e = 0.5 * (e + np.abs(e))
# 			b0 = b0 + 4 * yita * _e
# 			a0 = _Y * b0
# 			if np.max(_e) <= b_min:
# 				a.append(a0)
# 				b.append(b0)
# 				print(time)
# 				break
# 	print(a)
# 	for i in range(3):
# 		for id in range(100):
# 			x = np.array([[1.0], [D1[i][0][id]], [D1[i][1][id]], [D1[i][2][id]]]).T
# 			re = [x * a[0], x * a[1], x * a[2]]
# 			xtype = 0
# 			if re[0] < 0 and re[1] > 0:
# 				xtype = 1
# 			elif re[1] < 0 and re[2] > 0:
# 				xtype = 2
# 			if xtype == i:
# 				P += 1
# 			else:
# 				print(i, xtype, x)
# 	print(P/300)
# 	return P/300

def Ho_Kashyap_1(D1, D2):
	b_min = 0.0001
	yita = 0.01
	b = []
	a = []
	P = 0
	s0 = np.array([np.ones(len(D2[0][0]))]).T
	s = np.array(D2[0]).T
	T0 = np.concatenate((s0, s), axis=1)

	t0 = np.array([-1 * np.ones(len(D2[1][0]))]).T
	t = -1 * np.array(D2[1]).T
	T1 = np.concatenate((t0, t), axis=1)

	r0 = np.array([-1 * np.ones(len(D2[2][0]))]).T
	r = -1 * np.array(D2[2]).T
	T2 = np.concatenate((r0, r), axis=1)

	Y0 = np.concatenate((np.concatenate((T0, T1), axis=0), T2), axis=0)
	_Y0 = np.dot(np.mat(np.dot(Y0.T, Y0)).I, Y0.T)
	a0 = np.array([[1, 1, 1, 1]]).T
	b0 = np.array([np.ones(np.shape(Y0)[0])]).T
	time = 0
	while time < 200000:
		time += 1
		e = np.array(np.dot(Y0, a0) - b0)
		_e = 0.5 * (e + np.abs(e))
		b0 = b0 + 4 * yita * _e
		a0 = _Y0 * b0
		if np.max(_e) <= b_min:
			break
	a.append(a0)
	b.append(b0)
	print(time)

	T1 = -1 * T1

	Y1 = np.concatenate((T1, T2), axis=0)
	_Y1 = np.dot(np.mat(np.dot(Y1.T, Y1)).I, Y1.T)
	a0 = np.array([[1, 1, 1, 1]]).T
	b0 = np.array([np.ones(np.shape(Y1)[0])]).T
	time = 0
	while time < 200000:
		time += 1
		e = np.array(np.dot(Y1, a0) - b0)
		_e = 0.5 * (e + np.abs(e))
		b0 = b0 + 4 * yita * _e
		a0 = _Y1 * b0
		if np.max(_e) <= b_min:
			break
	a.append(a0)
	b.append(b0)
	print(time)
	print(a)
	for i in range(3):
		for id in range(100):
			x = np.array([[1.0], [D1[i][0][id]], [D1[i][1][id]], [D1[i][2][id]]]).T
			re = x * a[0]
			if re > 0:
				xtype = 0
			elif x * a[1] > 0:
				xtype = 1
			else:
				xtype = 2
			if xtype == i:
				P += 1
			else:
				print(i, xtype, x[0][1:4])
	print(P / 300)
	return P / 300


def Ho_Kashyap_2(D1, D2):
	b_min = 0.0001
	yita = 0.01
	b = []
	a = []
	P = 0
	s0 = np.array([np.ones(len(D2[0][0]))]).T
	s1 = np.array(D2[0]).T
	for j in range(3):
		s2 = np.array([D2[0][j] * D2[0][j]]).T
		s1 = np.concatenate((s1, s2), axis=1)
	T0 = np.concatenate((s0, s1), axis=1)

	t0 = np.array([-1 * np.ones(len(D2[1][0]))]).T
	t1 = -1 * np.array(D2[1]).T
	for j in range(3):
		t2 = -1 * np.array([D2[1][j] * D2[1][j]]).T
		t1 = np.concatenate((t1, t2), axis=1)
	T1 = np.concatenate((t0, t1), axis=1)

	r0 = np.array([-1 * np.ones(len(D2[2][0]))]).T
	r1 = -1 * np.array(D2[2]).T
	for j in range(3):
		r2 = -1 * np.array([D2[2][j] * D2[2][j]]).T
		r1 = np.concatenate((r1, r2), axis=1)
	T2 = np.concatenate((r0, r1), axis=1)

	Y0 = np.concatenate((np.concatenate((T0, T1), axis=0), T2), axis=0)
	_Y0 = np.dot(np.mat(np.dot(Y0.T, Y0)).I, Y0.T)
	a0 = np.array([[1, 1, 1, 1, 1, 1, 1]]).T
	b0 = np.array([np.ones(np.shape(Y0)[0])]).T
	time = 0
	while time < 200000:
		time += 1
		e = np.array(np.dot(Y0, a0) - b0)
		_e = 0.5 * (e + np.abs(e))
		b0 = b0 + 4 * yita * _e
		a0 = _Y0 * b0
		if np.max(_e) <= b_min:
			break
	a.append(a0)
	b.append(b0)
	print(time)

	T1 = -1 * T1

	Y1 = np.concatenate((T1, T2), axis=0)
	_Y1 = np.dot(np.mat(np.dot(Y1.T, Y1)).I, Y1.T)
	a0 = np.array([[1, 1, 1, 1, 1, 1, 1]]).T
	b0 = np.array([np.ones(np.shape(Y1)[0])]).T
	time = 0
	while time < 200000:
		time += 1
		e = np.array(np.dot(Y1, a0) - b0)
		_e = 0.5 * (e + np.abs(e))
		b0 = b0 + 4 * yita * _e
		a0 = _Y1 * b0
		if np.max(_e) <= b_min:
			break
	a.append(a0)
	b.append(b0)
	print(time)
	print(a)

	for i in range(3):
		for id in range(100):
			x = np.array([[1.0], [D1[i][0][id]], [D1[i][1][id]], [D1[i][2][id]], [D1[i][0][id] * D1[i][0][id]],
			              [D1[i][1][id] * D1[i][1][id]], [D1[i][2][id] * D1[i][2][id]]]).T
			re = x * a[0]
			if re > 0:
				xtype = 0
			elif x * a[1] > 0:
				xtype = 1
			else:
				xtype = 2
			if xtype == i:
				P += 1
			else:
				print(i, xtype, x[0][1:4])
	print(P / 300)
	return P / 300


def Ho_Kashyap_3(D1, D2):
	b_min = 0.0001
	yita = 0.01
	b = []
	a = []
	P = 0
	s0 = np.array([np.ones(len(D2[0][0]))]).T
	s1 = np.array(D2[0]).T
	for j in range(3):
		s2 = np.array([D2[0][j] * D2[0][(j + 1) % 3]]).T
		s1 = np.concatenate((s1, s2), axis=1)
	T0 = np.concatenate((s0, s1), axis=1)

	t0 = np.array([-1 * np.ones(len(D2[1][0]))]).T
	t1 = -1 * np.array(D2[1]).T
	for j in range(3):
		t2 = -1 * np.array([D2[1][j] * D2[1][(j + 1) % 3]]).T
		t1 = np.concatenate((t1, t2), axis=1)
	T1 = np.concatenate((t0, t1), axis=1)

	r0 = np.array([-1 * np.ones(len(D2[2][0]))]).T
	r1 = -1 * np.array(D2[2]).T
	for j in range(3):
		r2 = -1 * np.array([D2[2][j] * D2[2][(j + 1) % 3]]).T
		r1 = np.concatenate((r1, r2), axis=1)
	T2 = np.concatenate((r0, r1), axis=1)

	Y0 = np.concatenate((np.concatenate((T0, T1), axis=0), T2), axis=0)
	_Y0 = np.dot(np.mat(np.dot(Y0.T, Y0)).I, Y0.T)
	a0 = np.array([[1, 1, 1, 1, 1, 1, 1]]).T
	b0 = np.array([np.ones(np.shape(Y0)[0])]).T
	time = 0
	while time < 200000:
		time += 1
		e = np.array(np.dot(Y0, a0) - b0)
		_e = 0.5 * (e + np.abs(e))
		b0 = b0 + 4 * yita * _e
		a0 = _Y0 * b0
		if np.max(_e) <= b_min:
			break
	a.append(a0)
	b.append(b0)
	print(time)

	T1 = -1 * T1

	Y1 = np.concatenate((T1, T2), axis=0)
	_Y1 = np.dot(np.mat(np.dot(Y1.T, Y1)).I, Y1.T)
	a0 = np.array([[1, 1, 1, 1, 1, 1, 1]]).T
	b0 = np.array([np.ones(np.shape(Y1)[0])]).T
	time = 0
	while time < 200000:
		time += 1
		e = np.array(np.dot(Y1, a0) - b0)
		_e = 0.5 * (e + np.abs(e))
		b0 = b0 + 4 * yita * _e
		a0 = _Y1 * b0
		if np.max(_e) <= b_min:
			break
	a.append(a0)
	b.append(b0)
	print(time)
	print(a)

	for i in range(3):
		for id in range(100):
			x = np.array([[1.0], [D1[i][0][id]], [D1[i][1][id]], [D1[i][2][id]], [D1[i][0][id] * D1[i][1][id]],
			              [D1[i][1][id] * D1[i][2][id]], [D1[i][2][id] * D1[i][0][id]]]).T
			re = x * a[0]
			if re > 0:
				xtype = 0
			elif x * a[1] > 0:
				xtype = 1
			else:
				xtype = 2
			if xtype == i:
				P += 1
			else:
				print(i, xtype, x[0][1:4])
	print(P / 300)
	return P / 300


def Ho_Kashyap_4(D1, D2):
	b_min = 0.0001
	yita = 0.01
	b = []
	a = []
	P = 0
	s0 = np.array([np.ones(len(D2[0][0]))]).T
	s1 = np.array(D2[0]).T
	for j in range(3):
		s2 = np.array([D2[0][j] * D2[0][j]]).T
		s3 = np.array([D2[0][j] * D2[0][(j + 1) % 3]]).T
		s2 = np.concatenate((s2, s3), axis=1)
		s1 = np.concatenate((s1, s2), axis=1)
	T0 = np.concatenate((s0, s1), axis=1)

	t0 = np.array([-1 * np.ones(len(D2[1][0]))]).T
	t1 = -1 * np.array(D2[1]).T
	for j in range(3):
		t2 = -1 * np.array([D2[1][j] * D2[1][j]]).T
		t3 = -1 * np.array([D2[1][j] * D2[1][(j + 1) % 3]]).T
		t2 = np.concatenate((t2, t3), axis=1)
		t1 = np.concatenate((t1, t2), axis=1)
	T1 = np.concatenate((t0, t1), axis=1)

	r0 = np.array([-1 * np.ones(len(D2[2][0]))]).T
	r1 = -1 * np.array(D2[2]).T
	for j in range(3):
		r2 = -1 * np.array([D2[2][j] * D2[2][j]]).T
		r3 = -1 * np.array([D2[2][j] * D2[2][(j + 1) % 3]]).T
		r2 = np.concatenate((r2, r3), axis=1)
		r1 = np.concatenate((r1, r2), axis=1)
	T2 = np.concatenate((r0, r1), axis=1)
	Y0 = np.concatenate((np.concatenate((T0, T1), axis=0), T2), axis=0)
	_Y0 = np.dot(np.mat(np.dot(Y0.T, Y0)).I, Y0.T)
	a0 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).T
	b0 = np.array([np.ones(np.shape(Y0)[0])]).T
	time = 0
	while time < 200000:
		time += 1
		e = np.array(np.dot(Y0, a0) - b0)
		_e = 0.5 * (e + np.abs(e))
		b0 = b0 + 4 * yita * _e
		a0 = _Y0 * b0
		if np.max(_e) <= b_min:
			break
	a.append(a0)
	b.append(b0)
	print(time)

	T1 = -1 * T1

	Y1 = np.concatenate((T1, T2), axis=0)
	_Y1 = np.dot(np.mat(np.dot(Y1.T, Y1)).I, Y1.T)
	a0 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).T
	b0 = np.array([np.ones(np.shape(Y1)[0])]).T
	time = 0
	while time < 200000:
		time += 1
		e = np.array(np.dot(Y1, a0) - b0)
		_e = 0.5 * (e + np.abs(e))
		b0 = b0 + 4 * yita * _e
		a0 = _Y1 * b0
		if np.max(_e) <= b_min:
			break
	a.append(a0)
	b.append(b0)
	print(time)
	print(a)

	for i in range(3):
		for id in range(100):
			x = np.array([[1.0], [D1[i][0][id]], [D1[i][1][id]], [D1[i][2][id]], [D1[i][0][id] * D1[i][0][id]],
			              [D1[i][0][id] * D1[i][1][id]],
			              [D1[i][1][id] * D1[i][1][id]], [D1[i][1][id] * D1[i][2][id]], [D1[i][2][id] * D1[i][2][id]],
			              [D1[i][2][id] * D1[i][0][id]]]).T
			re = x * a[0]
			if re > 0:
				xtype = 0
			elif x * a[1] > 0:
				xtype = 1
			else:
				xtype = 2
			if xtype == i:
				P += 1
			else:
				print(i, xtype, x[0][1:4])
	print(P / 300)
	return P / 300


def gen():
	n0 = 100
	n = [1000, 600, 1600]
	mu = [[1, 1, 1], [3, 3, 3], [7, 8, 9]]
	sigma = np.sqrt([[1, 1, 1], [2, 3, 4], [6, 6, 9]])

	D1 = []
	D2 = []  # category.d.id
	for i in range(3):
		s = []
		t = []
		for k in range(3):
			s.append(np.random.normal(mu[i][k], sigma[i][k], n0))
			t.append(np.random.normal(mu[i][k], sigma[i][k], n[i]))
		D1.append(copy.deepcopy(s))
		D2.append(copy.deepcopy(t))
	with open("D1.txt", 'w') as f:
		for i in range(3):
			f.write(str(i) + '\n')
			for id in range(len(D1[i][0])):
				f.write(str(D1[i][0][id]) + '\t' + str(D1[i][1][id]) + '\t' + str(D1[i][2][id]) + '\n')

	with open("D2.txt", 'w') as f:
		for i in range(3):
			f.write(str(i) + '\n')
			for id in range(len(D2[i][0])):
				f.write(str(D2[i][0][id]) + '\t' + str(D2[i][1][id]) + '\t' + str(D2[i][2][id]) + '\n')

	ax = plt.subplot(111, projection='3d')
	ax.scatter(D1[0][0], D1[0][1], D1[0][2], c='y')
	ax.scatter(D1[1][0], D1[1][1], D1[1][2], c='r')
	ax.scatter(D1[2][0], D1[2][1], D1[2][2], c='g')

	ax.set_zlabel('Z')
	ax.set_ylabel('Y')
	ax.set_xlabel('X')
	plt.savefig("fig1.png")
	# plt.show()
	return D1, D2


def knn(n, D1, D2):
	P = 0
	for ca in range(3):
		for i in range(100):
			point = [D1[ca][0][i], D1[ca][1][i], D1[ca][2][i]]
			heap = []
			for category in range(3):
				for j in range(len(D2[category][0])):
					sample = [D2[category][0][j], D2[category][1][j], D2[category][2][j]]
					heap.append({'dis': distance(sample, point), 'cat': category, 'id': j})
			ns = heapq.nsmallest(n, heap, key=lambda s: s['dis'])
			type = [0, 0, 0]
			result = random.randint(0, 2)
			for sample in ns:
				type[sample['cat']] += 1
			for k in range(0, 3):
				if type[k] > type[result]:
					result = k
			if result == ca:
				P += 1
			else:
				print(ca, result, point, ns)
	print(P / 300)
	return P / 300


if __name__ == '__main__':
	D1, D2 = gen()
	P1_0 = knn(3, D1, D2)
	P1_1 = knn(7, D1, D2)
	P1_2 = knn(9, D1, D2)
	P2 = Ho_Kashyap_1(D1, D2)
	P3_1 = Ho_Kashyap_2(D1, D2)
	P3_2 = Ho_Kashyap_3(D1, D2)
	P3_3 = Ho_Kashyap_4(D1, D2)
	print(P1_0, P1_1, P1_2, P2, P3_1, P3_2, P3_3)
