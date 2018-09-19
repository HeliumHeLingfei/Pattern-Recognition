import numpy as np

dtype = np.float64
import copy

S = []
S.append(np.array([0, 0, 1, 1, 2, 2, 3, 3, 4]))
S.append(np.array([0, 1, 1, 2, 1, 1, 3, 3, 4]))
S.append(np.array([0, 2, 1, 2, 1, 2, 3, 4]))
S.append(np.array([0, 3, 4]))
S.append(np.array([0, 2, 1, 2, 1, 0, 1, 2, 3, 3, 4]))
S.append(np.array([1, 0, 1, 0, 0, 3, 3, 3, 4]))
S.append(np.array([1, 0, 1, 2, 3, 2, 2, 4]))
S.append(np.array([0, 1, 3, 1, 1, 2, 2, 3, 3, 4]))
S.append(np.array([0, 1, 0, 0, 0, 2, 3, 2, 2, 3, 4]))
S.append(np.array([0, 1, 3, 4]))
T = []
T.append(np.array([3, 3, 2, 2, 1, 1, 0, 0, 4]))
T.append(np.array([3, 3, 0, 1, 2, 1, 0, 4]))
T.append(np.array([2, 3, 2, 3, 2, 1, 0, 1, 0, 4]))
T.append(np.array([3, 3, 1, 1, 0, 4]))
T.append(np.array([3, 0, 3, 0, 2, 1, 1, 0, 0, 4]))
T.append(np.array([2, 3, 3, 2, 2, 1, 0, 4]))
T.append(np.array([1, 3, 3, 1, 2, 0, 0, 0, 0, 4]))
T.append(np.array([1, 1, 0, 1, 1, 3, 3, 3, 2, 3, 4]))
T.append(np.array([3, 3, 0, 3, 3, 1, 2, 0, 0, 4]))
T.append(np.array([3, 3, 2, 0, 0, 0, 4]))
Q = []
Q.append(np.array([0, 1, 1, 1, 2, 3, 3, 3, 4]))
Q.append(np.array([3, 0, 3, 1, 2, 1, 0, 0, 4]))
Q.append(np.array([2, 3, 2, 1, 0, 1, 0, 4]))
Q.append(np.array([0, 3, 1, 1, 1, 2, 3, 4]))
Q.append(np.array([1, 0, 3, 1, 3, 2, 1, 0, 4]))

a0 = np.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0, 0, 0, 1]])
b0 = np.array([[0.1, 0.2, 0.3, 0.4, 0], [0.1, 0.2, 0.3, 0.4, 0], [0.1, 0.2, 0.3, 0.4, 0], [0, 0, 0, 0, 1]])

pi0 = [1.0, 1.0, 1.0, 0]


def Forward(sample, a, b, pi):
	vector = []
	for i in range(4):
		vector.append(pi[i] * b[i][sample[0]])
	alpha = []
	alpha.append(copy.deepcopy(vector))
	for t in range(1, len(sample)):
		vector = []
		for j in range(4):
			sum = 0.0
			for i in range(4):
				sum += alpha[-1][i] * a[i][j]
			vector.append(sum * b[j][sample[t]])
		# print(vector)
		alpha.append(copy.deepcopy(vector))
	return alpha


def Backward(sample, a, b, pi):
	vector = [0.0, 0, 0, 1]
	beta = []
	beta.append(copy.deepcopy(vector))
	for t in reversed(range(1, len(sample))):
		# print(t)
		vector = []
		for i in range(4):
			sum = 0.0
			for j in range(4):
				sum += beta[-1][j] * a[i][j] * b[j][sample[t]]
			vector.append(sum)
		# print(vector)
		beta.append(copy.deepcopy(vector))
	_beta = []
	for t in reversed(range(len(beta))):
		_beta.append(beta[t])
	return _beta


def Gamma(alpha, beta):
	gamma = []
	for t in range(len(alpha)):
		sum = 0.0
		vector = []
		for j in range(4):
			sum += alpha[t][j] * beta[t][j]
		for i in range(4):
			vector.append(alpha[t][i] * beta[t][i] / sum)
		gamma.append(copy.deepcopy(vector))
	return gamma


def Epsilon(sample, alpha, beta, a, b, pi):
	epsilon = [[0.0, 0, 0, 0], [0.0, 0, 0, 0], [0.0, 0, 0, 0], [0.0, 0, 0, 0]]

	for t in range(len(sample) - 1):
		temp = [[0.0, 0, 0, 0], [0.0, 0, 0, 0], [0, 0.0, 0, 0], [0.0, 0, 0, 0]]
		normalization = 0.0
		for i in range(4):
			for j in range(4):
				tem = alpha[t][i] * a[i][j] * b[j][sample[t + 1]] * beta[t + 1][j]
				temp[i][j] = tem
				normalization += tem
		for i in range(4):
			for j in range(4):
				temp[i][j] /= normalization
				epsilon[i][j] += temp[i][j]
			# print('temp:\n', temp)
			# print('epsilon:\n', epsilon)
	return epsilon


def HMMlearning(samples, a, b, pi):
	theta = 0.00001
	z = 0
	while True:
		z += 1
		a1 = np.array([[0.0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
		a2 = np.array([0.0, 0, 0, 0])
		b1 = np.array([[0.0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
		b2 = np.array([0.0, 0, 0, 0])
		pi1 = np.array([0.0, 0, 0, 0])
		pi2 = len(samples)
		for l in samples:
			alpha = Forward(l, a, b, pi)
			beta = Backward(l, a, b, pi)
			gamma = Gamma(alpha, beta)
			epsilon = Epsilon(l, alpha, beta, a, b, pi)
			# print(l)
			# print(alpha)
			# print(beta)
			# print(gamma)
			# print(epsilon)
			for i in range(4):
				for t in gamma[:-1]:
					a2[i] += t[i]
				for j in range(4):
					a1[i][j] += epsilon[i][j]

			for t in range(len(l)):
				for j in range(4):
					b1[j][l[t]] += gamma[t][j]
					b2[j] += gamma[t][j]

			for i in range(4):
				pi1[i] += gamma[0][i]
		# print(a1, a2, b1, b2)
		for i in range(4):
			for j in range(5):
				b1[i][j] = b1[i][j] / b2[i]
			if i == 3:
				break
			for j in range(4):
				a1[i][j] = a1[i][j] / a2[i]
		pi1 = pi1 / pi2
		d = 0.0
		for i in range(4):
			for j in range(5):
				_d = b1[i][j] - b[i][j]
				b[i][j] = b1[i][j]
				if _d > d:
					d = _d
			_d = pi1[i] - pi[i]
			pi[i] = pi1[i]
			if _d > d:
				d = _d
			if i == 3:
				break
			for j in range(4):
				_d = a1[i][j] - a[i][j]
				a[i][j] = a1[i][j]
				if _d > d:
					d = _d

		if d < theta:
			break
	print('d:', d, 'z:', z)
	print('a:\n', a)
	print('b:\n', b)
	print('pi:\n', pi)
	return a1, b1, pi1


if __name__ == '__main__':
	a1, b1, pi1 = HMMlearning(S, copy.deepcopy(a0), copy.deepcopy(b0), copy.deepcopy(pi0))
	a2, b2, pi2 = HMMlearning(T, copy.deepcopy(a0), copy.deepcopy(b0), copy.deepcopy(pi0))
	for q in Q[:-1]:
		p1 = Forward(q, a1, b1, pi1)[-1]
		p2 = Forward(q, a2, b2, pi2)[-1]
		print(p1[-1], p2[-1])
		qtype = 0
		if p1[-1] > p2[-1]:
			qtype = 1
		else:
			qtype = 2
		print('type:', qtype)

	p1 = Forward(Q[-1], a1, b1, pi1)[-1]
	p2 = Forward(Q[-1], a2, b2, pi2)[-1]
	print(p1[-1], p2[-1])
	qtype = 0
	if p1[-1] > p2[-1]:
		qtype = 1
	else:
		qtype = 2
	print('Etype:', qtype)
	p = p2[-1] / (p2[-1] + p1[-1])
	print('p(lamda1):', p, 'p(lamda2):', 1 - p,)

# use backward to compute probability
# for q in Q[:-1]:
# 	p1 = Backward(q, a1, b1, pi1)[0]
# 	p2 = Backward(q, a2, b2, pi2)[0]
# 	eva1 = 0
# 	eva2 = 0
# 	for i in range(4):
# 		eva1 += pi1[i] * b1[i][q[0]] * p1[i]
# 		eva2 += pi2[i] * b2[i][q[0]] * p2[i]
# 	print(p1, p2, eva1, eva2)
# 	qtype = 0
# 	if eva1 > eva2:
# 		qtype = 1
# 	else:
# 		qtype = 2
# 	print('type:', qtype)
