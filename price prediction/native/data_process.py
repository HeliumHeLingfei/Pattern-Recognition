# -*- coding:UTF-8 -*-

from queue import Queue
import numpy as np
import re
import warnings
import os
from matplotlib import cm, pyplot as plt

# flist = os.listdir("D:\documents\大二下\模式\homework\pp/futuresData")
# namelist = []
# for i in range(0, len(flist)):
# 	path = os.path.join("D:\documents\大二下\模式\homework\pp/futuresData", flist[i])
# 	namelist.append(path)
#
timestep = 20
sumstep=4
# lpa1 = []
# lpa3 = []
# lpb2 = []
# lpb3 = []
# pattern_lastp = re.compile(u'lastPrice=(\d+)')
# pattern_hp = re.compile(u'highestPrice=(\d+)')
# pattern_lp = re.compile(u'lowestPrice=(\d+)')
# pattern_vol = re.compile(u'volume=(\d+)')
# pattern_bv = re.compile(u'bidVolume1=(\d+)')
# pattern_av = re.compile(u'askVolume1=(\d+)')
# pattern_type = re.compile(u'instrumentID=(.{2})')
# pattern_error = re.compile(u'CST')
# for filename in namelist:
# 	with open(filename, 'r') as f:
# 		lines = f.readlines()
# 		# lasttime = [Queue(), Queue(), Queue(), Queue()]
# 		# n = [0, 0, 0, 0]
# 		# hp, lp, a, b = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
# 		for line in lines:
# 			if re.search(pattern_error, line):
# 				# print(line)
# 				continue
# 			type = re.search(pattern_type, line)[0][-2:]
# 			if type == 'A1':
# 				i = 0
# 			elif type == 'A3':
# 				i = 1
# 			elif type == 'B2':
# 				i = 2
# 			elif type == 'B3':
# 				i = 3
# 			else:
# 				continue
# 			lastp = float(re.search(pattern_lastp, line)[0][10:])
# 			vol = float(re.search(pattern_vol, line)[0][7:])
# 			bv = float(re.search(pattern_bv, line)[0][11:])
# 			av = float(re.search(pattern_av, line)[0][11:])
# 			hp = float(re.search(pattern_hp, line)[0][13:])
# 			lp = float(re.search(pattern_lp, line)[0][12:])
# 			# b[i] += _b
# 			# a[i] += _a
# 			# hp[i] += _hp
# 			# lp[i] += _lp
# 			# lasttime[i].put([lastp, _b, _a, _hp, _lp])
# 			# print(b, a, hp, lp)
# 			# if n[i] < timestep:
# 			# 	n[i] += 1
# 			# 	continue
# 			# last = lasttime[i].get()
# 			# p = lastp / last[0]
# 			# hl_r = hp[i] / lp[i]
# 			# b_r = b[i] / (a[i] + b[i])
# 			# bpa_r = vol / (a + b + 1)
# 			tem = [lastp, vol, bv, av, hp, lp]
# 			# print(tem)
# 			# if bpa_r > 1000000:
# 			# 	print(a, b, vol)
# 			# 	print('here')
# 			# 	continue
# 			# if tem[0] < 2:
# 			# 	print(line)
# 			# 	print(last[0])
# 			# b[i] -= last[1]
# 			# a[i] -= last[2]
# 			# hp[i] -= last[3]
# 			# lp[i] -= last[4]
#
# 			if type == 'A1':
# 				lpa1.append(tem)
# 			elif type == 'A3':
# 				lpa3.append(tem)
# 			elif type == 'B2':
# 				lpb2.append(tem)
# 			elif type == 'B3':
# 				lpb3.append(tem)
# 		print(filename)
#
# a1 = np.array(lpa1[:int(len(lpa1) * 9 / 10)])
# a3 = np.array(lpa3[:int(len(lpa3) * 9 / 10)])
# b2 = np.array(lpb2[:int(len(lpb2) * 9 / 10)])
# b3 = np.array(lpb3[:int(len(lpb3) * 9 / 10)])
# t_a1 = np.array(lpa1[int(len(lpa1) * 9 / 10):])
# t_a3 = np.array(lpa3[int(len(lpa3) * 9 / 10):])
# t_b2 = np.array(lpb2[int(len(lpb2) * 9 / 10):])
# t_b3 = np.array(lpb3[int(len(lpb3) * 9 / 10):])
#
# a1.dump('a1/data_a1')
# a3.dump('a3/data_a3')
# b2.dump('b2/data_b2')
# b3.dump('b3/data_b3')
# t_a1.dump('a1/test_a1')
# t_a3.dump('a3/test_a3')
# t_b2.dump('b2/test_b2')
# t_b3.dump('b3/test_b3')

theta = 0.0004

dataa1 = np.load('a1/data_a1')
pa1=[]
a1 = [[], [], [], [], []]
sum = [0, 0, 0, 0, 0]
n = -1
price, hp, lp, a, b ,volume,testsum= 0,0,0,0,0,0,0
teststep=10
mean_p=0
mean_bpa=0
for i in range(len(dataa1)-teststep):
	n+=1
	# tem = [lastp, vol, bv, av, hp, lp]
	lastp,vol,_b,_a,_hp,_lp=dataa1[i]
	b += _b
	a += _a
	hp += _hp
	lp += _lp
	price+=lastp
	volume+=vol
	# print(b[0], a[0], hp[0], lp[0],price[0],volume[0])
	testsum+=dataa1[i+teststep][0]
	if n < sumstep:
		continue
	last = dataa1[i-sumstep]
	price -= last[0]
	volume -= last[1]
	b-= last[2]
	a -= last[3]
	hp-=last[4]
	lp-=last[5]
	# print(price, hp, lp, a, b ,volume)
	p = sumstep*lastp / price
	hl_r = hp / lp
	b_r = b / (a + b)
	bpa_r = sumstep*vol / volume
	pa1.append(np.exp([p,hl_r,b_r,bpa_r]))
	if n<teststep:
		continue
	else:
		testsum-=lastp
	t = teststep*lastp/testsum
	if n<timestep:
		continue
	if t < 1 - 3 * theta:
		a1[0].extend(pa1[-timestep:])
		sum[0] += 1
	elif t < 1 - theta:
		a1[1].extend(pa1[-timestep:])
		sum[1] += 1
	elif t < 1 + theta:
		a1[2].extend(pa1[-timestep:])
		sum[2] += 1
	elif t < 1 + 3 * theta:
		a1[3].extend(pa1[-timestep:])
		sum[3] += 1
	else:
		a1[4].extend(pa1[-timestep:])
		sum[4] += 1
	if i % 5000 == 0:
		print(np.array(sum) / np.sum(sum))

pa1=np.array(pa1).T
plt.figure()
plt.subplot(411)
plt.plot(pa1[0])
plt.subplot(412)
plt.plot(pa1[1])
plt.subplot(413)
plt.plot(pa1[2])
plt.subplot(414)
plt.plot(pa1[3])
plt.show()
a1 = np.array(a1)
print(len(a1[0]))
np.array(a1[0]).dump('a1/type0')
np.array(a1[1]).dump('a1/type1')
np.array(a1[2]).dump('a1/type2')
np.array(a1[3]).dump('a1/type3')
np.array(a1[4]).dump('a1/type4')