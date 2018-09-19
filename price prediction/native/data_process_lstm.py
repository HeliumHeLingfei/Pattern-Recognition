# -*- coding:UTF-8 -*-


import numpy as np
import re
import os

flist = os.listdir("D:\documents\大二下\模式\homework\pp/futuresData")
namelist = []
for i in range(0, len(flist)):
	path = os.path.join("D:\documents\大二下\模式\homework\pp/futuresData", flist[i])
	namelist.append(path)

lpa1 = []
lpa3 = []
lpb2 = []
lpb3 = []
for filename in namelist:
	with open(filename, 'r') as f:
		lines = f.readlines()
		pattern_lastprice = re.compile(u'lastPrice=(\d+)')
		pattern_type = re.compile(u'instrumentID=(.{2})')
		for line in lines:
			type = re.search(pattern_type, line)[0][-2:]
			lp = float(re.search(pattern_lastprice, line)[0][10:])
			if type == 'A1':
				lpa1.append(lp / 100)
			elif type == 'A3':
				lpa3.append(lp / 100)
			elif type == 'B2':
				lpb2.append(lp / 1000)
			elif type == 'B3':
				lpb3.append(lp / 1000)
		print(filename)

a1 = np.array(lpa1[:int(len(lpa1) * 4 / 5)])
a3 = np.array(lpa3[:int(len(lpa3) * 4 / 5)])
b2 = np.array(lpb2[:int(len(lpb2) * 4 / 5)])
b3 = np.array(lpb3[:int(len(lpb3) * 4 / 5)])
t_a1 = np.array(lpa1[int(len(lpa1) * 4 / 5):])
t_a3 = np.array(lpa3[int(len(lpa3) * 4 / 5):])
t_b2 = np.array(lpb2[int(len(lpb2) * 4 / 5):])
t_b3 = np.array(lpb3[int(len(lpb3) * 4 / 5):])

a1.dump('data_a1')
a3.dump('data_a3')
b2.dump('data_b2')
b3.dump('data_b3')
t_a1.dump('test_a1')
t_a3.dump('test_a3')
t_b2.dump('test_b2')
t_b3.dump('test_b3')
