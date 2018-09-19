# -*- coding:UTF-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import re
import os

time_step = 60  # 时间步
# test_step=50
test_step=20

def loadin():
	# 初步预处理，将原始数据分别存于testdata和traindata文件夹下，预处理训练集时将testdata换为traindata
	flist = os.listdir("D:\documents\大二下\模式\homework\pp+/testdata")
	namelist = []
	for i in range(0, len(flist)):
		path = os.path.join("D:\documents\大二下\模式\homework\pp+/testdata", flist[i])
		namelist.append(path)
	print(namelist)
	k = 0.3

	data = [[], [], [], []]
	pattern_lastp = re.compile(u'lastPrice=(\d+)')
	pattern_bp = re.compile(u'bidPrice1=(\d+)')
	pattern_ap = re.compile(u'askPrice1=(\d+)')
	pattern_vol = re.compile(u'volume=(\d+)')
	pattern_tu = re.compile(u'turnover=(\d+)')
	pattern_bv = re.compile(u'bidVolume1=(\d+)')
	pattern_av = re.compile(u'askVolume1=(\d+)')
	pattern_type = re.compile(u'instrumentID=(.{2})')
	pattern_error = re.compile(u'CST')
	for filename in namelist:
		lasttime = [0, 0, 0, 0]
		_vol, _tu = [0, 0, 0, 0], [0, 0, 0, 0]
		with open(filename, 'r') as f:
			lines = f.readlines()
			for line in lines:
				if re.search(pattern_error, line):
					continue
				type = re.search(pattern_type, line)[0][-2:]
				if type == 'A1':
					i = 0
				elif type == 'A3':
					i = 1
				elif type == 'B2':
					i = 2
				elif type == 'B3':
					i = 3
				else:
					continue
				currenttime = float(line[17:23])
				alltime=line[0:23]
				# print(type, currenttime, lasttime[i])
				if abs(currenttime - lasttime[i]) < 0.01:
					continue
				lasttime[i] = currenttime
				vol = float(re.search(pattern_vol, line)[0][7:])
				tu = float(re.search(pattern_tu, line)[0][9:])
				bv = float(re.search(pattern_bv, line)[0][11:])
				av = float(re.search(pattern_av, line)[0][11:])
				lastp = float(re.search(pattern_lastp, line)[0][10:])
				if _vol[i] == 0 and _tu[i] == 0:
					p=lastp
					data[i].append([lastp, av / (av + bv + 1)])
				else:
					ap = float(re.search(pattern_ap, line)[0][10:])
					bp = float(re.search(pattern_bp, line)[0][10:])
					if vol - _vol[i] == 0:
						divide=lastp
					else:
						divide=(tu - _tu[i]) / (vol - _vol[i])
					if ap==0:
						price=bp
					elif bp==0:
						price=ap
					else:
						price=(bp + ap) / 2
					p = k * divide + (1 - k) * price
					data[i].append([p, av / (av + bv + 1)])
				_vol[i] = vol
				_tu[i] = tu
				if abs(lastp - p) > 50000:
					print(alltime)
					print(lastp,p)
					continue
			print(filename)

	a1 = np.array(data[0])
	a3 = np.array(data[1])
	b2 = np.array(data[2])
	b3 = np.array(data[3])

	plt.figure()
	fi=list(a1.T[0]/100)
	plt.plot(fi)
	plt.show()

	a1.dump('test1/test_a1')
	a3.dump('test1/test_a3')
	b2.dump('test1/test_b2')
	b3.dump('test1/test_b3')

def set():
	#更改文件进行不同类别期货的预处理
	data = np.load('data1/data_b3').T
	print(data)
	test = np.load('test1/test_b3').T
	print(test)
	# 以折线图展示data
	plt.figure()
	fi=list(data[0])
	plt.plot(fi)
	plt.show()

	mean_d=np.mean(data,axis=1).reshape(2,1)
	std_d=np.std(data,axis=1).reshape(2,1)

	normalize_data = (data - mean_d) / std_d  # 标准化
	normalize_test = (test - mean_d) / std_d  # 标准化


	normalize_data = normalize_data.T
	normalize_test = normalize_test.T
	print(normalize_data)
	print(normalize_test)
	print(len(normalize_data), len(normalize_test), len(normalize_test) + len(normalize_data))
	# 生成训练集
	train_x, train_y = [], []  # 训练集
	count=[0,0,0]
	for i in range(len(normalize_data) - time_step-test_step - 1):
		x = normalize_data[i:i + time_step]
		price=normalize_data[i + time_step+ test_step][0]
		# 求后50步的偏差最大值
		# _price=data.T[i+time_step+test_step][0]
		# price=0
		# _price=0
		# for p in range(i+time_step+10,i+time_step+test_step):
		# 	price+=normalize_data[p][0]
		# 	_price+=data.T[p][0]
		# price=price/test_step
		# _price/=test_step
		# print(_price,data.T[i+time_step][0])
		# if _price/data.T[i + time_step][0]>=1.0015:
		# 	y=[2]
		# 	count[2]+=1
		# elif _price/data.T[i + time_step][0]<=0.9985:
		# 	y=[0]
		# 	count[0]+=1
		# else:
		# 	y=[1]
		# 	count[1]+=1
		train_x.append(x.tolist())
		train_y.append(price)
	# print(count) 求各类所占比例
	test_x, test_y = [], []
	for i in range(len(normalize_test) - time_step-test_step  - 1):
		x = normalize_test[i:i + time_step]
		price=normalize_test[i + time_step+ test_step][0]
		# 求后50步的偏差最大值
		# price=normalize_test[i + time_step][0]
		# d=0
		# for p in range(i+time_step+10,i+time_step+test_step):
		# 	_d=abs(normalize_test[p][0]-price)
		# 	if _d>d:
		# 		price=normalize_test[p][0]
		# 		d=_d

		# if price/normalize_test[i + time_step][0]>=1.0015:
		# 	y=[2]
		# elif price/normalize_test[i + time_step][0]<=0.9985:
		# 	y=[0]
		# else:
		# 	y=[1]
		test_x.append(x.tolist())
		test_y.append(price)

	# plt.figure()
	# fig=list(normalize_data.T[0][60:140])
	# plt.subplot(211)
	# plt.plot(fig)
	# plt.subplot(212)
	# plt.plot(train_y[:80])
	# plt.show()
	print(train_x[:5])
	print(train_y[:10])

	np.array(train_x).dump('b3_train_x12')
	np.array(train_y).dump('b3_train_y12')
	np.array(test_x).dump('b3_test_x12')
	np.array(test_y).dump('b3_test_y12')

if __name__ == '__main__':
	loadin()

	set()

	#7不会一划 求下一秒 无法利用
	#8不归一化 求后一段时间偏差最大 绝对值差的太多
	#9归一化train test也归一化train 看60步 取后50步的偏差最大值
	#10归一化train test也归一化train 看60步 取(后50步的偏差最大值-当前）/当前 问题：变化过小
	#11归一化train test也归一化train 看60步 取[后50步的偏差最大值,当前]
	#12归一化train test也归一化train 看60步 取后20步的值