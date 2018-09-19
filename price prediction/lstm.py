# -*- coding:UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn
import random

# 设置常量
time_step = 60  # 时间步
test_step=50
rnn_unit = 32  # hidden layer units
batch_size = 1000  # 每一批次训练多少个样例
layer_num = 2
input_size = 1  # 输入层维度
output_size = 1  # 输出层维度
# lr = 0.0001  # 学习率
# ——————————————————导入数据——————————————————————

# ——————————————————定义神经网络变量——————————————————
X = tf.placeholder(tf.float32, [None, time_step, input_size])  # 每批次输入网络的tensor
Y = tf.placeholder(tf.float32, [None,])  # 每批次tensor对应的标签
keep_prob = tf.placeholder(tf.float32)

# ——————————————————定义神经网络变量——————————————————
def lstm(batch):  # 参数：输入网络批次数目
	# 输入层、输出层权重、偏置
	weights = {
		'in': tf.Variable(tf.truncated_normal([input_size, rnn_unit])),
		'out': tf.Variable(tf.truncated_normal([rnn_unit, output_size]))
	}
	biases = {
		'in': tf.Variable(tf.constant(0.01, shape=[rnn_unit, ]), dtype=tf.float32),
		'out': tf.Variable(tf.constant(0.01, shape=[output_size, ]), dtype=tf.float32)
	}
	w_in = weights['in']
	b_in = biases['in']
	input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
	input_ = tf.nn.dropout(tf.nn.relu_layer(input, w_in ,b_in),keep_prob)
	input_rnn = tf.reshape(input_, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
	basic_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit,reuse=tf.get_variable_scope().reuse)
	lstm_cell = rnn.DropoutWrapper(cell=basic_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
	cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*layer_num)
	init_state = cell.zero_state(batch, dtype=tf.float32)
	output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
	                                             dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
	output = final_states[-1][1] # 作为输出层的输入
	w_out = weights['out']
	b_out = biases['out']
	pred = tf.matmul(output, w_out) + b_out
	return input_,output,pred, final_states


# ——————————————————训练模型——————————————————
def train_lstm():
	train_x=np.load('D:\documents\大二下\模式\homework\pp+/b3_train_x12')[:,:,0,np.newaxis]
	train_y=np.load('D:\documents\大二下\模式\homework\pp+/b3_train_y12')
	global batch_size
	inp,out, pred, _ = lstm(batch_size)
	# 损失函数
	# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=pred)
	loss = tf.reduce_mean(tf.square(tf.subtract(pred,Y)))

	global_step=tf.Variable(0)
	learning_rate = tf.train.exponential_decay(1e-3, global_step, decay_steps=50, decay_rate=0.96,
	                                           staircase=True)
	train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

	# correct_prediction = tf.cast(tf.equal(tf.argmax(pred, 1), Y), tf.float64)
	accuracy = tf.reduce_mean(tf.abs(tf.subtract(pred,Y)))

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# saver.restore(sess,'./b2_modulef12/stock.model')
		# 重复训练10次
		for i in range(10):
			step = 0
			start = 0
			end = start + batch_size
			while (end < len(train_x)):
				_, loss_ = sess.run([train_op, loss],
				                    feed_dict={X: train_x[start:end], Y: train_y[start:end], keep_prob:0.5})
				start += batch_size
				end = start + batch_size
				# 每100步保存一次参数
				if step % 100 == 0:
					_s=random.randint(1,400000)
					print(_s)
					_e=_s+batch_size
					ac=0
					pred_=[]
					_i=0
					while _i<10:
						lrate,inpu,o,_ac,_pred_ = sess.run([learning_rate,inp,out, accuracy,pred],
						                    feed_dict={X: train_x[_s:_e], Y: train_y[_s:_e], keep_prob: 1.0})
						ac+=_ac
						pred_.extend(_pred_)
						_i+=1
						_s += batch_size
						_e = _s + batch_size
					print(np.array(pred_)[:4])
					print(train_y[_s:_s+4])
					print(i, step, ac/_i,loss_,lrate)
					print("保存模型：", saver.save(sess, 'b3_modulef12/stock.model'))
				step += 1


# ————————————————预测模型————————————————————
def prediction():
	test_x=np.load('D:\documents\大二下\模式\homework\pp+/b3_test_x12')[:,:,0,np.newaxis]
	inp, out, pred, _ = lstm(batch_size)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		# 参数恢复
		saver.restore(sess, './b3_modulef12/stock.model')

		predict = []
		for i in range(int(len(test_x)/batch_size)):
			prev_seq = test_x[i*batch_size:i*batch_size+batch_size]
			next_seq = sess.run(pred, feed_dict={X: prev_seq,keep_prob:1.0})
			# print(next_seq)
			predict.extend(next_seq[:,0])
		# 以折线图表示结果
		plt.figure()
		# test_y=test_y*std_t[0]+mean_t[0]
		# trainp=np.array(predict)*std_t[0]+mean_t[0]
		np.array(predict).dump('b3_predicting12')

def eva():
	data = np.load('data1/data_b3').T
	mean_d = np.mean(data, axis=1).reshape(2, 1)
	std_d = np.std(data, axis=1).reshape(2, 1)
	test_y = np.load('D:\documents\大二下\模式\homework\pp+/b3_test_y12')
	predict=np.load('b3_predicting12')
	predict=np.array(predict)*std_d[0]+mean_d[0]
	test_y=test_y*std_d[0]+mean_d[0]
	plt.plot(list(range(len(predict))), predict, color='b')
	plt.plot(list(range(len(predict))), test_y[:len(predict)][:], color='r')
	# plt.show()
	kind=np.array([0,0,0])
	TP=np.array([0,0,0])
	pre=np.array([0,0,0])
	sum,_sum=[],[]
	_predict=[]
	for i in range(6):
		sum.append(test_y[i])
		_sum.append(predict[i])
		_predict.append(test_y[i])
	for i in range(6,len(predict)):
		sum=sum[1:]
		sum.append(test_y[i])
		_sum=_sum[1:]
		_sum.append(predict[i])
		k=np.mean(sum)/np.mean(_sum)
		_predict.append(k*predict[i])
	plt.plot(list(range(len(_predict))), _predict, color='b')
	plt.plot(list(range(len(_predict))), test_y[:len(_predict)][:], color='r')
	plt.show()
	for i in range(len(_predict)-20):
		if _predict[i+20]/_predict[i]>=1.0015:
			a=2
		elif _predict[i+20]/_predict[i]<=0.9985:
			a=0
		else:
			a=1
		pre[a]+=1
		if test_y[i+20]/test_y[i]>=1.0015:
			b=2
		elif test_y[i+20]/test_y[i]<=0.9985:
			b=0
		else:
			b=1
		kind[b]+=1
		if a==b:
			TP[a]+=1
	print(kind,TP,pre)
	print(TP/pre,TP/kind)
	kind=np.array([0,0,0])
	TP=np.array([0,0,0])
	pre=np.array([0,0,0])
	for i in range(len(_predict)-20):
		a,b=0,0
		if _predict[i+20]/_predict[i]>=1.00055:
			a=2
		elif _predict[i+20]/_predict[i]<=0.99945:
			a=0
		else:
			a=1
		pre[a]+=1
		if test_y[i+20]/test_y[i]>=1.00055:
			b=2
		elif test_y[i+20]/test_y[i]<=0.99945:
			b=0
		else:
			b=1
		kind[b]+=1
		if a==b:
			TP[a]+=1
	print(kind,TP,pre)
	print(TP/pre,TP/kind)

def eva1():

	a1 = np.load('D:\documents\大二下\模式\homework\pp+/train_y12')
	a3 = np.load('D:\documents\大二下\模式\homework\pp+/a3_train_y12')
	b2 = np.load('D:\documents\大二下\模式\homework\pp+/b2_train_y12')
	b3 = np.load('D:\documents\大二下\模式\homework\pp+/b3_train_y12')
	data1 = np.load('data1/data_a1').T
	data2 = np.load('data1/data_a3').T
	data3 = np.load('data1/data_b2').T
	data4 = np.load('data1/data_b3').T
	m=[a1,a3,b2,b3]
	n=[data1,data2,data3,data4]
	for _j in range(4):
		j=m[_j]
		data=n[_j]
		mean_d = np.mean(data, axis=1).reshape(2, 1)
		std_d = np.std(data, axis=1).reshape(2, 1)
		kind=[0,0,0]
		j=j*std_d[0]+mean_d[0]
		for i in range(len(j)-20):
			if j[i+20]/j[i]>=1.00055:
				b=2
			elif j[i+20]/j[i]<=0.99945:
				b=0
			else:
				b=1
			kind[b]+=1
		print(kind,np.array(kind)/np.sum(kind))

if __name__ == '__main__':
	print('ok')
	# train_lstm()
	# prediction()
	# eva()
	eva1()


# 9
# [  6178 175694   6108] [  3785 173286   3557] [  4969 178226   4785]
# [0.76172268 0.97228238 0.74336468] [0.61265782 0.98629435 0.58235102]
# [44689 98147 45144] [30237 86255 30602] [ 36322 114840  36818]1.00055
# [0.83247068 0.75108847 0.83116954] [0.67660946 0.87883481 0.67787524]

# 9_1
# [  6178 175694   6108] [  3671 161749   3552] [ 10583 166755  10642]
# [0.34687707 0.96997991 0.33377185] [0.59420524 0.92062905 0.58153242]
# [44689 98147 45144] [28499 68097 28490] [45335 97260 45385]
# [0.6286313  0.70015423 0.62774044] [0.63771845 0.69382661 0.63109162]

# 12
# [  5003 177690   5287] [  3313 175260   3526] [  4505 178680   4795]
# [0.73540511 0.98085964 0.73534932] [0.66220268 0.9863245  0.66691886]
# [ 42055 102976  42949] [29491 90403 30108] [ 35977 115545  36458]
# [0.81971815 0.78240512 0.82582698] [0.70124837 0.87790359 0.70101749]

# 12_1
# [  5003 177690   5287] [  3121 159051   3213] [ 12153 162972  12855]
# [0.25680902 0.97594065 0.24994166] [0.6238257  0.89510383 0.60771704]
# [ 42055 102976  42949] [25950 56242 26238] [51701 84276 52003]
# [0.50192453 0.66735488 0.50454781] [0.6170491  0.5461661  0.61091061]

# a3
# [  6373 496059   6548] [  4140 494011   4104] [  5151 498619   5210]
# [0.80372743 0.99075847 0.78771593] [0.64961557 0.99587146 0.62675626]
# [ 79315 349219  80446] [ 57242 331569  58645] [ 66184 375034  67762]
# [0.86489182 0.88410384 0.86545557] [0.7217046  0.94945865 0.72899833]

# b2
# [   737 405354    889] [   535 404953    681] [   721 405337    922]
# [0.74202497 0.99905264 0.73861171] [0.72591588 0.99901074 0.76602925]
# [ 15821 375866  15293] [ 10973 368683  10353] [ 14652 378381  13947]
# [0.748908   0.97436975 0.74231017] [0.69357183 0.98088947 0.67697639]

# b3
# [  1636 494729   1615] [  1135 493855   1141] [  1561 494808   1611]
# [0.72709801 0.998074   0.70825574] [0.69376528 0.99823338 0.70650155]
# [ 29241 439228  29511] [ 21829 424193  21820] [ 29331 439186  29463]
# [0.74422965 0.96586184 0.74058989] [0.7465203  0.96576949 0.73938531]