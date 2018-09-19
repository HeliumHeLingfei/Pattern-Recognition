# -*- coding:UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ——————————————————导入数据——————————————————————
data = np.load('data_a1')
print(data)
test = np.load('test_a1')
print(test)
# 以折线图展示data
plt.figure()
fi=list(data)
fi.extend(list(test))
plt.plot(fi)
plt.show()
normalize_data = (data - np.mean(data)) / np.std(data)  # 标准化
normalize_test = (test - np.mean(test)) / np.std(test)  # 标准化
print(normalize_data)
print(normalize_test)
normalize_data = normalize_data[:, np.newaxis]  # 增加维度
normalize_test = normalize_test[:, np.newaxis]
print(normalize_data)
print(normalize_test)
print(len(normalize_data), len(normalize_test), len(normalize_test) + len(normalize_data))
# 生成训练集
# 设置常量
time_step = 20  # 时间步
rnn_unit = 10  # hidden layer units
batch_size = 60  # 每一批次训练多少个样例
input_size = 1  # 输入层维度
output_size = 1  # 输出层维度
lr = 0.0006  # 学习率
train_x, train_y = [], []  # 训练集
for i in range(len(normalize_data) - time_step - 1):
	x = normalize_data[i:i + time_step]
	y = normalize_data[i + 1:i + time_step + 1]
	train_x.append(x.tolist())
	train_y.append(y.tolist())

test_x, test_y = [], []
for i in range(len(normalize_test) - time_step - 1):
	x = normalize_test[i:i + time_step]
	y = normalize_test[i + 1:i + time_step + 1]
	test_x.append(x.tolist())
	test_y.append(y.tolist())

plt.figure()
fig=list(normalize_data)
fig.extend(list(normalize_test))
plt.plot(fig)
plt.show()
# print(train_x)
# print(train_y)

# ——————————————————定义神经网络变量——————————————————
X = tf.placeholder(tf.float32, [None, time_step, input_size])  # 每批次输入网络的tensor
Y = tf.placeholder(tf.float32, [None, time_step, output_size])  # 每批次tensor对应的标签
# 输入层、输出层权重、偏置
weights = {
	'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
	'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
	'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
	'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


# ——————————————————定义神经网络变量——————————————————
def lstm(batch):  # 参数：输入网络批次数目
	w_in = weights['in']
	b_in = biases['in']
	input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
	input_rnn = tf.matmul(input, w_in) + b_in
	input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
	cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
	init_state = cell.zero_state(batch, dtype=tf.float32)
	output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
	                                             dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
	output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
	w_out = weights['out']
	b_out = biases['out']
	pred = tf.matmul(output, w_out) + b_out
	return pred, final_states


# ——————————————————训练模型——————————————————
def train_lstm():
	global batch_size
	pred, _ = lstm(batch_size)
	# 损失函数
	loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
	train_op = tf.train.AdamOptimizer(lr).minimize(loss)
	saver = tf.train.Saver(tf.global_variables())
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# 重复训练10000次
		for i in range(3):
			step = 0
			start = 0
			end = start + batch_size
			while (end < len(train_x)):
				_, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[start:end], Y: train_y[start:end]})
				start += batch_size
				end = start + batch_size
				# 每10步保存一次参数
				if step % 500 == 0:
					print(i, step, loss_)
					print("保存模型：", saver.save(sess, 'module/stock.model'))
				step += 1


# ————————————————预测模型————————————————————
def prediction():
	pred, _ = lstm(1)  # 预测时只输入[1,time_step,input_size]的测试数据
	saver = tf.train.Saver(tf.global_variables())
	with tf.Session() as sess:
		# 参数恢复
		module_file = tf.train.latest_checkpoint('module/')
		saver.restore(sess, module_file)

		# 取训练集最后一行为测试样本。shape=[1,time_step,input_size]
		predict = []
		# 得到之后100个预测结果
		for i in range(len(test_x)):
			prev_seq = test_x[i]
			next_seq = sess.run(pred, feed_dict={X: [prev_seq]})
			predict.append(next_seq[-1])
		# 每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
		# 以折线图表示结果
		plt.figure()
		plt.plot(list(range(len(normalize_data)+len(normalize_test))), fig, color='b')
		plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict,
		         color='r')
		plt.show()


if __name__ == '__main__':
	print('ok')
	# train_lstm()
	prediction()
