from hmmlearn.hmm import GaussianHMM
import numpy as np
from matplotlib import cm, pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import datetime
import warnings
from sklearn.externals import joblib

n = 3  # 3个隐藏状态
tlist=['a1','a3','b2','b3']
test=[]
p=np.array([0,30,70,30])
for tname in tlist[:1]:
	for ftype in range(1,4):
		dataname=tname+'/'+'type'+str(ftype)
		modelname=tname+'/'+'model'+str(ftype)+'.pkl'
		print(dataname,modelname)
		loaddata = np.load(dataname)
		loaddata=np.delete(loaddata,[1,3],axis=1)
		data=loaddata[:int(len(loaddata)*0.99)]
		for i in range(p[ftype]):
			test.append(loaddata[int(len(loaddata) * 0.99)+i*20:int(len(loaddata) * 0.99)+i*20+20])
	# 	# plt.figure()
	# 	# data_T=data.T
	# 	# plt.subplot(411)
	# 	# plt.plot(data_T[0])
	# 	# plt.subplot(412)
	# 	# plt.plot(data_T[1])
	# 	# plt.subplot(413)
	# 	# plt.plot(data_T[2])
	# 	# plt.subplot(414)
	# 	# plt.plot(data_T[3])
	# 	# plt.show()
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			leng=[]
			for i in range(int(len(data)/20)):
				leng.append(20)
			print(len(leng))
			model = GaussianHMM(n_components=n, covariance_type="diag").fit(data,lengths=leng)
			joblib.dump(model, modelname)
			# model=joblib.load("a1/model3.pkl")
			f=model.n_features
			a=model.transmat_
			pi=model.startprob_
			mean=model.means_
			cov=model.covars_
			print(f)
			print(a)
			print(pi)
			print(mean)
			print(cov)


	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		model0 = joblib.load("a1/model0.pkl")
		model1= joblib.load("a1/model1.pkl")
		model2= joblib.load("a1/model2.pkl")
		model3= joblib.load("a1/model3.pkl")
		model4= joblib.load("a1/model4.pkl")
		n=0
		d=0
		acc=[0,0,0]
		recall=[0,0,0]
		for j in test:
			if n<30:
				ty=0
			elif n<100:
				ty=1
			else:
				ty=2
			sc1=model1.score(j)
			sc2=model2.score(j)
			sc3=model3.score(j)
			re=np.argmax([sc1,sc2,sc3])
			recall[re]+=1
			d+=np.abs(re-ty)
			print(np.abs(re-ty))
			print(re,ty)
			if ty==re:
				acc[ty]+=1
			n+=1
		print(np.array(acc)/p[1:])
		print(np.array(acc)/np.array(recall))
		print(d)
		print(d/132)
