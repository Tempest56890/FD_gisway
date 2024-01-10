import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
import tensorflow as tf

#data0=pd.read_csv('sdwpf_baidukddcup2022_full.csv')
df=pd.read_csv('test.csv')
df=df[['Wspd','Wdir','Etmp','Itmp','Ndir','Pab1','Pab2','Pab3','Prtv']]
df=np.nan_to_num(df)
#print(data0)

def build_sequences(data, window_size=24, step=1):
    # text:list of capacity
    x= []
    for i in range(len(data) - window_size-step+1):
        sequence = data[i:i + window_size]
        x.append(sequence)
    return np.array(x)

# 载入模型
mymodel = tf.saved_model.load("step_1")
x_mean=np.loadtxt('x_mean.txt')
x_var=np.loadtxt('x_var.txt')
y_mean=np.loadtxt('y_mean.txt')
y_var=np.loadtxt('y_var.txt')
with open("window_size.txt", "r") as file:
    window_size = file.read()
window_size=int(window_size)
lll=len(x_mean)

X_=build_sequences(df,window_size=window_size)

for mtr in X_:
    for i in range(window_size):
        for j in range(lll):
            mtr[i,j]=(mtr[i,j]-x_mean[j])/math.sqrt(x_var[j])
#X_=[X_[-1]]
X_=tf.cast(X_,tf.float32)
prd=mymodel(X_)
prd=np.array(prd)
for row in prd:
    row[0]=row[0]*math.sqrt(y_var)+y_mean
print(prd)

