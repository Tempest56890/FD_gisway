import plot_
import eval
import model_
from keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# data0=pd.read_csv('sdwpf_baidukddcup2022_full.csv')
data0 = pd.read_csv('wtbdata_245days.csv')
data0 = data0[['Wspd', 'Wdir', 'Etmp', 'Itmp',
               'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'Patv']]
data0 = data0[:3000]
data0 = np.nan_to_num(data0)
# print(data0)


def build_sequences(text, window_size=24, step=1):
    # text:list of capacity
    x, y = [], []
    for i in range(len(text) - window_size - step + 1):
        sequence = text[i:i + window_size]
        target = text[i + window_size + step - 1]
        x.append(sequence)
        y.append(target)
    return np.array(x), np.array(y)


def get_traintest(data, train_size=len(data0), window_size=24, step=1):
    train = data[:train_size]
    test = data[train_size - window_size:]
    X_train, y_train = build_sequences(
        train, window_size=window_size, step=step)
    X_test, y_test = build_sequences(test, window_size=window_size, step=step)
    return X_train[:, :, :-1], y_train[:, -1], X_test[:, :, :-1], y_test[:, -1]


df_eval_all = pd.DataFrame(columns=['MAE', 'RMSE', 'MAPE', 'R2'])
df_preds_all = pd.DataFrame()


def train_fuc(
        mode='LSTM',
        window_size=64,
        batch_size=32,
        epochs=50,
        hidden_dim=[32,16],
    train_ratio=0.8,
    show_loss=True,
    show_fit=True,
        step=1):
    # 准备数据
    data = data0  # .to_numpy()
    # 归一化
    scaler = StandardScaler()
    scaler = scaler.fit(data[:, :-1])
    X = scaler.transform(data[:, :-1])

    y_scaler = StandardScaler()
    y_scaler = y_scaler.fit(data[:, -1].reshape(-1, 1))
    y = y_scaler.transform(data[:, -1].reshape(-1, 1))

    train_size = int(len(data) * train_ratio)
    X_train, y_train, X_test, y_test = get_traintest(
        np.c_[X, y], window_size=window_size, train_size=train_size, step=step)
    # print(X_test)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # 构建模型
    s = time.time()
    model_.set_my_seed()
    model = model_.build_model(
        X_train=X_train,
        mode=mode,
        hidden_dim=hidden_dim)
    earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=5)
    hist = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[earlystop],
        verbose=0)
    if show_loss:
        plot_.plot_loss(hist)

    # 预测
    y_pred = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred)
    y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1))
    # print(f'真实y的形状：{y_test.shape},预测y的形状：{y_pred.shape}')
    if show_fit:
        plot_.plot_fit(y_test, y_pred)
    e = time.time()
    print(f"运行时间为{round(e - s, 3)}")
    df_preds_all[mode] = y_pred.reshape(-1, )

    s = list(eval.evaluation(y_test, y_pred))
    df_eval_all.loc[f'{mode}', :] = s
    s = [round(i, 3) for i in s]
    print(f'{mode}的预测效果为：MAE:{s[0]},RMSE:{s[1]},R2:{s[3]}')
    print("=======================================运行结束==========================================")

    # 导出模型， 模型目录
    tf.saved_model.save(model, "step_1")

    return model, scaler.mean_, scaler.var_, y_scaler.mean_, y_scaler.var_


window_size = 18
batch_size = 32
epochs = 50
hidden_dim = [window_size * 6, window_size * 2]
train_ratio = 0.8
show_fit = True
show_loss = True
mode = 'RNN'  # LSTM,RNN,GRU,CNN,MLP
step = 1

model_.set_my_seed()
model, x_mean, x_var, y_mean, y_var = train_fuc(
    mode=mode, window_size=window_size, batch_size=batch_size, epochs=epochs, hidden_dim=hidden_dim, step=step)
np.savetxt('x_mean.txt', np.array(x_mean), delimiter=',', fmt='%f')
np.savetxt('x_var.txt', np.array(x_var), delimiter=',', fmt='%f')
np.savetxt('y_mean.txt', np.array(y_mean), delimiter=',', fmt='%f')
np.savetxt('y_var.txt', np.array(y_var), delimiter=',', fmt='%f')
with open("window_size.txt", "w") as file:
    file.write(str(window_size))
# print(x_mean,x_var,y_mean,y_var)
