import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

def plot_loss(hist, imfname=''):
    plt.subplots(1, 4, figsize=(16, 2))
    for i, key in enumerate(hist.history.keys()):
        n = int(str('14') + str(i + 1))
        plt.subplot(n)
        plt.plot(hist.history[key], 'k', label=f'Training {key}')
        plt.title(f'{imfname} Training {key}')
        plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.legend()
    plt.tight_layout()
    plt.show()


def plot_fit(y_test, y_pred):
    plt.figure(figsize=(4, 2))
    plt.plot(y_test, color="red", label="actual")
    plt.plot(y_pred, color="blue", label="predict")
    plt.title(f"pred & true")
    plt.xlabel("Time")
    plt.ylabel('power')
    plt.legend()
    plt.show()