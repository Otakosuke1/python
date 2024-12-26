import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
#嶋崎さんスクリプトを一部改変
from tools.test_ms import ms

def calc_cNMF(epochs_data, muscles, confidence_level=0.95):
    """各被験者の平均エポッキングデータを取得し、連結、cNMFの計算、信頼区間の計算を行う"""
    # 各被験者の平均エポッキングデータを取得し、連結
    #epoch_means = [np.mean(epoch, axis=0) for epoch in epochs_data]
    ###############add 2024/7/18#####################
    epoch_means = epochs_data #正規化したデータを使用する
    ##############add 2024/7/18#####################
    combined_data = np.concatenate(epoch_means, axis=0)
    
    # 連結したデータのシナジーを計算
    m_cNMF = ms(max_n_components=5, max_iter=1000, rep=20, vaf=0.9, vaf_mus=0.9)
    m_cNMF.fit(combined_data, muscles, scale=True)
    m_cNMF.est_best_n()
    
    # 重み係数に基づいて各被験者の活性係数を計算
    W_list = []
    for epoch_mean in epoch_means:
        W = m_cNMF.best_syn['W']
        if W.shape[1] == 7:
            result_matrix = np.dot(epoch_mean[:,:7], W.T)  # 100x7行列と7x5行列の内積は100x5行列
        else:
            result_matrix = np.dot(epoch_mean, W.T)
        W_list.append(result_matrix)
    
    # 信頼区間の計算
    alpha = 1 - confidence_level
    n = len(W_list)
    mean_W = np.mean(W_list, axis=0)
    std_W = np.std(W_list, axis=0, ddof=1)
    t_value = stats.t.ppf(1 - alpha / 2, n - 1)
    confidence_interval = t_value * std_W / np.sqrt(n)
    
    return combined_data, m_cNMF, mean_W, confidence_interval

def plot_syns(m_cNMF, mean_W, confidence_interval):
    """cNMFの結果と重み係数のプロットを行う"""
    # 5×2のサブプロットを作成
    A = len(m_cNMF.best_syn['W'])
    fig, axs = plt.subplots(A, 2, figsize=(10, A*3))

    # m_cNMF.plot_W()のプロットを1列目に挿入
    for i in range(A):
        ax = axs[i, 0]
        m_cNMF.plot_W(i+1, ax)

    # result_matrixのプロットを2列目に挿入
    for i in range(A):
        ax = axs[i, 1]
        ax.plot(mean_W[:, i], color=f'C{i}')
        #ax.set_title(f'Plot for column {i+1}')
        #ax.set_xlabel('Sample Index')
        #ax.set_ylabel('Value')
        # シェーディングを追加
        ax.fill_between(np.arange(len(mean_W)), mean_W[:, i] - confidence_interval[:, i],
                        mean_W[:, i] + confidence_interval[:, i], color=f'C{i}', alpha=0.3, label='95% CI')

        #ax.set_title(f'Plot for column {i+1}')
        #ax.set_xlabel('Sample Index')
        #ax.set_ylabel('Value')
        ax.legend()

    # サブプロット間の間隔を調整
    plt.tight_layout()

    # 図を表示
    plt.show()

