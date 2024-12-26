import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np    
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def plot_emg(epochs_data, muscles, confidence_level=0.95):
    """
    エポックデータの平均値と信頼区間を計算し、プロットする。

    :param epochs_data: エポックデータのリスト（各要素が被験者ごとのエポックデータ）
    :param confidence_level: 信頼区間の信頼水準（デフォルトは95%）
    """
    # すべてのエポックデータを1次元目に沿って連結
    reshaped_epochs_data = []

    for data in epochs_data:
        # 3次元目の最初の7つの要素を取り出す
        reshaped_data = data[:, :, :7]
        reshaped_epochs_data.append(reshaped_data)

    epochs_concat = np.concatenate(reshaped_epochs_data[:4], axis=0)

    # 平均と標準誤差、信頼区間の計算
    alpha = 1 - confidence_level
    n = epochs_concat.shape[0]
    mean_across_epochs = np.mean(epochs_concat, axis=0)
    std_error = np.std(epochs_concat, axis=0, ddof=1)
    t_value = stats.t.ppf(1 - alpha / 2, n - 1)
    confidence_interval = t_value * std_error / np.sqrt(n)

    # プロットの設定
    time_points = np.arange(mean_across_epochs.shape[0])  # タイムポイントの数
    channels = muscles

    plt.figure(figsize=(12, 6))

    # 各チャンネルを異なる色でプロット
    for i in range(mean_across_epochs.shape[1]):
        mean = mean_across_epochs[:, i]
        ci = confidence_interval[:, i]

        plt.plot(time_points, mean, label=channels[i])  # 平均値のプロット
        plt.fill_between(time_points, mean - ci, mean + ci, alpha=0.3)  # 信頼区間のシェーディング

    # プロットの装飾
    plt.xlabel('Time Points')
    plt.ylabel('Amplitude')
    #plt.title('Mean and 95% Confidence Interval Across Channels of Young')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

# エポックデータのリスト
#epochs_young = [...]  # ここにエポックデータを入れる

# 関数を呼び出してプロット
#plot_epochs_with_confidence_interval(epochs_young)    

# def plot_emg_test(epochs_data, muscles, selected_channels=None, confidence_level=0.95):
#     """
#     エポックデータの平均値と信頼区間を計算し、プロットする。
#     
#     :param epochs_data: エポックデータのリスト（各要素が被験者ごとのエポックデータ）
#     :param muscles: チャンネルの名前のリスト
#     :param selected_channels: プロットするチャンネルのリスト（デフォルトはすべてのチャンネル）
#     :param confidence_level: 信頼区間の信頼水準（デフォルトは95%）
#     """
#     # すべてのエポックデータを1次元目に沿って連結
#     epochs_concat = np.concatenate(epochs_data[:4], axis=0)
# 
#     # 平均と標準誤差、信頼区間の計算
#     alpha = 1 - confidence_level
#     n = epochs_concat.shape[0]
#     mean_across_epochs = np.mean(epochs_concat, axis=0)
#     std_error = np.std(epochs_concat, axis=0, ddof=1)
#     t_value = stats.t.ppf(1 - alpha / 2, n - 1)
#     confidence_interval = t_value * std_error / np.sqrt(n)
# 
#     # プロットの設定
#     time_points = np.arange(mean_across_epochs.shape[0])  # タイムポイントの数
#     num_channels = mean_across_epochs.shape[1]
# 
#     # 選択されたチャンネルがない場合はすべてのチャンネルをプロット
#     if selected_channels is None:
#         selected_channels = muscles
# 
#     plt.figure(figsize=(12, 6))
# 
#     # 各チャンネルを異なる色でプロット
#     for i in range(num_channels):
#         if muscles[i] in selected_channels:
#             mean = mean_across_epochs[:, i]
#             ci = confidence_interval[:, i]
#             
#             plt.plot(time_points, mean, label=muscles[i])  # 平均値のプロット
#             plt.fill_between(time_points, mean - ci, mean + ci, alpha=0.3)  # 信頼区間のシェーディング
# 
#     # プロットの装飾
#     plt.xlabel('Time Points')
#     plt.ylabel('Amplitude')
#     plt.legend(loc='upper right')
#     plt.grid(True)
#     plt.show()

# エポックデータのリスト
# epochs_young = [...]  # ここにエポックデータを入れる
# muscles = ['rTA', 'rGast', 'rHam', 'rQuad']  # 例: チャンネルの名前リスト

# 関数を呼び出してプロット
# plot_emg_test(epochs_young, muscles, selected_channels=['rTA', 'rHam'])

# def heatmap_emg(norm_data, muscles):
#      # norm_dataをnumpy配列に変換
#      norm_data = np.array(norm_data)
#      #norm_data = pd.DataFrame(norm_data)
#      fig, axs = plt.subplots(14, 1, figsize=(12, 1 * 14), sharex=True)
#      
#      for i in range(14):
#          # 特定のチャネルを抽出
#          channel_index = i
#          channel_data = norm_data[:, :, channel_index]
#          #channel_data = pd.DataFrame(channel_data)
#          # プロット
#          ax = axs[i]
#          cax = ax.imshow(channel_data, aspect='auto', interpolation='none')
#          #print(type(channel_data))
#          ax.set_ylabel('Subjects')
#          num_sets = channel_data.shape[0]
#          ax.set_yticks(np.arange(num_sets))
#          ax.set_yticklabels(np.arange(1, num_sets + 1))
#          ax.set_title(muscles[i])
#  
#      # 共通のカラーバーをプロットの一番下に固定
#      cbar_ax = fig.add_axes([0.125, 0.01, 0.775, 0.02])
#      fig.colorbar(cax, cax=cbar_ax, orientation='horizontal', label='emg')
#      
#      # レイアウト調整
#      plt.tight_layout()
#      plt.show()


# def heatmap_muscles(norm_data, muscles, muscle):
#     norm_data = np.array(norm_data)
#     channel_index = muscles.index(muscle)
#     #channel_index = i
#     channel_data = norm_data[ :, channel_index]
# 
#     # プロット
#     plt.imshow(channel_data, aspect='auto', interpolation='none')
#     #plt.colorbar(orientation='horizontal')
#     #plt.xlabel('Time')
#     plt.ylabel('subjects')
#     num_sets = channel_data.shape[0]
#     plt.yticks(ticks=np.arange(num_sets), labels=np.arange(1, num_sets + 1))
#     plt.title(muscle)
#     plt.tight_layout()
#     plt.show()
    
    
def heatmap_mean_emg(norm_data, muscles):
    norm_data = np.array(norm_data)
    fig, axs = plt.subplots(len(muscles), figsize=(12, 1 * len(muscles)), sharex=True)
    
    for i in range(len(muscles)):
        muscle_data = norm_data[:, i]
        ax = axs[i]
        cax = ax.imshow(muscle_data[:, np.newaxis].T, aspect='auto', interpolation='none')
        ax.set_title(muscles[i], fontsize=16)
        ax.set_yticks([])
    
    plt.xlabel('timepoint', fontsize=16)
    # 共通のカラーバーをプロットの一番下に固定
    cbar_ax = fig.add_axes([0.125, 0.05, 0.775, 0.02])
    fig.colorbar(cax, cax=cbar_ax, orientation='horizontal', label='')
    
    # レイアウト調整
    plt.tight_layout(rect=[0, 0.1, 1, 1], pad=2.0)
    plt.show()

#アップロードデータ用のamp関数。一人分のデータでも動くように修正    
def plot_emg_upload(epochs_data, muscles, confidence_level=0.95):
    """
    エポックデータの平均値と信頼区間を計算し、プロットする。

    :param epochs_data: エポックデータのリスト
    :param confidence_level: 信頼区間の信頼水準（デフォルトは95%）
    """

    epochs_concat = epochs_data

    # 平均と標準誤差、信頼区間の計算
    alpha = 1 - confidence_level
    n = epochs_concat.shape[0]
    mean_across_epochs = np.mean(epochs_concat, axis=0)
    std_error = np.std(epochs_concat, axis=0, ddof=1)
    t_value = stats.t.ppf(1 - alpha / 2, n - 1)
    confidence_interval = t_value * std_error / np.sqrt(n)

    # プロットの設定
    time_points = np.arange(mean_across_epochs.shape[0])  # タイムポイントの数
    channels = muscles

    plt.figure(figsize=(12, 6))

    # 各チャンネルを異なる色でプロット
    for i in range(mean_across_epochs.shape[1]):
        mean = mean_across_epochs[:, i]
        ci = confidence_interval[:, i]

        plt.plot(time_points, mean, label=channels[i])  # 平均値のプロット
        plt.fill_between(time_points, mean - ci, mean + ci, alpha=0.3)  # 信頼区間のシェーディング

    # プロットの装飾
    plt.xlabel('Time Points')
    plt.ylabel('Amplitude')
    #plt.title('Mean and 95% Confidence Interval Across Channels of Young')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

