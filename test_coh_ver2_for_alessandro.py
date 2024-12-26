#コヒーレンスヒートマップを作成するための関数の修正版
import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy.stats import t
from scipy.stats import norm

def collect_coherence_test_for_ale(data, muscle_names, Fs=200):
#各被験者のコヒーレンスデータを計算および格納する関数
#引数はダウンサンプリングしたデータ、筋肉の名前が入ったリスト、サンプリング周波数
#返り値は各被験者の筋ペアごとのコヒーレンスと対応する周波数
    coherence_data = {muscle_pair: [] for muscle_pair in itertools.combinations(muscle_names, 2)}
    length = len(data)
    NFFT = min(length//2, 256)
 
    
    for muscle_pair in itertools.combinations(muscle_names, 2):
        Cxy, freqs = plt.cohere(data[muscle_pair[0]], data[muscle_pair[1]], NFFT=NFFT, Fs = Fs)
        plt.close()
        coherence_data[muscle_pair].append((Cxy, freqs))

    return coherence_data
  
  

def test_coh_coh_heatmap_for_ale(coherence_data, min_freq=8, max_freq=50):
#コヒーレンスヒートマップを作成する関数
#引数はcollect_coherence_test()の返り値
    ex_data = calculate_mean_and_freq(coherence_data)
    mean_values = []
    muscle_pairs = []
    common_freqs = None
   
    
    for muscle_pair, (mean_data, freqs) in ex_data.items():
        if common_freqs is None:
            common_freqs = freqs  # 一度だけ周波数リストを取得
        
        mean_values.append(mean_data)
        muscle_pairs.append(muscle_pair)
        

    # リストをnumpy配列に変換します
    mean_values = np.array(mean_values)
    
    # min_freqおよびmax_freqに基づいてインデックスを計算
    min_freq = min_freq
    max_freq = max_freq
    min_index = np.max(np.where(common_freqs <= min_freq))
    max_index = np.min(np.where(common_freqs >= max_freq))

    # 列の任意の要素を取り出す
    subset_mean_values = mean_values[:, min_index:max_index + 1]
    subset_common_freqs = common_freqs[min_index:max_index + 1]

    # データフレームを作成します
    data_df = pd.DataFrame(subset_mean_values, index=muscle_pairs)

    # ヒートマップをプロットします
    g = sns.clustermap(data_df, cmap="viridis", cbar_kws={'label': 'Mean Values'}, 
                   row_cluster=True, col_cluster=False, figsize=(6, 4)) 

    # x軸の設定
    ax = g.ax_heatmap
    ax.set_xticks(np.arange(0.5, len(subset_common_freqs), 2))  # メモリの位置を2単位ごとに設定
    ax.set_xticklabels(np.round(subset_common_freqs[::2], 1), fontsize=14)
    ax.tick_params(axis='x', rotation=90)  # ラベルの回転を設定します
    ax.set_xlabel('Frequency (Hz)', fontsize=14)

    # y軸の設定
    ax.set_yticks(np.arange(0.5, len(muscle_pairs)))  # メモリの位置を1単位ごとに設定
    ax.set_yticklabels(muscle_pairs, fontsize=14)  # y軸のラベルのフォントサイズを指定
    ax.set_ylabel('Muscle Pairs', fontsize=14)

    plt.show()
