#コヒーレンスヒートマップを作成するための関数の修正版
import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy.stats import t
from scipy.stats import norm

def collect_coherence_test(data, muscle_names, Fs=200):
#各被験者のコヒーレンスデータを計算および格納する関数
#引数はダウンサンプリングしたデータ、筋肉の名前が入ったリスト、サンプリング周波数
#返り値は各被験者の筋ペアごとのコヒーレンスと対応する周波数
    coherence_data = {muscle_pair: [] for muscle_pair in itertools.combinations(muscle_names[:7], 2)}

    for subject_data in data:
        for muscle_pair in itertools.combinations(muscle_names[:7], 2):
            Cxy, freqs = plt.cohere(subject_data[muscle_pair[0]], subject_data[muscle_pair[1]], Fs = Fs)
            plt.close()
            coherence_data[muscle_pair].append((Cxy, freqs))

    return coherence_data
  
  
  
def calculate_mean_and_freq(coherence_data):
#collect_coherence_test()の返り値を受け取り、被験者間の平均値を算出
#返り値は平均コヒーレンスと対応する周波数
    result = {}

    for muscle_pair, data in coherence_data.items():
        Cxy_list = []
        freq_list = []

        for subject_data in data:
            Cxy = subject_data[0]
            freqs = subject_data[1]
            Cxy_list.append(Cxy)
            freq_list.append(freqs)

        # 全被験者のデータを集約
        Cxy_array = np.array(Cxy_list)
        freq_array = np.array(freq_list)

        # 各周波数のコヒーレンス値の平均を計算
        mean_Cxy = np.mean(Cxy_array, axis=0)
        
        # 周波数はすべての被験者で共通なので、最初の被験者の周波数を使用
        common_freqs = freq_array[0]

        result[muscle_pair] = (mean_Cxy, common_freqs)

    return result
  
  
def test_coh_coh_heatmap(coherence_data, min_freq=8, max_freq=50):
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
    
    
def calculate_band_means(data, freq_bands):
#帯域のコヒーレンス平均値を算出する関数
    band_means = {band: [] for band in freq_bands}
    freqs = None
    
    for muscle_pair, (Cxy, freqs) in data.items():
        for band, (min_freq, max_freq) in freq_bands.items():
            # 指定された帯域のインデックスを取得
            band_indices = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
            # 帯域のコヒーレンスの平均値を計算
            band_mean = np.mean(Cxy[band_indices])
            band_means[band].append((muscle_pair, band_mean))
    
    return band_means

def plot_coherence_comparison(young_data, old_data):
#平均コヒーレンスをプロット
#引数はcalculate_mean_and_freq()の返り値
    freq_bands = {'alpha': (8, 12), 'beta': (13, 30), 'gamma': (31, 50)}
    
    # 若年群と高齢群の帯域ごとのコヒーレンスの平均値を計算
    young_band_means = calculate_band_means(young_data, freq_bands)
    old_band_means = calculate_band_means(old_data, freq_bands)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (band, (min_freq, max_freq)) in enumerate(freq_bands.items()):
        ax = axes[i]
        
        # 若年群と高齢群のデータフレームを作成
        young_df = pd.DataFrame(young_band_means[band], columns=['muscle_pair', 'coherence'])
        old_df = pd.DataFrame(old_band_means[band], columns=['muscle_pair', 'coherence'])
        
        # プロット用のデータを準備
        all_data = pd.concat([young_df.assign(group='Young'), old_df.assign(group='Old')])
        
        # 薄い色の箱ひげ図をプロット
        sns.boxplot(x='group', y='coherence', data=all_data, ax=ax, palette=[(0.5, 0.5, 1, 0.3), (1, 0.5, 0.5, 0.3)])
        
        # 各筋ペアのコヒーレンスの値をドットでプロットし、若年群と高齢群の値を線で結ぶ
        for muscle_pair in young_df['muscle_pair']:
            young_value = young_df[young_df['muscle_pair'] == muscle_pair]['coherence'].values[0]
            old_value = old_df[old_df['muscle_pair'] == muscle_pair]['coherence'].values[0]
            #ax.plot(['Young', 'Old'], [young_value, old_value], 'k-', lw=0.5)
            ax.plot('Young', young_value, 'bo')
            ax.plot('Old', old_value, 'ro')
        
        correlations_A = young_df['coherence'] #young_df['coherence']には、若年群の各筋ペアのコヒーレンスの値が入っている
        correlations_B = old_df['coherence'] #old_df['coherence']には、若年群の各筋ペアのコヒーレンスの値が入っている
        
        # Fisherのz変換
        z_A = 0.5 * np.log((1 + np.array(correlations_A)) / (1 - np.array(correlations_A)))
        z_B = 0.5 * np.log((1 + np.array(correlations_B)) / (1 - np.array(correlations_B)))
        
        # 平均z値
        mean_z_A = np.mean(z_A)
        mean_z_B = np.mean(z_B)
        
        # サンプルサイズ
        n_A = len(correlations_A)
        n_B = len(correlations_B)
        
        # 標準誤差
        SE = np.sqrt(1/(n_A - 3) + 1/(n_B - 3))
        
        # zスコア
        z = (mean_z_A - mean_z_B) / SE
        
        # p値
        p_value = 2 * (1 - norm.cdf(np.abs(z)))
        ax.text(0.5, 0.9, f'p= {p_value:.4f}', ha='center', va='center', transform=ax.transAxes, fontsize=12, color = "red")       
        ax.set_title(f'{band} ({min_freq}-{max_freq} Hz)')
    ax.set_ylabel('Coherence')
    ax.set_xlabel('')
    
    plt.tight_layout()
    plt.show()
