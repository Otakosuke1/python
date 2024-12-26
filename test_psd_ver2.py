#psdのヒーマップを作成するための関数の修正版
import numpy as np
import glob
from scipy.stats import t
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
from tools.EMG import * 
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

def group_psd_data(psd_data, muscles, num_subjects=4):
#psdデータの平均値と標準誤差を計算
#引数はprocessdataのpsdの値
#返り値は平均値と標準偏差
    # 各被験者のPSD値を格納する辞書を初期化
    psd_values_all_subjects = {muscle: [] for muscle in muscles}
    freq_values_all_subjects = {muscle: [] for muscle in muscles}

    # 各被験者のPSD値をリストに格納し、被験者間で同じ筋肉のリストを結合する
    for i in range(num_subjects):
        psd_results = psd_data[i]

        # 各筋肉のPSD値と周波数をリストに格納
        for muscle in muscles:
            psd_values_all_subjects[muscle].append(psd_results[muscle]['psd'][0])
            freq_values_all_subjects[muscle].append(psd_results[muscle]['freq'][0])

    # 列方向の平均を計算
    means = {muscle: {'mean_psd': np.mean(values, axis=0), 'freq': freq_values_all_subjects[muscle][0]}
             for muscle, values in psd_values_all_subjects.items()}

    # 列方向の標準誤差を計算
    std_errs = {muscle: {'std_err_psd': np.std(values, axis=0, ddof=1) / np.sqrt(len(values)), 'freq': freq_values_all_subjects[muscle][0]}
                for muscle, values in psd_values_all_subjects.items()}

    return means, std_errs
  
  
def plot_psd_heatmaps(means, muscles, fmin=None, fmax=None):
#psdヒートマップを作成
#引数は平均値と筋肉のリスト
    num_subjects = len(means[muscles[0]]['mean_psd'])  # 各筋肉の被験者数を取得

    fig, axs = plt.subplots(len(muscles), 1, figsize=(12, 1 * len(muscles)), sharex=True)
    
    if len(muscles) == 1:
        axs = [axs]  # 1つの筋肉の場合、axsが1つのAxesオブジェクトになるのでリストに変換

    for i, (ax, muscle) in enumerate(zip(axs, muscles)):
        # 各筋肉のPSDデータの平均値を取得
        mean_psd = means[muscle]['mean_psd']
        freq = means[muscle]['freq']
        
        # 対数スケールに変換
        log_psd = np.log10(mean_psd + 1e-10)  # 0の値を避けるために小さな値を追加

        # fminとfmaxが指定されている場合、その範囲内のデータのみを使用
        if fmin is not None:
            fmin_idx = np.where(freq >= fmin)[0][0]
        else:
            fmin_idx = 0
        
        if fmax is not None:
            fmax_idx = np.where(freq <= fmax)[0][-1]
        else:
            fmax_idx = len(freq) - 1
        
        selected_freq = freq[fmin_idx:fmax_idx+1]
        selected_log_psd = log_psd[fmin_idx:fmax_idx+1]

        # ヒートマップのプロット
        cax = ax.imshow(selected_log_psd[np.newaxis, :], aspect='auto', cmap='viridis', origin='lower', extent=[selected_freq[0], selected_freq[-1], 0, 1])
        ax.set_title(muscle, fontsize=16)
        ax.set_yticks([])
        
    # 共通のX軸ラベル
    plt.xlabel('Frequency (Hz)', fontsize=16)
    
    # 共通のカラーバーをプロットの一番下に固定
    cbar_ax = fig.add_axes([0.125, 0.05, 0.775, 0.02])
    fig.colorbar(cax, cax=cbar_ax, orientation='horizontal', label='Log(PSD)')
    
    # レイアウト調整
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # カラーバーのためにスペースを確保
    plt.show()
    
    

def analyze_psd_bands(psd_young, psd_elderly, muscles):
#各周波数帯域のパワーの平均値をプロットする関数
#引数は２群のpsdの値(processdataで出てきたやつ)
    # 周波数帯の定義
    bands = {'alpha': (8, 12), 'beta': (13, 30), 'gamma': (31, 50)}
    
    # 平均パワーを格納する辞書を初期化
    avg_power_young = {band: [] for band in bands}
    avg_power_elderly = {band: [] for band in bands}
    
    def calculate_band_power(psd_data, freq_data, band):
        band_min, band_max = band
        freq_data = np.array(freq_data)  # リストをNumPy配列に変換
        psd_data = np.array(psd_data)  # psd_dataもNumPy配列に変換
        band_indices = (freq_data >= band_min) & (freq_data <= band_max)
        return np.mean(psd_data[band_indices])
    
    def collect_band_powers(psd_group, avg_power_group):
        for subject_psd in psd_group:
            for muscle in muscles:
                psd_data = subject_psd[muscle]['psd']
                freq_data = subject_psd[muscle]['freq']
                for band_name, band_range in bands.items():
                    band_power = calculate_band_power(psd_data, freq_data, band_range)
                    avg_power_group[band_name].append(band_power)
    
    # 各群のバンドパワーを収集
    collect_band_powers(psd_young, avg_power_young)
    collect_band_powers(psd_elderly, avg_power_elderly)
    
    # 平均値と標準誤差を計算
    avg_power_means = {}
    avg_power_std_errs = {}
    
    for band_name in bands:
        avg_power_means[band_name] = {
            'young': np.mean(avg_power_young[band_name]),
            'elderly': np.mean(avg_power_elderly[band_name])
        }
        avg_power_std_errs[band_name] = {
            'young': np.std(avg_power_young[band_name], ddof=1) / np.sqrt(len(avg_power_young[band_name])),
            'elderly': np.std(avg_power_elderly[band_name], ddof=1) / np.sqrt(len(avg_power_elderly[band_name]))
        }
    
    # 統計検定を実施
    p_values = {}
    for band_name in bands:
        young_data = avg_power_young[band_name]
        elderly_data = avg_power_elderly[band_name]
        
        # 正規性検定
        shapiro_young = shapiro(young_data)
        shapiro_elderly = shapiro(elderly_data)
        
        if shapiro_young.pvalue > 0.05 and shapiro_elderly.pvalue > 0.05:
            # 正規分布に従う場合
            stat, p_value = ttest_ind(young_data, elderly_data)
        else:
            # 正規分布に従わない場合
            stat, p_value = mannwhitneyu(young_data, elderly_data)
        
        p_values[band_name] = p_value
    
    # プロットの作成
    fig, ax = plt.subplots()
    x = np.arange(len(bands))
    width = 0.35
    
    young_means = [avg_power_means[band]['young'] for band in bands]
    elderly_means = [avg_power_means[band]['elderly'] for band in bands]
    young_errors = [avg_power_std_errs[band]['young'] for band in bands]
    elderly_errors = [avg_power_std_errs[band]['elderly'] for band in bands]
    
    bars1 = ax.bar(x - width/2, young_means, width, yerr=young_errors, label='Young', capsize=5)
    bars2 = ax.bar(x + width/2, elderly_means, width, yerr=elderly_errors, label='Elderly', capsize=5)
    
    # p値のアノテーション
    for i, band_name in enumerate(bands):
        p_value = p_values[band_name]
        if p_value < 0.01:
            annotation = '**'
        elif p_value < 0.05:
            annotation = '*'
        else:
            annotation = 'n.s.'
        
        max_height = max(young_means[i] + young_errors[i], elderly_means[i] + elderly_errors[i])
        ax.text(x[i], max_height * 1.05, annotation, ha='center', va='bottom', fontsize=12, color='red')
    
    ax.set_ylabel('Power')
    #ax.set_title('Power by frequency band and group')
    
    # x軸のラベルを周波数帯の定義に基づいて設定
    band_labels = [f'{band_name.capitalize()} ({band_range[0]}-{band_range[1]}Hz)' for band_name, band_range in bands.items()]
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels)
    
    ax.legend()
    
    fig.tight_layout()
    plt.show()

def find_median_frequency(psd_results):
        """
        面積を半分にするx軸の値（中央値の周波数）を計算
        """
        median_frequencies = {}
        for ch, (f, Pxx_den) in psd_results.items():
            cumulative_sum = np.cumsum(Pxx_den)
            total_area = cumulative_sum[-1]
            median_index = np.searchsorted(cumulative_sum, total_area / 2)
            median_frequency = f[median_index]
            median_frequencies[ch] = median_frequency
        
        return median_frequencies
      
def calculate_median_frequency(sub_range, data_type):
    med_freq_group = []

    for i in range(1, sub_range+1):
        sub_num = f"sub0{i:01d}"
        fname = glob.glob(f'/home/ohta/Atlas/data/Data_original/{data_type}/EMG/{sub_num}/*noRAS1.mat')[0] #globを関数として利用するときはglob.glob()とする
        emg = RawEMG(fname)
        muscles = emg.info['ch_names']
        emg.crop(15, 120)
        emg.filter(degree=4, high_freq=10, low_freq= 499, btype='bandpass')
        fig, psd_results = emg.plot_psd()
        plt.close(fig)
        
        # 中央値の周波数を計算
        median_frequencies = find_median_frequency(psd_results)
        med_freq_list = list(median_frequencies.values())
        med_freq_group.append(med_freq_list)

    return med_freq_group, muscles
  

def statistical_tests_and_plot(med_ferq_Young, med_ferq_Elderly, muscles):
    #med_ferq_Youngおよびmed_ferq_Elderlyはcalculate_median_frequency()の返り値
    
    # Shapiro-Wilk検定の結果を格納する辞書
    shapiro_results_Elderly = {}
    shapiro_results_Young = {}

    # 正規性の検定
    for muscle_idx in range(14):
        young_data = [subject[muscle_idx] for subject in med_ferq_Young]
        elderly_data = [subject[muscle_idx] for subject in med_ferq_Elderly]
        
        
        stat_e, p_value_e = shapiro(elderly_data)
        stat_y, p_value_y = shapiro(young_data)
        
        shapiro_results_Elderly[f"Muscle {muscle_idx + 1}"] = (stat_e, p_value_e)
        shapiro_results_Young[f"Muscle {muscle_idx + 1}"] = (stat_y, p_value_y)

    # t検定またはU検定を行う
    t_test_results = {}
    u_test_results = {}
    p_values = []

    for muscle_idx in range(14):
        muscle = f"Muscle {muscle_idx + 1}"
        elderly_data = [subject[muscle_idx] for subject in med_ferq_Elderly]
        young_data = [subject[muscle_idx] for subject in med_ferq_Young]
        
        # 両群ともに正規分布に従うかをチェック
        if shapiro_results_Elderly[muscle][1] > 0.05 and shapiro_results_Young[muscle][1] > 0.05:
            # 対応のないt検定
            t_stat, t_p_value = ttest_ind(elderly_data, young_data)
            t_test_results[muscle] = (t_stat, t_p_value)
            p_values.append(t_p_value)
        else:
            # U検定
            u_stat, u_p_value = mannwhitneyu(elderly_data, young_data)
            u_test_results[muscle] = (u_stat, u_p_value)
            p_values.append(u_p_value)

    # 結果の表示
    print("\nT-Test Results (for normally distributed data):")
    for muscle, result in t_test_results.items():
        print(f"{muscle}: T-Statistics={result[0]:.3f}, p-value={result[1]:.3f}")

    print("\nMann-Whitney U Test Results (for non-normally distributed data):")
    for muscle, result in u_test_results.items():
        print(f"{muscle}: U-Statistics={result[0]:.3f}, p-value={result[1]:.3f}")

    # 各グループの平均と標準偏差を計算
    means_young = [np.mean([subject[muscle_idx] for subject in med_ferq_Young]) for muscle_idx in range(14)]
    sd_young = [np.std([subject[muscle_idx] for subject in med_ferq_Young]) for muscle_idx in range(14)]
    means_eld = [np.mean([subject[muscle_idx] for subject in med_ferq_Elderly]) for muscle_idx in range(14)]
    sd_eld = [np.std([subject[muscle_idx] for subject in med_ferq_Elderly]) for muscle_idx in range(14)]

    # プロット
    N = len(muscles)
    bar_width = 0.35
    index = np.arange(N)

    fig, ax = plt.subplots(figsize=(12, 6))
    bar1 = ax.bar(index, means_young, bar_width, yerr=sd_young, label='Young', capsize=5)
    bar2 = ax.bar(index + bar_width, means_eld, bar_width, yerr=sd_eld, label='Elderly', capsize=5)

    ax.set_xlabel('Muscle')
    ax.set_ylabel('Mean Frequency')
    # ax.set_title('Mean Frequency of Young and Elderly for Different Muscles')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(muscles)
    ax.legend()

    # 有意なp値に注釈を追加
    for i, p in enumerate(p_values):
        if p < 0.009:
            annotation = '**'
        elif p < 0.049:
            annotation = '*'
        else:
            #continue  # Skip if p-value is not significant
            annotation = ' '
        max_height = max(means_young[i] + sd_young[i], means_eld[i] + sd_eld[i])
        #ax.text(index[i] + bar_width / 2, max_height + 0.1, annotation, ha='center', va='bottom', fontsize=12, color='red')
        ax.text(index[i] + bar_width / 2, max_height + 0.1, f"p={p:.3f} {annotation}", ha='center', va='bottom', fontsize=12, color='red')

    plt.tight_layout()
    plt.show()
