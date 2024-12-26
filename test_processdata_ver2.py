#データ処理の関数の修正版
import pandas as pd
from glob import glob
from sklearn.preprocessing import MinMaxScaler
from tools.EMG import *                 
from tools.Epoch import *  

def group_emg_data(data_type, ras, num_subjects=4):
    emg_data = []
    epochs_data = []
    norm_data = []
    psd = []

    for i in range(1, num_subjects + 1):
        sub_num = f"sub0{i:01d}"
        fname = glob.glob(f'/home/ohta/Atlas/data/Data_original/{data_type}/EMG/{sub_num}/*{ras}.mat')[0]
        emg = RawEMG(fname)
        
        muscles = emg.info['ch_names']
        emg.crop(15, 120)
        emg.filter(degree=4, high_freq=10, low_freq=499, btype='bandpass')
        
        psd_indi = {muscle: {'psd': [], 'freq': []} for muscle in muscles}
        fig, psd_results = emg.plot_psd(muscles, fmin=0, fmax=60)
        plt.close(fig)
        # 各筋肉のPSD値と周波数をリストに格納
        psd_values = {muscle: {'psd': psd_results[muscle][1], 'freq': psd_results[muscle][0]} for muscle in muscles}
        # 各筋肉のPSD値と周波数を全体の辞書に追加（行方向に追加）
        for muscle in muscles:
            psd_indi[muscle]['psd'].append(psd_values[muscle]['psd'])
            psd_indi[muscle]['freq'].append(psd_values[muscle]['freq'])
        psd.append(psd_indi)

        emg.rectification()
        emg = emg.smooth(method='low-pass', freq=6)
        emg.downsampling(q=5)
       
        df = pd.DataFrame(emg.__data_matrix__)
        emg_data.append(df)
        
        epochs = create_norm_epochs(emg, foot='r', n=100)
        epochs_data.append(epochs._data)
        
        averaged_data = np.mean(epochs._data, axis=0)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(averaged_data)
        norm_data.append(scaled_data)

    return emg_data, epochs_data, norm_data, psd, muscles
