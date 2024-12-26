import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import uniform_filter1d

class EMGAnalyzer:
    def __init__(self, emg_data):
        self.cycles = emg_data['cycles']
        self.emg = emg_data['emg']
        self.preprocessed_data = self.emg.copy()
        self.emg_matrix = self.emg.iloc[:, 1:].values  # 時間列を除いたEMGデータのマトリックス
        self.events = self.cycles['V1']  # タッチダウンイベント（時刻）
    
    def apply_bandpass_filter(self, lowcut=20, highcut=450, fs=1000, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        
        for muscle in self.emg.columns[1:]:
            self.preprocessed_data[muscle] = signal.filtfilt(b, a, self.emg[muscle])
    
    def smooth_data(self, window_size=10):
        for muscle in self.emg.columns[1:]:
            self.preprocessed_data[muscle] = self.preprocessed_data[muscle].rolling(window=window_size, min_periods=1).mean()
    
    def rectify_data(self):
        for muscle in self.emg.columns[1:]:
            self.preprocessed_data[muscle] = np.abs(self.preprocessed_data[muscle])
    
    def plot_data(self, muscles=['ME', 'MA'], start_time=None, end_time=None):
        plt.figure(figsize=(10, 6))
        for muscle in muscles:
            if muscle in self.preprocessed_data.columns:
                data_to_plot = self.preprocessed_data[(self.preprocessed_data['time'] >= start_time) & (self.preprocessed_data['time'] <= end_time)] if start_time and end_time else self.preprocessed_data
                plt.plot(data_to_plot['time'], data_to_plot[muscle], label=muscle)
        plt.xlabel('Time (s)')
        plt.ylabel('EMG Signal')
        plt.legend()
        plt.show()
    
    def calculate_cadence(self):
        cycle_durations = self.cycles['V2'] - self.cycles['V1']
        cadence = 60 / cycle_durations.mean()
        return cadence

    def downsample(self, factor):
        self.preprocessed_data = self.preprocessed_data.iloc[::factor, :].reset_index(drop=True)
        
        
    def create_emg_epochs(self):
        """
        Creates epochs from the processed EMG data based on touchdown events.

        Returns:
        List of pandas.DataFrame: A list of epochs where each epoch is a DataFrame.
        """
        # 処理されたデータのマトリックスを使用
        emg_matrix_processed = self.preprocessed_data.iloc[:, 1:].values
        events = self.events
        emg_epochs = self.gen_epochs(emg_matrix_processed, events)
        return emg_epochs
    
    
    def gen_epochs(self, array, events):
        """
        Generates epochs from the data array based on events.

        Parameters:
        array (ndarray): 2D array of data (time series x channels).
        events (Series): Series of events (must correspond to the time series of the array).

        Returns:
        List of pandas.DataFrame: List of epochs.
        """
        # イベントインデックスを取得
        events_index = np.searchsorted(self.preprocessed_data['time'], events)
        df = pd.DataFrame(array)
        epochs = []
        for ev in range(len(events_index)):
            if ev + 1 == len(events_index):
                break
            else:
                # イベント間のデータをエポックとしてスライス
                epochs.append(df.iloc[events_index[ev]:events_index[ev + 1]].reset_index(drop=True))
        return epochs
      
    def align_epochs(self, epochs, n=100):
        """
        Aligns epochs to a specified number of points.

        Parameters:
        epochs (list of DataFrame): List of epochs to align.
        n (int): Number of points to align to.

        Returns:
        ndarray: Aligned epochs.
        """
        aln_epochs = np.zeros((len(epochs), n, epochs[0].shape[1]))
        for i, epoch in enumerate(epochs):
            for j in range(epoch.shape[1]):
                x_old = np.linspace(0, 1, epoch.shape[0])
                x_new = np.linspace(0, 1, n)
                aln_epochs[i, :, j] = np.interp(x_new, x_old, epoch.iloc[:, j].values)
        return aln_epochs
      


    def align_epochs(self, epochs, n=100, smooth_window=20):
        """
        Aligns epochs to a specified number of points and applies smoothing.
    
        Parameters:
        epochs (list of DataFrame): List of epochs to align.
        n (int): Number of points to align to.
        smooth_window (int): Window size for smoothing.
    
        Returns:
        ndarray: Aligned and smoothed epochs.
        """
        # 初期化：n個のデータポイントに合わせる
        aln_epochs = np.zeros((len(epochs), n, epochs[0].shape[1]))
    
        for i, epoch in enumerate(epochs):
            for j in range(epoch.shape[1]):
                # 線形補間によるデータのアラインメント
                x_old = np.linspace(0, 1, epoch.shape[0])
                x_new = np.linspace(0, 1, n)
                aln_epochs[i, :, j] = np.interp(x_new, x_old, epoch.iloc[:, j].values)
    
            # 各エポックに対して平滑化を適用
            for j in range(epoch.shape[1]):
                aln_epochs[i, :, j] = uniform_filter1d(aln_epochs[i, :, j], size=smooth_window)
    
        return aln_epochs

      
    def process_emg(self, n=100):
        """
        Processes the EMG data to create and align epochs.

        Parameters:
        n (int): Number of points to align to.

        Returns:
        ndarray: Aligned epochs.
        """
        emg_epochs = self.create_emg_epochs()  # エポックを作成
        aligned_epochs = self.align_epochs(emg_epochs, n=n)  # エポックを整列
        return aligned_epochs
  
    def scaler(self, aligned_epochs):
        """
        Averages aligned epochs along the first dimension and applies MinMaxScaler.
    
        Parameters:
        aligned_epochs (ndarray): 3D array of aligned epochs (epochs x points x channels).
    
        Returns:
        ndarray: Scaled 2D array (points x channels).
        """
        # エポックの方向（1次元目）で平均をとる
        mean_epochs = np.mean(aligned_epochs, axis=0)  # (points x channels)
    
        # MinMaxScalerを適用
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_mean_epochs = scaler.fit_transform(mean_epochs)
    
        return scaled_mean_epochs
