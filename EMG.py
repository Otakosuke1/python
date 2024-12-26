import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal
from scipy.interpolate import interp1d
import seaborn as sns
from copy import deepcopy

class EMG:
    '''
    EMGデータのクラス
    データ読み込みでMatlabファイルをデータフレームに変換し、中心化を行う。
    その後、前処理を行う。
    実装済みの前処理は、
    - ローパスフィルタ
    - ハイパスフィルタ
    - 平滑化
    - 整流化
    - ダウンサンプリング
    がある。
    '''
    # 基本変数の定義
    # _labels = ["rTA", "rSOL", "rGM", "rGL", "rVM", "rVL", "rHam", 
    #            "lTA", "lSOL", "lGM", "lGL", "lVM", "lVL", "lHam", "rfoot", "lfoot"]    
    # info = {}
    # info['sampling_rate'] = 1000
    
    def __init__(self, fname, name = None):
        self._labels = ["rTA", "rSOL", "rGM", "rGL", "rVM", "rVL", "rHam", 
                "lTA", "lSOL", "lGM", "lGL", "lVM", "lVL", "lHam", "rfoot", "lfoot"]    
        self.info = {}
        
        self.info['sampling_rate'] = 1000
        self._nyq = self.info['sampling_rate']/2
        self.info['low_freq'] = None   # ローパスフィルタ周波数
        self.info['high_freq'] = None  # ハイパスフィルタ周波数
        self.info['ch_names'] = self._labels[:14]
        
        mat = sp.io.loadmat(fname)
        _dat = mat["data"]
        _start_idx = mat["datastart"]
        _end_idx = mat["dataend"]
        
        self.data_matrix = pd.DataFrame()
        for s, e, la in zip(_start_idx, _end_idx, self._labels):
            if s == -1:
                continue
            else:
                self.data_matrix[la] = _dat[0][int(s-1):int(e-1)]
        
        self.__data_matrix__ = self.data_matrix.copy()
        self.__foot_sensor__ = self.data_matrix.iloc[:,14:].copy()

        for i, foo in enumerate(["rfoot", "lfoot"]):
            _rt = self.data_matrix[foo].values
            _rt = _rt - min(_rt)
            _rt[_rt < np.max(_rt)/5] = 0
            _events = [i+1 if _rt[j-1] == 0 and _rt[j]>0 else 0 for j in range(len(_rt))]
            _events = pd.Series(_events)
            idx = _events[_events != 0].index
            skip=False
            for i in range(len(idx)):
                if i == len(idx)-2:
                    break
                elif idx[i] == 0|idx[i-1]==0:
                    continue
                elif skip:
                    skip = False
                    continue
                lng = idx[i+2]-idx[i+1]
                sht = idx[i+1]-idx[i]
                
                if lng/2>sht:
                    _events[idx[i+1]]=0
                    skip = True
            
            self.data_matrix[foo] = _events
        
        self.foot_sensor = self.data_matrix[["rfoot", "lfoot"]]
        self.events = self.foot_sensor.max(axis=1) # eventsというイベントデータ(0: no event,1: rt_foot,2: lt_foot)
        
        emg = self.data_matrix.iloc[:,:14]   # rawというEMGデータ
        
        self.emg_matrix = emg - emg.mean()  # 中心化
        self.emg_raw = self.emg_matrix.copy()
        self.cadence = self.culc_cadence()
        if name:
            self.name = name
        else:
            self.name = fname
        

    def _reset_data(self):
        self.emg_matrix = self.data_matrix.iloc[:,:14].copy()
        self.foot_sensor = self.data_matrix.iloc[:,14:].copy()
        self.events = self.foot_sensor.max(axis=1)
    
    def crop(self, tmin = None, tmax = None):
        """
        crop the data
        tmin: start time (sec)
        tmax: end time (sec)
        """
        fs = self.info['sampling_rate']
        if tmin is None:
            tmin = 0
        if tmax is None:
            tmax = len(self.data_matrix)

        tmin, tmax = int(tmin*fs), int(tmax*fs)
        self.data_matrix = self.data_matrix.iloc[tmin:tmax,:].reset_index(drop=True)
        self.__data_matrix__ = self.__data_matrix__.iloc[tmin:tmax,:].copy()
        self.__foot_sensor__ = self.__data_matrix__.iloc[:,14:]
        self.emg_raw = self.emg_raw.iloc[tmin:tmax,:]
        self._reset_data()
        return self

    def culc_epoch_len(self, foot = "Rt"):
        if "Rt" in foot:
            idx = self.events[self.events == 1].index
        elif "Lt" in foot:
            idx = self.events[self.events == 2].index
        else:
            KeyError("foot must be 'Rt' or 'Lt'" )
        length = []
        for i in range(len(idx)):
            if i == len(idx)-1:
                break
            length.append(idx[i+1]-idx[i])
        return length
    
    def culc_cadence(self,foot="Rt"):
        self.mean_length = np.median(self.culc_epoch_len())
        return 60/self.mean_length*self.info['sampling_rate']
    
    def copy(self):
        return deepcopy(self)
    
    def plot(self, tmin=None, tmax=None):
        ax =  self.emg_matrix.plot(figsize=(20,10), subplots=True, sharex=True, title=self.name)
        for a in ax:
            a.legend(loc='upper right')
        if tmin is not None:
            tmin = int(tmin*self.info['sampling_rate'])
        if tmax is not None:
            tmax = int(tmax*self.info['sampling_rate'])
        if tmin is not None or tmax is not None:
            plt.xlim(tmin, tmax)
        return ax
    
    def downsampling(self, q=5):
        """
        downsampling the data
        q: downsampling point (int object, default=5)
        """
        self.info['sampling_rate'] = int(self.info['sampling_rate']/q)
        self.__data_matrix__.iloc[:,14:] = self.__foot_sensor__
        __dat__ = signal.decimate(self.__data_matrix__, q, axis=0)  # __data_matrix__の更新
        __dat__ = pd.DataFrame(__dat__, columns=self.data_matrix.columns)
        self.__data_matrix__ = __dat__.copy()
        dat = signal.decimate(self.data_matrix, q, axis=0)  # data_matrixの更新
        dat = pd.DataFrame(dat, columns=self.data_matrix.columns)
        self.data_matrix = dat.copy()
        self._reset_data()

        for i, foo in enumerate(["rfoot", "lfoot"]):
            _rt = self.__data_matrix__[foo].values
            _rt = _rt - min(_rt)
            _rt[_rt < np.max(_rt)/5] = 0
            _events = [i+1 if _rt[j-1] == 0 and _rt[j]>0 else 0 for j in range(len(_rt))]
            _events = pd.Series(_events)
            idx = _events[_events != 0].index
            skip=False
            for i in range(len(idx)):
                if i == len(idx)-2:
                    break
                elif idx[i] == 0|idx[i-1]==0:
                    continue
                elif skip:
                    skip = False
                    continue
                lng = idx[i+2]-idx[i+1]
                sht = idx[i+1]-idx[i]
                
                if lng/2>sht:
                    _events[idx[i+1]]=0
                    skip = True
            self.__data_matrix__[foo] = _events
            self.data_matrix[foo] = _events
        
        self.foot_sensor = self.data_matrix[["rfoot", "lfoot"]]
        self.events = self.foot_sensor.max(axis=1) # eventsというイベントデータ(0: no event,1: rt_foot,2: lt_foot)


class RawEMG(EMG):
    def __init__(self, data, name=None):
        super().__init__(data, name)
    
    def rectification(self):
        """
        all wave rectification
        """
        self.data_matrix.iloc[:,:14] = self.emg_matrix.abs()
        self._reset_data()

    def filter(self, degree = 4, high_freq = 10, low_freq = 500, btype = "bandpass"):
        """
        filter the EMG data by butterworth filter
        degree: degree of filter
        high_freq: high frequency of filter
        low_freq: low frequency of filter
        btype: type of filter
        """
        self.info['high_freq'] = high_freq
        self.info['low_freq'] = low_freq
        emg = self.emg_matrix #.abs()
        
        h_freq = high_freq/self._nyq
        l_freq = low_freq/self._nyq
        b, a = sp.signal.butter(degree, [h_freq, l_freq], btype=btype)
        for ch in emg.columns:
            emg[ch] = sp.signal.filtfilt(b, a, emg[ch])
        self.data_matrix.iloc[:,:14] = emg
        self._reset_data()

    def plot_psd(self, picks=None, fmin=None, fmax=None):
        """
        plot power spectral density
        picks: channels to plot
        """
        fig, ax = plt.subplots(figsize=(12,7))
        color = plt.get_cmap("tab10")
        psd_results = {} #add 2024/06/03
        for i, ch in enumerate(self.emg_matrix.columns):
            if picks is not None:
                if ch not in picks:
                    continue
            #f, Pxx_den = sp.signal.welch(self.emg_matrix[ch], fs=self.info['sampling_rate'], nperseg=1024)
            #周波数分解能を1Hz、オーバーラップは50%（デフォルト）
            f, Pxx_den = sp.signal.welch(self.emg_matrix[ch], fs=self.info['sampling_rate'], nperseg=self.info['sampling_rate']) 
            # ax.semilogy(f, Pxx_den, label=ch, color=colors[i])
            ax.semilogy(f, Pxx_den, label=ch, color=color(i/len(self.emg_matrix.columns)))
            psd_results[ch] = (f, Pxx_den) #####add 2024/06/03#####
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('$µV^2/Hz(dB)$')
        # legendは図の外側に配置
        ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center', borderaxespad=0, fontsize=18)
        ax.grid(alpha=0.5)
        if fmin is not None:
            ax.set_xlim(left = fmin)
        if fmax is not None:
            ax.set_xlim(right = fmax)
        return fig, psd_results #####add 2024_06_03#####
        

    def smooth(self, method="low-pass", degree = 4, freq=20, window=100):
        """
        smoothing the EMG data
        method: "low-path" or "movag" (low-pass filter or moving average)
        degree: degree of filter (only low-pass filter)
        freq: low-pass filter frequency (only low-pass filter)
        window: window size of moving average (only moving average)
        """
        # print("smoothing...")
        return SmoothEMG(self, method, degree, freq, window)
    

class SmoothEMG(EMG):
    def __init__(self, emg, method="low-pass", degree = 4, freq=20, window=100):
        """
        smoothing the EMG data
        method: "low-path" or "movag" (low-pass filter or moving average)
        degree: degree of filter (only low-pass filter)
        freq: low-pass filter frequency (only low-pass filter)
        window: window size of moving average (only moving average)
        """
        self.info = emg.info
        self._labels = emg._labels
        self.name = emg.name
        self.info['smoothing'] = method
        self.data_matrix = emg.data_matrix.copy()
        self.__data_matrix__ = emg.__data_matrix__.copy()
        self.__foot_sensor__ = emg.__foot_sensor__.copy()
        
        emg = emg.emg_matrix
        
        if method == "low-pass":
            nyq = self.info['sampling_rate']/2
            low_pass = freq/nyq
            
            b2, a2 = sp.signal.butter(degree, low_pass, btype = 'lowpass')
            for ch in emg.columns:
                emg[ch] = sp.signal.filtfilt(b2, a2, emg[ch])
                emg[ch] = np.abs(sp.signal.hilbert(emg[ch]))
        elif method == "movag":
            emg = emg.abs()
            emg = emg.rolling(window=window).mean()
        else:
            KeyError("Method is value error. Use 'low-pass' or 'movag'.")
        self.data_matrix.iloc[:,:14] = emg
        self._reset_data()
        self.emg_raw = self.emg_matrix

