# %%
import sys
sys.path.append('/home/owner/shinyEMG/share')


import scipy as sp
import scipy.signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pycwt
import skfda
from copy import deepcopy
from tools.prep import align_epochs, create_emg_epochs

class Epochs:
    """
    Epochデータのクラス
    EMGクラスのインスタンスを引数に取り、データ内のイベントデータをもとにEpochデータを作成する。
    前処理を実装している。
    実装されている前処理は以下の通り。
    - フィルタリング
    - スムージング
    - 時間正規化
    """
    def __init__(self, raw, foot='r'):
        self.info = raw.info.copy()
        self.info['drop_epochs'] = []
        
        self._labels = raw._labels[:14]

        self.info['foot'] = foot
        self.info['all_channels'] = self._labels.copy()
        self.info['ch_names'] = [la for la in self._labels if foot in la]
        self._nyq = self.info['sampling_rate']/2
        self.raw_data = create_emg_epochs(raw, foot)
        self._data = [dat.abs() for dat in self.raw_data]
        self._update_data()
    
    def _update_data(self):
        if type(self._data) == np.ndarray:
            if self.info['foot'] == 'r':
                self.data = self._data[:,:,:7].copy()
            elif self.info['foot'] == 'l':
                self.data = self._data[:,:,7:].copy()
        elif type(self._data) == list:
            self.data = [dat.loc[:,self.info['ch_names']] for dat in self._data]
        
        
    def filtering(self, degree=4, h_freq=0.5, l_freq=200, btype='bandpass'):
        self.info['high_freq'], self.info['low_freq'] = h_freq, l_freq
        h_freq = h_freq/self._nyq
        l_freq = l_freq/self._nyq
        b, a = sp.signal.butter(degree, [h_freq, l_freq], btype=btype)
        dat = self._data
        filtered = []
        for d in dat:
            d_len=d.shape[0]
            dd = d.copy()
            d_rev = dd.iloc[::-1].copy()
            d = pd.concat([d_rev,dd,d_rev], axis=0)
            for ch in d.columns:
                d[ch]=sp.signal.filtfilt(b,a,d[ch])
            d = d.iloc[d_len:2*d_len,:]
            filtered.append(d)
        self._data = filtered.copy()
        self._update_data()
        

    def smoothing(self, degree=4, freq=20):
        low_pass = freq/self._nyq
        b2, a2 = sp.signal.butter(degree, low_pass, btype = 'lowpass')
        dat = self._data
        smoothed = []
        
        for d in dat:
            d_len=d.shape[0]
            dd = d.copy()
            d_rev = dd.iloc[::-1].copy()
            d = pd.concat([d_rev,dd,d_rev], axis=0)
            for ch in d.columns:
                d[ch]=sp.signal.filtfilt(b2,a2,d[ch])
                d[ch]=np.abs(sp.signal.hilbert(d[ch]))
            d = d.iloc[d_len:2*d_len,:]
            smoothed.append(d)
        self._data = smoothed.copy()
        self._update_data()
        
    def llnormalization(self, n=100):
        self._data = align_epochs(self._data, n=n)
        if self.info['foot'] == 'r':
            self.data = self._data[:,:,:7].copy()
        else:
            self.data = self._data[:,:,7:].copy()
        self.info['lln'] = True
        
    def _drop_epochs(self, drop_idx):
        if drop_idx == []:
            self.drop_data = []
            return
        if type(drop_idx)!=list:
            TypeError('drop index must be list object.')
        elif type(drop_idx[0])!=int:
            TypeError('drop index must be list of integer.')
        # drop_idx = drop_idx.sort()
        
        slice_idx = [i for i in range(len(self._data)) if not i in drop_idx]
        self.all_data = self._data.copy()
        if type(self._data) == np.ndarray:
            self._data = self.all_data[slice_idx,:,:]
            self._update_data()
            self.drop_data = self.all_data[drop_idx,:,:]
        elif type(self._data) == list:
            self._data = [self.all_data[i] for i in slice_idx]
            self._update_data()
            self.drop_data = [self.all_data[i] for i in drop_idx]

    def copy(self):
        return deepcopy(self)
    
    def plot(self, bilaterally=True, ch_names=None, drop=False):
        """
        plot epochs data
        bilaterally: bool if True, plot all channels in 2x7 subplots
        ch_names: list of str, channel names to plot
        drop: bool, if True, show dropped epochs in red
        """
        dat = self.copy()
        dat = NormEpochs(dat)
        dat.plot(bilaterally, ch_names, drop)

def create_norm_epochs(emg, foot='r', n=100):
    """
    emg: EMG class object
    foot: select foot 'Rt' or 'Lt'
    n: number of points to align
    """
    epochs = Epochs(emg, foot)
    return NormEpochs(epochs, n=n)

class NormEpochs(Epochs):
    def __init__(self, epochs, n=100):
        # super().__init__(raw, foot)
        self.info = epochs.info.copy()
        self._llnormalization(epochs, n)
    
    def _llnormalization(self, epochs, n=100):
        self._data = align_epochs(epochs._data, n=n)
        if self.info['foot'] == 'r':
            self.data = self._data[:,:,:7].copy()
        else:
            self.data = self._data[:,:,7:].copy()
    
    def plot(self, bilaterally=False, ch_names=None, drop=False):
        """
        plot epochs data
        bilaterally: bool if True, plot all channels in 2x7 subplots
        ch_names: list of str, channel names to plot
        drop: bool, if True, show dropped epochs in red
        """
        dat = self.copy()
        dat._drop_epochs(dat.info['drop_epochs'])
        if ch_names is locals():
            dat.info['ch_names'] = ch_names
            
        if drop:
            drop_data = dat.drop_data   # the plot data
        data = dat.data   # the plot data
        
        if bilaterally:
            fig, ax = plt.subplots(7,2, figsize=(10,10))
            for i, a in enumerate(ax.flatten(order='F')):
                a.plot(dat._data[:,:,i].T, color='black', alpha=0.5)
                if drop:                     # ここでdrop_dataが空の場合の処理をしている
                    if dat.drop_data == []:  # 逃げの選択なので、今後修正が必要になる可能性がある
                        continue
                    a.plot(dat.drop_data[:,:,i].T, color='red', alpha=0.5)
                # a.set_title(dat.info['all_channels'][i])
        else:
            fig, ax = plt.subplots(7,1, figsize=(10,10))
            for i, a in enumerate(ax.flatten(order='F')):
                a.plot(data[:,:,i].T, color='black', alpha=0.5)
                if drop:                     # ここでdrop_dataが空の場合の処理をしている
                    if dat.drop_data == []:  # 逃げの選択なので、今後修正が必要になる可能性がある
                        continue
                    a.plot(drop_data[:,:,i].T, color='red', alpha=0.5)
                a.set_title(dat.info['ch_names'][i])
