from scipy.interpolate import interp1d
import numpy as np
from numpy.fft import fft
#import skfda
#from skfda.exploratory.visualization import Boxplot
import pandas as pd
from scipy import signal
#import pycwt
import matplotlib.pyplot as plt

def approx(x, method='linear', n=100):
    y = np.arange(0, len(x), 1)
    f = interp1d(y, x, kind = method)
    y_resample = np.linspace(0, len(x)-1, n)
    return f(y_resample)

def spectrum(data, sampling_rate):
    if type(data) == pd.DataFrame:
        label = data.columns
        for la in data:
            dat = data[la]
            fxx = fft(dat)
            tmax = len(dat)
            Amp = np.abs(fxx/(tmax/2))
            if 'amp_df' in locals():
                amp_df[la] = Amp
            else:
                amp_df = pd.DataFrame({la : Amp})
        amp_df.index = np.linspace(0,sampling_rate, amp_df.shape[0])
        return amp_df
    else:
        pass

def plot_wct(wct, freq, time_length, muscle_label, ax=None, coi=None):
    if not ax:
        fig,ax = plt.subplots()
    t = np.linspace(0,time_length, wct.shape[0])
    period = 1/freq
    
    ax.contourf(t,np.log2(freq),wct,extend='both')
    if coi:
        ax.fill(np.concatenate([t, t[-1:], t[-1:],
                                t[:1], t[:1]]),
                -np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]),
                                np.log2(period[-1:]), [1e-9]]),
                'w', alpha=0.3, hatch='x')
    ax.set_ylim(ymin=np.log2(1), ymax=np.log2(50))
    ax.set_yticklabels(np.arange(1,7)**2)
    ax.set_title('c) wavelet coherence spectrum(morlet)')
    ax.set_ylabel('frequency(Hz)')
    plt.grid()
    return ax

def smooth_signal(data, fs, high=10, low=100,fq=10, show=True):
    nyq=0.5*fs
    high, low = high/nyq, low/nyq
    # print(high, low)
    data_rev = np.array(data)[::-1]
    data_copy = np.concatenate([data_rev, data, data_rev])

    b, a = signal.butter(4, [high, low], btype='band')
    data_filt = signal.filtfilt(b, a, np.abs(data_copy))

    fq = fq/nyq
    b2, a2 = signal.butter(4, fq, btype='low')
    data_filt = signal.filtfilt(b2, a2, data_filt)
    data_sm = np.abs(signal.hilbert(data_filt))
    data_filt = data_filt[len(data):2*len(data)]
    data_sm = data_sm[len(data):2*len(data)]
    if show:
        plt.plot(data_filt)
        plt.plot(data_sm)
    return data_sm