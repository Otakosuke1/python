import numpy as np
import scipy.signal as signal
# 時間または周波数方向に三角窓で平滑化する関数
def smoothing(sp, window=np.array([1.0, 2.0, 1.0])):
    '''
    sp: data of time-frequency domain(2D array)
    window: window function for smoothing(1D array)
            default: np.array([1.0, 2.0, 1.0])
            in scipy.signal or other modules(numpy, scipy, etc), 
            there are many window functions        
    '''
    n_f, n_t = sp.shape
    sp_smthd = np.zeros_like(sp)

    for i in range(n_t):
        krnl = window
        sp_smthd[:, i] = np.convolve(sp[:, i], krnl, mode='same')/np.sum(krnl)

    for j in range(n_f):
        krnl = window
        sp_smthd[j, :] = np.convolve(sp_smthd[j, :], krnl, mode='same')/np.sum(krnl)

    return sp_smthd

#------------------------------------------------------------------------------
def stcoh(sig01, sig02, fs, nperseg=512, noverlap=0):
    # STFTを実行
    f, t, sp01 = signal.stft(sig01, fs, nperseg=nperseg, noverlap=noverlap)
    f, t, sp02 = signal.stft(sig02, fs, nperseg=nperseg, noverlap=noverlap)
    # クロススペクトルの計算
    xsp = sp01*np.conjugate(sp02)

    # コヒーレンスとフェイズの算出
    sp01_pw_smthd = smoothing(np.abs(sp01)**2) # stftからパワーを計算し平滑化
    sp02_pw_smthd = smoothing(np.abs(sp02)**2) # コヒーレンスは定義から平均を取らないと
    xsp_smthd = smoothing(xsp)                 # 全て1になってしまう

    coh = np.abs(xsp_smthd)**2 / (sp01_pw_smthd*sp02_pw_smthd) # （二乗）コヒーレンス
    phs = np.rad2deg(np.arctan2(np.imag(xsp_smthd), np.real(xsp_smthd))) # フェイズ
    return f, t, coh, phs
