# %%
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
class MuscleSynergy:
    def __init__(self, max_n_components, max_iter = 200, rep = 20, vaf = 0.75, vaf_mus = 0.9):
        self.max_n_components = max_n_components
        self.best_n = None
        self.best_syn = None
        self.vaf_log = "You must do est_best_n function before this command"
        self.nmf_log = "You must do est_best_n function before this command"
        self.max_iter = max_iter
        self.rep = rep
        self.vaf_threshold = vaf
        self.vaf_mus_threshold = vaf_mus

    def fit(self, epochs, subject=None, bilateral=True, scale=True):
        self.subject = subject
        self.info = epochs.info.copy()
        slice_idx = [i for i in range(len(epochs.data)) if not i in epochs.info['drop_epochs']]
        if not bilateral:
            X = epochs.data[slice_idx,:,:].mean(axis=0)
            self.label = self.info['ch_names']
        else:
            X = epochs._data[slice_idx,:,:].mean(axis=0)
            self.label = self.info['all_channels']
        if scale:
            scaler = MinMaxScaler()
            self.X = scaler.fit_transform(X)
        else:
            self.X = X

    def _culc_loss(self, X, n_components = None):
        nmf = NMF(n_components=n_components, max_iter=self.max_iter)
        nmf.fit(X)
        W = nmf.components_
        C = nmf.fit_transform(X)
        WC = np.dot(C, W)
        return np.sum((X - WC)**2), nmf

    def f_nmf(self, n_components = None, max_iter=20):
        old_loss, old_nmf = self._culc_loss(X = self.X, n_components=n_components)
        for i in range(max_iter - 1):
            loss, nmf = self._culc_loss(X = self.X, n_components=n_components)
            if loss < old_loss:
                old_loss = loss
                old_nmf = nmf
        return old_nmf

    def _culc_vaf(self, nmf):
        """
        Permetrics.regressionに実装されているが、気づかなかったので今回はこのままで。
        今後修正していこうと思う。
        https://permetrics.readthedocs.io/en/latest/pages/regression/VAF.html
        """
        W = nmf.components_
        C = nmf.fit_transform(self.X)
        WC = np.dot(C, W)
        e = self.X - WC
        vaf = 1 - (np.sum(e**2)/np.sum(self.X**2))
        return vaf

    def _culc_vaf_mus(self, nmf, axis = 0):
        W = nmf.components_
        C = nmf.fit_transform(self.X)
        WC = np.dot(C, W)
        e = self.X - WC
        vaf_mus = []
        if axis == 0:
            for i in range(self.X.shape[1]):
                xi = self.X.T[i]
                ei = e.T[i]
                vaf_i = 1 - (np.sum(ei**2)/np.sum(xi**2))
                vaf_mus.append(vaf_i)
        else:
            for i in range(self.X.shape[0]):
                xi = self.X[i]
                ei = e[i]
                vaf_i = 1 - (np.sum(ei**2)/np.sum(xi**2))
                vaf_mus.append(vaf_i)
        return vaf_mus

    def est_best_n(self, threshold = 0.75):
        self.vaf_log = []
        self.nmf_log = {}
        for n in range(self.max_n_components):
            nmf = self.f_nmf(n+1, max_iter=self.rep)
            vaf = self._culc_vaf(nmf)
            # print(n + 1, vaf)
            self.vaf_log.append(vaf)
            self.nmf_log[f"{n+1}"] = nmf
        
        for i, vaf in enumerate(self.vaf_log):
            if vaf > self.vaf_threshold:
                nmf = self.nmf_log[f"{i+1}"]
                vaf_mus = self._culc_vaf_mus(nmf)
                if np.min(vaf_mus) > self.vaf_mus_threshold:
                    # print(i+1, vaf, vaf_mus)
                    break
        self.best_n = i+1
        best_syn = self.nmf_log[f"{i+1}"]
        self.best_syn = {"W":best_syn.components_, "C":best_syn.fit_transform(self.X)}
        return i+1

    def plot_vaf(self, ax = None):
        if not ax:
            fig, ax = plt.subplots()
        ax.plot(self.vaf_log, '--', color = "black", marker = 'o', markeredgecolor = 'black', markerfacecolor = 'white')
        ax.set_title(self.subject)
        ax.set_xticks(np.arange(0,self.max_n_components),np.arange(1, self.max_n_components+1))

    def plot_W(self, n, ax = None):
        if not ax:
            fig, ax = plt.subplots()
        ax.bar(np.arange(len(self.best_syn["W"][n-1])), self.best_syn["W"][n-1], color = f"C{n-1}",
               tick_label = self.label)
        ax.tick_params(axis='x', labelrotation=90)

    def plot_C(self, n, ax = None):
        if not ax:
            fig, ax = plt.subplots()
        t = np.linspace(0, 100, len(self.best_syn['C'].T[n-1]))
        ax.plot(t, self.best_syn['C'].T[n-1], color = f'C{n-1}')
        ax.set_xlabel("time(%)")

    def plot_synergies(self, figsize = None):
        if not figsize:
            figsize = (10, self.best_n*4)
        fig, ax = plt.subplots(self.best_n, 2, figsize = figsize)
        for i in range(self.best_n):
            self.plot_W(i+1, ax=ax[i,0])
            self.plot_C(i+1, ax=ax[i,1])
        plt.suptitle(self.subject, y = 0.99)
        fig.tight_layout()

