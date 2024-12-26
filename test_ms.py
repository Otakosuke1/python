import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
class ms:
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

    # def fit(self, muscles, data, bilateral= False, subject=None, scale=True):
    #     label_bilateral = ["rTA", "rSOL", "rGM", "rGL", "rVM", "rVL", "rHam", "lTA", "lSOL", "lGM","lGL", "lVM", "lVL", "lHam"]
    #     label_nonbilateral = ["rTA", "rSOL", "rGM", "rGL", "rVM", "rVL", "rHam"]
    #     label_other_data = muscles
    #     #self.info = epochs.info.copy()
    #     #slice_idx = [i for i in range(len(data)) if not i in epochs.info['drop_epochs']]
    #     #平均値を取るのは、データの整理の段階で行なった
    #     if not bilateral:
    #         X = data[:,:7]
    #         self.label = label_nonbilateral
    #     else:
    #         X = data
    #         self.label = label_other_data
    #     
    #     #X = data
    #     if scale:
    #         scaler = MinMaxScaler()
    #         self.X = scaler.fit_transform(X)
    #     else:
    #         self.X = data
    
    def fit(self, data, muscles, bilateral=False, subject=None, scale=True):
        # データとmusclesのリストの長さを取得
        num_muscles = len(muscles)
        num_data_columns = data.shape[1]
    
        # `muscles`の個数に基づいてデータを切り出し
        if num_muscles != num_data_columns:
            X = data[:, :num_muscles]
        else:
            X = data
        
        # `muscles`リストをデータのラベルに設定
        self.label = muscles
    
        # スケーリングの処理
        if scale:
            scaler = MinMaxScaler()
            self.X = scaler.fit_transform(X)
        else:
            self.X = X


    # def _culc_loss(self, X, n_components = None):
    #     nmf = NMF(n_components=n_components, max_iter=self.max_iter)
    #     nmf.fit(X)
    #     W = nmf.components_
    #     C = nmf.fit_transform(X)
    #     WC = np.dot(C, W)
    #     return np.sum((X - WC)**2), nmf
    #############add 2024/07/11##########################
    def _culc_loss(self, X, threshold=0.90, tolerance=0.05, n_components = None):
        # 元の行列の合計二乗和
        total_variance = np.sum(X ** 2)

        # 初期化
        previous_residual_variance_ratio = 0
        optimal_n_components = 1
        nmf_model = None

        # 残差が90%を超えるか、残差が0.05以上大きくならないタイミングを見つける
        for n_components in range(1, X.shape[1] + 1):
            nmf = NMF(n_components=n_components, max_iter=self.max_iter, init='random', random_state=42)
            C = nmf.fit_transform(X)
            W = nmf.components_
            WC = np.dot(C, W)

            # 残差を計算
            residual = np.sum((X - WC) ** 2)
            residual_variance_ratio = 1 - (residual / total_variance)

            # 現在の残差割合と前回の残差割合の差を計算
            residual_increase = residual_variance_ratio - previous_residual_variance_ratio

            # 条件をチェック
            if residual_variance_ratio >= threshold or residual_increase < tolerance:
                optimal_n_components = n_components
                nmf_model = nmf
                break

            # 前回の残差割合を更新
            previous_residual_variance_ratio = residual_variance_ratio

        # 最適な基底ベクトル数で再計算
        if nmf_model is not None:
            C = nmf_model.fit_transform(X)
            W = nmf_model.components_
            WC = np.dot(C, W)
            final_residual = np.sum((X - WC) ** 2)
        else:
            final_residual = residual
            nmf = nmf

        return final_residual, nmf
    #################################add 2024/07/11###############################
    #最終のVAFの値しか返さないから、下のvaf_logが使い物になっていない。plot_vaf()も無理

    def f_nmf(self, n_components = None, max_iter=20):
        old_loss, old_nmf = self._culc_loss(X = self.X, n_components=n_components)
        for i in range(max_iter - 1):
            loss, nmf = self._culc_loss(X = self.X, n_components=n_components)
            if loss < old_loss:
                old_loss = loss
                old_nmf = nmf
        return old_nmf

    def _culc_vaf(self, nmf):
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
        #ax.set_title(self.subject)
        ax.set_xticks(np.arange(0,self.max_n_components),np.arange(1, self.max_n_components+1))

    def plot_W(self, n, ax = None):
        if not ax:
            fig, ax = plt.subplots()
        ax.bar(np.arange(len(self.best_syn["W"][n-1])), self.best_syn["W"][n-1], color = f"C{n-1}",
               tick_label = self.label)
        ax.tick_params(axis='x', labelrotation=0)

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
        #plt.suptitle(self.subject, y = 0.99)
        fig.tight_layout()
