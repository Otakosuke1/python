import numpy as np
import scipy as sp
#import skfda
import pandas as pd
from tools.function import approx


def create_emg_epochs(emg, foot="r"):
    """
    emg: EMG class object
    foot: select foot 'r' or 'l'
    """
    if "r" in foot:
        foot = 1
    elif "l" in foot:
        foot = 2
    else:
        KeyError("foot must be 'r' or 'l'" )
    events = emg.events[emg.events == foot]
    
    dat = emg.emg_matrix
    emg_epochs = gen_epochs(dat, events)
    return emg_epochs

def align_epochs(epochs, n=100):
    """
    epochs: list of epochs
    n: number of points to align
    """
    method = "linear"
    aln_epochs = np.zeros([len(epochs), n, epochs[0].shape[1]])
    for i, epoch in enumerate(epochs):
        for j, mus in enumerate(epoch):
            if i == len(epochs):
                break
            aln_epochs[i,:,j] = approx(epoch[mus], method, n)
    return aln_epochs

def gen_epochs(array, events):
    """
    array: 2D array of data(time series x channel)
    events: Series of events(must be the same length as time series of array)
    """
    events = events.index
    df = pd.DataFrame(array)
    # if len(df) != len(events):
    #     df = df.T
    #     if len(df) != len(events):
    #         ValueError(f'Array length({max(df.shape)}) does not match events length ({len(events)})')
    epochs = []
    for ev in range(len(events)):
        if ev+1==len(events):
            break
        else:
            epochs.append(df.iloc[events[ev]:events[ev+1]])
    return epochs
