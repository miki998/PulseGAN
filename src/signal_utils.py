# Pre-process & Post-process
import numpy as np
import scipy.fftpack as fftpack
from scipy import signal, sparse
from scipy.signal import butter, lfilter, filtfilt, freqz,welch


def normalize(array):
    m = np.mean(array)
    s = np.std(array)
    if s == 0:
        return array - m
    else:
        return (array - m)/s

# https://github.com/phuselab/pyVHR/blob/758364e92dab384fe49be975eba380d157da21a7/pyVHR
def BPfilter(x, minHz, maxHz, fs, order=6):
    """
    desc: filtering out frequency that is in the desired interval
    
    args: 
        - x::[array<float>]
            signal
        - minHz::[float]
        - maxHz::[float]
        - fs::[int]
            sampling rate
    ret:  
    
    """
    """Band Pass filter (using BPM band)"""

    #nyq = fs * 0.5
    #low = minHz/nyq
    #high = maxHz/nyq

    #print(low, high)
    b, a = butter(order, Wn=[minHz, maxHz], fs=fs, btype='bandpass')
    #TODO verificare filtfilt o lfilter
    #y = lfilter(b, a, x)
    y = filtfilt(b, a, x)

    return y

def zeroMeanSTDnorm(x):
    """
    desc: mean/std normalizing
    
    args:
        - x::[array<float>]
            signal
    ret:
        - y::[array<float>]
    
    """
    # -- normalization along rows (1-3 channels)
    mx = x.mean(axis=1).reshape(-1,1)
    sx = x.std(axis=1).reshape(-1,1)
    y = (x - mx) / sx
    return y

def detrend(X, detLambda=10):
    """
    desc: get rid of a randomness trend might deal with sudden increase trend coming from head movements
    
    args:
        - X::[array<float>]
            signal
    ret:
        - detrendedX::[array<float>]
            detrended signal
    """
    # Smoothness prior approach as in the paper appendix:
    # "An advanced detrending method with application to HRV analysis"
    # by Tarvainen, Ranta-aho and Karjaalainen
    t = X.shape[0]
    l = t/detLambda #lambda
    I = np.identity(t)
    D2 = sparse.diags([1, -2, 1], [0,1,2],shape=(t-2,t)).toarray() # this works better than spdiags in python
    detrendedX = (I-np.linalg.inv(I+l**2*(np.transpose(D2).dot(D2)))).dot(X)
    return detrendedX

# Temporal bandpass filter with Fast-Fourier Transform
def fft_filter(video, freq_min, freq_max, fps):
    	
    fft = fftpack.fft(video, axis=0)
    frequencies = fftpack.fftfreq(video.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - freq_min)).argmin()
    bound_high = (np.abs(frequencies - freq_max)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff = fftpack.ifft(fft, axis=0)
    result = np.abs(iff)
    result *= 100  # Amplification factor

    return result, fft, frequencies