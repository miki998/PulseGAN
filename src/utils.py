# general libs
import os
import numpy as np
import pandas as pd

# vision libs
import cv2

# plotting libs
import matplotlib.pyplot as plt

# signal process libs
from scipy.signal import hamming

# import own libs
from src.signal_utils import *
from src.hr_estimation import *


# image related
def recthull(points):
    """
    desc: from some points, get the bbox including all 
            pixels from the delimiting points
    
    args: 
        - points::[array <array<int> >]
    
    ret: 
        - x,y,w,h::[tuple<int>]
    """
    
    x = min(points[:,0])
    y = min(points[:,1])
    w = max(points[:,0]) - x
    h = max(points[:,1]) - y
    
    return x,y,w,h



def img2uint8(img):
    # convert to 8 bit if needed
    if img.dtype is np.dtype(np.uint16):
        if np.max(img[:]) < 256:
            scale = 255.  # 8 bit stored as 16 bit...
        elif np.max(img[:]) < 4096:
            scale = 4095.  # 12 bit
        else:
            scale = 65535.  # 16 bit
        img = cv2.convertScaleAbs(img, alpha=(225. / scale))
        
    return img


def color_mapping(img,scheme='HSV'):
    """
    desc: map general `RGB` to other coloring scheme e.g YUV, HSV
    
    args:
        - img::[array<array<int> >]
        - scheme::[str]
        
    ret:
        - mapped::[array<array<int> >]
    """
    
    if scheme.strip().upper() == 'HSV':
        mapped = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    elif scheme.strip().upper() == 'YUV':
        mapped = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        
    elif scheme.strip().upper() == 'YCRCB':
        mapped = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        
    else:
        return None
    
    return mapped

def plot_channelcomp(img):
    """
    desc: plot distribution of values in each channel
        to check when doing color filtering for segmentation
    args:
        - img::[array<int>]
    ret:
        - None 
    
    """
    N = img.shape[-1]
    fig, ax = plt.subplots(1,N,figsize=(10,5))
    for i in range(N):
        ax[i].hist(img[:,:,i])
    plt.suptitle('Channel distribution of image')
    plt.show()


def overlap_add(signal, wsize=3):
    """
    desc: smoothen a signal by adding to part of itself to other intervals

    args:
        - signal::[array<float>]
            signal to be overlapp added

    ret:
        - overlapped::[array<float>]
    
    """
    
    overlapped = np.concatenate([np.convolve(signal, np.ones(wsize)/wsize, mode='valid'),signal[-(wsize-1):]])
    return overlapped


def get_window_hr(signal_df,fpe,window=2):
    """
    desc: estimate-hr from a window interval belonging to stream of signal dataframe
    
    args:
        - signal_df::[dataframe]
        - fpe::[float]
        - windows::[int]

    ret:
        - ret::[dict]
            described below
        - step::[int]
            window length
    """

    ret = {
        'Channel1': [],
        'LSL timestamp': [],
        'signal': [],
        'Estimated_hr': []
        
    }
    fpe = int(fpe)
    step = int(window * fpe)
    for idx in range(0,signal_df.shape[0],step):
        
        if idx + step >= (signal_df.shape[0]-1): 
            break
            
        window_sig = []
        for i in range(idx, idx+step):
            cha = signal_df.iloc[i]['Channel1']
            sig = signal_df.iloc[i]['signal']
            timestamp = signal_df.iloc[i]['LSL timestamp']
            ret['Channel1'].append(cha)
            ret['signal'].append(sig)
            ret['LSL timestamp'].append(timestamp)
            
            window_sig.append(sig)
            
        # set values
        window_sig = np.asarray(window_sig)
        window_sig = hamming(len(window_sig)) * window_sig
        hr_sig = get_rfft_hr(window_sig, fpe)
        ret['Estimated_hr'] += [hr_sig] * step
        
    return ret, step