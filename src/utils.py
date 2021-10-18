"""

@author: Chun Hei Michael Chan
@copyright: Copyright Logitech
@credits: [Chun Hei Michael Chan]
@maintainer: Chun Hei Michael Chan
@email: cchan5@logitech.com

"""

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
from src.process import *
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
    
def record_video(path='./videos/michael1.avi', cam_idx=2):
    """
    desc:
        from a webcam, capture live images to then write/save
    args:
        - path::[str]
            path to where we save the live capture
        - cam_idx::[int] 
            camera index
    ret:
        - None
    """
    video = cv2.VideoCapture(cam_idx)

    if (video.isOpened() == False): 
        print("Error reading video file")

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    size = (frame_width, frame_height)

    result = cv2.VideoWriter(path, 
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             28, size)

    while(True):
        ret, frame = video.read()

        if ret == True: 
            result.write(frame)
            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                break

        else:
            break

    video.release()
    result.release()

    cv2.destroyAllWindows()

    print("The video was successfully saved")
    
def test_camera(cam_idx):
    """
    desc: camera testing with live capture device
        
    args:
        - cam_idx::[int]
            camera index
    
    ret:
        - None
    """
    
    vid = cv2.VideoCapture(cam_idx)

    while(True):
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

def video2img(video_path,img_root):
    """
    desc: turn videos into saved images
        
    args:
        - video_path::[str]
            string that indicates path to video
        - img_root::[str]
            root folder where to store the images
    ret:
    
    """
    person_id = video_path.split('/')[-2]
    vid = cv2.VideoCapture(video_path)
    
    outpath = img_root + '/' + person_id + '/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    idx = 0
    while(True):

        ret, frame = vid.read()
        if not ret:
            break
        cv2.imwrite(outpath+str(idx)+'.jpg', frame)
        
        idx += 1
    vid.release()
    
# structural reorganising
def reorder_folder(strings):
    """
    desc: reorder parse order of os.listdir to normal integers (name of files inside folder are numbers ending with `.`)
    
    args:
        - strings::[array<str>]
            list of strings to reorder
    ret:
        - ordered::[array<str>]
            ordered list of strings
    """
    ints = [int(s.split('.')[0]) for s in strings]
    ordered = np.array(strings)[np.argsort(ints)]
    
    return ordered

def read_xmp(path): 
    """
    desc: read xmp ground truth for UBFC-RPPG dataset 
        
    args: 
        - path::[str]
            path to the .xmp file
    ret: 
        - ret::[dict]
            extracted info from .xmp file
    """
    ret = {
        'timestamp': [],
        'heart_rate': [],
        'sig_length': []
    }
    
    with open(path,'r') as f:
        
        lines = f.readlines()
        for line in lines:
            t, hr, s, _ = line.strip().split(',')
            ret['timestamp'].append(int(t))
            ret['heart_rate'].append(int(hr))
            ret['sig_length'].append(int(s))

    return ret

def read_txt(path):
    """
    desc: read text ground truth for UBFC-RPPG dataset 

    args:
        - path::[str]
            path to the .txt file

    ret: 
        - store::[dict]
            extracted info from .txt file
    """
    store = {
        'name':[],
        'rppg_signal':[],
        'heart_rate':[],
        'timestamp':[]
    }

    with open(path, 'r') as f:
        t = f.readlines()
        store['rppg_signal'] += t[0].split()
        store['name'] += (len(t[0].split()) * [path])
        store['heart_rate'] += t[1].split()
        store['timestamp'] += t[2].split()
            
    
    return store


# background post-process of predictions
def interpolate1d(small_interval,large_interval):
    """
    desc: interpolate an interval to another one to obtain values that match 1 to 1

    args:
        - small_interval::[array<float>]
            small interval to you match to
        - large_interval::[array<float>]
            large interval that you reduce
    ret:
        - interpolated::[array<float>]
            array that you was originally large interval

    """

    window =  len(large_interval)//len(small_interval)
    interpolated = []
    for i in range(len(small_interval)):
        
        interpolated.append(np.mean(large_interval[i*window:(i+1)*window]))
        
    return interpolated

def risk_interval(sqi, cutoff):
    """
    desc:

    args:
        - sqi::[array<float>]
            sqi proxy, high sqi would mean low sqi
        - cutoff::[float]
            cutoff to decide of a sqi beloging to risk or not 

    ret:
        - rintervals::[list]
            pairs of value where sqi is low

    """
    intervals = np.where((sqi > cutoff) == True)[0] 

    nums = sorted(set(intervals))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    rinvtervals = list(zip(edges, edges))
    return rinvtervals

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

def get_mfpe(samp_time, start, end):
    """
    desc:
        get sample rate according to interval between timestamp
    args:
        - samp_time::[array<float>]
            intervals
        - start/end::[int]
            start and end time for interval
    ret:
        - fpe::[float]
            sample rate
    """


    fpe = np.mean([samp_time.iloc[i+1] - samp_time.iloc[i] 
                   for i in range(start,end)])
    fpe = 1/fpe
    return fpe

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

def merge_obs_ecg(obs_df, ecg_df):
    """
    desc:
        merge obs and ecg dataframes to match the timestamps
    args:
        - obs_df::[dataframe]
        - ecg_df::[dataframe]
    ret:
        - merged_df[dataframe]
    """

    merged = {
        'Frame_nb': [],
        'timestamp': [],
        'Estimated_hr': []

    }

    idx_ecg = 0
    idx_obs = 0
    current_signal = []
    while (idx_obs < obs_df.shape[0]) and (idx_ecg < ecg_df.shape[0]):


        timestamp = obs_df.iloc[idx_obs]['LSL timestamp']
        framenb = obs_df.iloc[idx_obs]['Frame number']
        sig = ecg_df.iloc[idx_ecg]['Estimated_hr']

        if ecg_df.iloc[idx_ecg]['LSL timestamp'] > timestamp:
            # update
            merged['Frame_nb'].append(framenb)
            merged['timestamp'].append(timestamp)
            if len(current_signal) == 0:
                merged['Estimated_hr'].append(None)
            else:
                merged['Estimated_hr'].append(np.mean(current_signal))
            current_signal = []
            idx_obs += 1
        else:
            current_signal.append(sig)
            idx_ecg += 1


    # update once more for last stack
    merged_df = pd.DataFrame.from_dict(merged)
    
    return merged_df