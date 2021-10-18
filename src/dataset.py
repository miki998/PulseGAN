"""

@author: Chun Hei Michael Chan
@copyright: Copyright Logitech
@credits: [Chun Hei Michael Chan]
@maintainer: Chun Hei Michael Chan
@email: cchan5@logitech.com

"""

import os
import numpy as np
import pandas as pd

# own libs
from src.utils import * 
from src.process import *

class Dataset:
    """
    desc: 
        this class stores paths to the targetted videos and its respective ground-truth files
        to make it easier to do hr-estimation and then matching the ground truth to it

        use a dataframe to store all data, and get processed signals i.e heart beat 

    
    """
    def __init__(self, name):
        self.name = name.strip().lower()
        self.path = None

        self.info = None
        # one day used for motion track and other info
        self.extra = {}


    def load(self,path):
        self.path = path


        self.info = {
            'id': [],
            'timestamp': [],
            'heart_rate': [],
            'frame': []

        }
        if self.name == 'ubfc_rppg1':
            xmps = [file for file in os.listdir(self.path) if file.endswith('.xmp')]
            for xmp in xmps:
                store = read_xmp(self.path+'/'+xmp)
                self.info['id'] += [xmp[:-4]] * len(store['timestamp'])
                self.info['timestamp'] += list(store['timestamp'])
                self.info['heart_rate'] += list(store['heart_rate'])
                self.info['frame'] += ([0] * len(store['timestamp']))

        elif self.name == 'ubfc_rppg2':
            txts = [file for file in os.listdir(self.path) if file.endswith('.txt')]
            for txt in txts:
                store = read_txt(self.path+'/'+txt)
                self.info['id'] += [txt[:-4]] * len(store['timestamp'])
                self.info['timestamp'] += list(store['timestamp'])
                self.info['heart_rate'] += list(store['heart_rate'])
                self.info['frame'] += ([0] * len(store['timestamp']))

        elif self.name == 'logi_rppg':
            # format needs to be ecg#number.csv and obs#number.csv
            csvs = set([file[-5:] for file in os.listdir(self.path) if file.endswith('.csv')])
            
            for csv in csvs:
                obs = 'obs'+csv
                ecg = 'ecg'+csv
                ID = csv.split('.')[0]

                obs_df = pd.read_csv(self.path+'/'+obs)
                ecg_df = pd.read_csv(self.path+'/'+ecg)
                fpe = get_mfpe(ecg_df['LSL timestamp'],0,ecg_df.shape[0]-1)
                detrended = detrend(ecg_df['Channel1'])
                ecg_df['signal'] = detrended
                
                ecg_processed, step = get_window_hr(ecg_df, fpe, window=3)
                ecg_processed = pd.DataFrame.from_dict(ecg_processed)
                merged_df = merge_obs_ecg(obs_df, ecg_processed)

                self.info['id'] += (list(ID) * len(merged_df['timestamp']))
                self.info['timestamp'] += list(merged_df['timestamp'])
                self.info['heart_rate'] += list(merged_df['Estimated_hr'])
                self.info['frame'] += list(merged_df['Frame_nb'])
        else:
            print('wrong dataset input | please choose between ubfc_rppg1/ubfc_rppg2/logi_rppg')

        self.info = pd.DataFrame.from_dict(self.info)