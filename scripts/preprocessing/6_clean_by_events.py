# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: January 6, 2025
# Description: This script averages pupil timecourse within events

import os
import glob
import scipy.io as sio
import numpy as np
from numpy.fft import fft, ifft, fftfreq
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from sklearn.utils import check_random_state
from numpy import interp

# ------------------ Hardcoded parameters ------------------ #
os.chdir('/Users/jadyn/repo/paranoia/scripts/preprocessing')
_THISDIR = os.getcwd()
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/5_timelocked'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/6_events'))
EVENTS_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/timestamps'))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    
SUBJ_IDS = range(1002, 1037)
STORY_LENGTH = 1302 # Length of the story in seconds

# ------------------ Main ------------------ #

# Load event timestamps
timestamps = pd.read_csv(os.path.join(EVENTS_PATH, 'paranoia_events_TR.csv'), header=0)
event_start_TR = np.array(timestamps['TR_onset'])
event_end_TR = np.array(timestamps['TR_offset'])

# Create empty dataframe to save new data
dat_events = pd.DataFrame(columns=['event', 'pupilSize'])

for sub in SUBJ_IDS:
    
    # Load timelocked data
    file_path = os.path.join(DAT_PATH, str(sub) + "_timelocked.csv")
    if not os.path.exists(file_path):
        continue
    dat = pd.read_csv(file_path)
    
    TR = np.array(dat['TR'])
    pupilSize = np.array(dat['pupilSize'])

    # Pad data with NaNs if subject's data is shorter 
    if len(pupilSize) < STORY_LENGTH:
        pupilSize = np.append(pupilSize, np.repeat(np.nan, STORY_LENGTH - len(pupilSize)))
    
    # Standardize pupil data using z-score
    pupilSize_z = (pupilSize - np.nanmean(pupilSize)) / np.nanstd(pupilSize)
    
    # Average pupil size within each event
    for i in range(len(event_start_TR)):
        event_start = event_start_TR[i]
        event_end = event_end_TR[i]
        thisEvent_pupil = pupilSize_z[event_start:event_end]
        thisEvent_pupil = thisEvent_pupil[~np.isnan(thisEvent_pupil)] # Remove NaNs, if there are any
        thisEvent_pupil_avg = np.mean(thisEvent_pupil)
        
        # Add average pupil size to dataframe
        dat_events.loc[i] = [i+1, thisEvent_pupil_avg]
        
    # Save dataframe
    dat_events.to_csv(os.path.join(SAVE_PATH, str(sub) + "_events.csv"), index=False)
