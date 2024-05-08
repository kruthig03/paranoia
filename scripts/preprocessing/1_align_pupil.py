# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: May 8, 2024
<<<<<<< Updated upstream
# Description: The script aligns pupil data to stimulus presentation and excludes noisy participants
# A participant is excluded if 25% of the raw samples 
# (1) contain amplitudes of <1.5mm or (2) the change in amplitude is >0.075mm relative to the directly preceding sample (Murphy et al., 2014)
=======
# Description The script aligns pupil data to stimulus presentation and 
# excludes noisy participants.

import numpy as np
import pandas as pd
import os
import scipy.io as sio
import math



def fetch_mat(path, subj):
    """
    Grabs .mat file for a given subject and saves each struct as an array.
    
    Samples (1x1 Struct): contains time, posX, posY, pupilSize, etc.
    Events (1x1 Struct): contains Messages (another Struct), Sblink, Eblink, etc.
        Sblink: time of the start of the blink
        Eblink: time of the start and end of the blink, and blink duration
        Detailed description of the variables: http://sr-research.jp/support/EyeLink%201000%20User%20Manual%201.5.0.pdf
    
    """
    
    mat = sio.loadmat(os.path.join(path, str(subj), str(subj) + "_ET.mat"))
    samples = mat['Samples']
    events = mat['Events']
    
    return samples, events


# Set data directories
_thisDir = os.getcwd()
mat_path = os.path.normpath(os.path.join(_thisDir, '..', '..', '..', 'pupil',  'data', '2_mat'))
ts_path = os.path.normpath(os.path.join(_thisDir, '..', '..', 'data'))

# Set save directory
save_path = os.path.normpath(os.path.join(mat_path, '..', '3_processed', '1_blinks_removed'))
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Set range of subjects
subj_ids = range(1002, 1031)

# Align pupil data to stimulus

for sub in subj_ids:

    # get data files
    timestamps = pd.read_csv(os.path.join(ts_path, str(sub) + "_paranoia_timestamps.csv"))
    samples, events = fetch_mat(mat_path, sub)

     # Time stamp of samples
    time = samples['time'][0,0]
    
    # Pupil size during the entire timecourse
    pupilSize = samples['pupilSize'][0,0]

    # story start and end
    story_start = timestamps["storyStart"][0]
    story_end = timestamps["storyEnd"][0]

    ## first, create sample array that is aligned to stimulus onset and offset
    
    # get start and end indexes
    f_sample = int(500) # Sampling frequency/rate(Hz)
    start_idx = int(math.floor(story_start * f_sample))
    end_idx = int(math.ceil(story_start * f_sample))

    # new array of only samples during stimulus presentation
    pupilSize_encoding = pupilSize[start_idx, end_idx]

    # number of samples
    nSample_orig = len(pupilSize)
    nSample = len(pupilSize_encoding)

    # create array to keep track of noisy data
    noise = np.array([])

    ## second, filter out samples with excess noise


        
    # Save data mat file if no excessive noise found

    if len(noise) / nSample >= 0.25:
        continue
    else:
        filename = os.path.join(save_path, str(sub) + "SAVEFILE HERE.mat")
>>>>>>> Stashed changes
