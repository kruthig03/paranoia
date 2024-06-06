# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: May 8, 2024
# Description: The script aligns pupil data to stimulus presentation and excludes non-encoding data


import numpy as np
import pandas as pd
import os
import scipy.io as sio
import mat73
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

    mat = mat73.loadmat(os.path.join(path, str(subj), str(subj) + "_ET.mat"))
    samples = mat['Samples']
    events = mat['Events']
    
    return samples, events


# Set data directories
#_thisDir = os.getcwd()
mat_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/pupil/2_mat')
ts_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/timestamps')

# Set save directory
save_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/pupil/3_processed/1_aligned')


if not os.path.exists(save_path):
    os.makedirs(save_path)

# Set range of subjects
subj_ids = range(1002, 1016)

# Align pupil data to stimulus

for sub in subj_ids:

    # get data files
    timestamps = pd.read_csv(os.path.join(ts_path, str(sub) + "_paranoia_timestamps.csv"))
    samples, events = fetch_mat(mat_path, sub)

    # Time stamp of samples
    time = samples['time']
    
    # Pupil size during the entire timecourse
    pupilSize = samples['pupilSize']

    # story start and end
    story_start = timestamps["storyStart"][0]
    story_end = timestamps["storyEnd"][0]
    
    # get start and end indexes
    f_sample = int(500) # Sampling frequency/rate(Hz)
    start_idx = int(math.floor(story_start * f_sample))
    end_idx = int(math.ceil(story_end * f_sample))

    # new array of only samples during stimulus presentation
    pupilSize_encoding = pupilSize[start_idx: end_idx]
    pupil_time = time[start_idx: end_idx]

    sample_num = len(pupilSize_encoding)
    stim_length = sample_num / 30000

    # number of samples
    #nSample_orig = len(pupilSize)
    #nSample = len(pupilSize_encoding)
    
    filename = os.path.join(save_path, str(sub) + "_aligned_ET.mat")
    sio.savemat(filename, {'pupilEncoding':pupilSize_encoding, 'time': pupil_time, 'sample_num': sample_num, 'stim_min': stim_length})

