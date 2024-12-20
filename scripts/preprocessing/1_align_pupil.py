# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: December 16, 2024
# Description: The script aligns pupil data to stimulus presentation and excludes non-encoding data

import numpy as np
import pandas as pd
import os
import math
import mat73 # to load .mat files in MATLAB v7.3

# ------------------ Hardcoded parameters ------------------ #
os.chdir('/Users/jadyn/repo/paranoia/scripts/preprocessing')
_THISDIR = os.getcwd()
MAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/2_mat'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/1_aligned'))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

SUBJ_IDS = range(1002, 1037)
SAMPLING_RATE = 500 # Hz
# PUPIL_INFO = Area (Area or Diameter)

# ------------------ Define functions ------------------ # 
def fetch_mat(mat_path, sub_id):
    """
    Grabs .mat file for a given subject and saves each struct as an array.
    
    Samples (1x1 Struct): contains time, posX, posY, pupilSize, etc.
    Events (1x1 Struct): contains Messages (another Struct), Sblink, Eblink, etc.
        Sblink: time of the start of the blink
        Eblink: time of the start and end of the blink, and blink duration
        Detailed description of the variables: http://sr-research.jp/support/EyeLink%201000%20User%20Manual%201.5.0.pdf
    
    """
    mat = mat73.loadmat(os.path.join(mat_path, str(sub_id), str(sub_id) + "_ET.mat"))
    samples = mat['Samples']
    events = mat['Events']
        
    return samples, events

# ------------------- Main ------------------ #
for sub in SUBJ_IDS:
    
    # Load .mat data
    samples, events = fetch_mat(MAT_PATH, sub)

    # Time stamp of samples
    samples_time = samples['time'] # in milliseconds; samples_time[1] - samples_time[0] = 2 ms
    
    # Pupil size during the entire timecourse
    samples_pupilSize = samples['pupilSize']

    # Event messages
    events_messages_info = events['Messages']['info']
    events_messages_time = events['Messages']['time']

    # Index and timestamp of STORY_START and STORY_END
    story_start_idx = events_messages_info.index('STORY_START')
    story_end_idx = events_messages_info.index('STORY_END')

    story_start_time = events_messages_time[story_start_idx]
    story_end_time = events_messages_time[story_end_idx]

    # Align pupil data to stimulus presentation
    pupil_start_idx = np.where(samples_time == story_start_time)
    pupil_end_idx = np.where(samples_time == story_end_time)

    # If Samples.time and events.Messages.time aren't aligned
    if len(pupil_start_idx[0]) == 0:
        story_start_time = story_start_time - 1
        pupil_start_idx = np.where(samples_time == story_start_time)
    if len(pupil_end_idx[0]) == 0:
        story_end_time = story_end_time + 1
        pupil_end_idx = np.where(samples_time == story_end_time)
        
    # Extract element from array
    pupil_start_idx = pupil_start_idx[0][0]
    pupil_end_idx = pupil_end_idx[0][0]
    
    # New array of samples during stimulus presentation
    pupilSize_encoding = samples_pupilSize[pupil_start_idx:pupil_end_idx]
    
    # Corresponding time stamp of the new array
    encoding_time = samples_time[pupil_start_idx:pupil_end_idx]
    encoding_time_corrected = encoding_time - encoding_time[0]
    
    print(f"Saving subject {sub} ... ")
    
    filename = os.path.join(SAVE_PATH, str(sub) + "_aligned_ET.csv")
    # pd.DataFrame({'pupilSize': pupilSize_encoding, 'time_in_ms': encoding_time, 'time_in_ms_corrected': encoding_time_corrected}).to_csv(filename, index=False)


