# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: July 31, 2024
# Description: This script takes in pupil data and averages by story event

import numpy as np
import pandas as pd
import os
import scipy.io as sio
import math
import importlib

# Set data directories
mat_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/pupil/3_processed/5_last_interp')
ts_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia')

# Set save directory
save_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/pupil/3_processed/7_avg_by_event')

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Set range of subjects
subj_ids = range(1033, 1034)

#Load timestamps
event_ts = pd.read_excel(os.path.join(ts_path, "paranoia_events.xlsx"), engine='openpyxl')
num_events = event_ts.shape[0]

for sub in subj_ids:
    
    # fetch data
    mat = sio.loadmat(os.path.join(mat_path, str(sub) + "_final_interp_ET.mat"))
    pupilSize = mat['pupilFinal'].flatten()


    TR_onset = event_ts['TR_onset']
    TR_offset = event_ts['TR_offset']


    averaged_data = np.array([])

    for event in range(num_events):
        
        # get TR timestamps for each event
        tr_1 = TR_onset[event]
        tr_2 = TR_offset[event]

        # get pupil data for event
        pupil_event = pupilSize[tr_1:tr_2]
        avg_pupil_event = np.average(pupil_event)

        # append to array
        averaged_data = np.append(avg_pupil_event, averaged_data)
    
    filename = os.path.join(save_path, str(sub) + "_avg_event_ET.mat")
    sio.savemat(filename, {'pupilByEvent': averaged_data})
    