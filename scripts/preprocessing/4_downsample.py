# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: May 28, 2024
# Description: The script downsamples the interpolated pupil data to 60 Hz using [](Murphy et al. 2014)


import numpy as np
import pandas as pd
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# Set data directories
mat_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/pupil/3_processed/3_interpolated')
ts_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/timestamps')

# Set save directory
save_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/pupil/3_processed/4_downsampled')

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Set range of subjects
subj_ids = range(1002, 1003)


# Define butter lowpass filter
def butter_lowpass_filter(data, cutoff, nyq, order):
    '''
    Takes in data of certain sampling freq. and applies butter lowpass filter to downsample 
    data to indicated cuttoff freq.

    Inputs:

    Outputs:


    '''
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Define other iterative downsampling method

def iterative_filter(data, f_sample, cutoff):
    '''
    Takes in data of certain sampling freq. and downsamples by selecting and keeping
    every f_sample/cuttoff sample from the data.

    Inputs:

    Outputs:

    '''

    # Calculate the downsampling factor
    downsample_factor = f_sample // cutoff

    # Downsample the data
    downsampled_pupil_data = data[::downsample_factor]

    return downsampled_pupil_data


# define freq. parameters
f_sample = 500 
f_cutoff = 60
order = 2
nyq = 0.5 * f_sample

# iterate over subjects
for sub in subj_ids:

    # fetch data
    mat = sio.loadmat(os.path.join(mat_path, str(sub) + "_interpolated_ET.mat"))
    pupilSize = mat['pupilInterpolated'].flatten()
    time = mat['time'].flatten()

    n = len(pupilSize)

    # approximate sin wave
    #sig = np.sin(1.2*2*np.pi*pupilSize)
    noise = 1.5*np.cos(9*2*np.pi*pupilSize) + 0.5*np.sin(12.0*2*np.pi*pupilSize)
    data = pupilSize + noise

    y = butter_lowpass_filter(pupilSize, f_cutoff, f_sample, order)

    # plot

    #T = int(n/f_sample)
    #t = np.linspace(0, T, n, endpoint=False)

    t1 = range(len(pupilSize))
    t2 = range(len(y))

    assert t1 == t2
    
    plt.subplot(2, 1, 2)
    plt.plot(t1, pupilSize, 'b-', label='data')
    plt.plot(t1, y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()
    plt.show()


    

