# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: June 29, 2024
# Description: This script takes in subjects' preprocessed pupil data and calculates the one-to-average ISC, testing significance
# using bootstrapping


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


# Set data directory
_thisDir = os.getcwd()
path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/pupil/3_processed/5_last_interp')

# Set save directory
save_path = os.path.normpath('/Users/kruthigollapudi/src/paranoia/data/pupil/3_processed/6_isc')


## FUNCTIONS HERE

def isc_loo(df, thisSub_idx):
    """
    One-to-average ISC

    Parameters:
        df (pd.DataFrame): dataframe of pupilSize by subject
        thisSub_idx (int): index of this subject's data column
    
    Returns:
        corr (np.float): one-to-average ISC for a given subject

    """

    i = thisSub_idx
    thisSubj = df.iloc[:,i]
    everyoneElse = df.drop(df.columns[[i]], axis=1)
    
    # Average everyone else's data
    avg = everyoneElse.mean(axis=1)
    
    # Create a temporary df to store thisSubj and avg
    df_temp = pd.DataFrame({'thisSubj': thisSubj, 'avg': avg})
    
    # Correlate this Subject's data with the average of everyone else's
    corr = df_temp.corr(method='pearson').iloc[0,1]
    
    return corr

def phase_randomize(data, random_state=None):
    """Perform phase randomization on time-series signal (from nltools.stats)

    This procedure preserves the power spectrum/autocorrelation,
    but destroys any nonlinear behavior. Based on the algorithm
    described in:

    Theiler, J., Galdrikian, B., Longtin, A., Eubank, S., & Farmer, J. D. (1991).
    Testing for nonlinearity in time series: the method of surrogate data
    (No. LA-UR-91-3343; CONF-9108181-1). Los Alamos National Lab., NM (United States).

    Lancaster, G., Iatsenko, D., Pidde, A., Ticcinelli, V., & Stefanovska, A. (2018).
    Surrogate data for hypothesis testing of physical systems. Physics Reports, 748, 1-60.

    1. Calculate the Fourier transform ftx of the original signal xn.
    2. Generate a vector of random phases in the range[0, 2π]) with
       length L/2,where L is the length of the time series.
    3. As the Fourier transform is symmetrical, to create the new phase
       randomized vector ftr , multiply the first half of ftx (i.e.the half
       corresponding to the positive frequencies) by exp(iφr) to create the
       first half of ftr.The remainder of ftr is then the horizontally flipped
       complex conjugate of the first half.
    4. Finally, the inverse Fourier transform of ftr gives the FT surrogate.

    Args:

        data: (np.array) data (can be 1d or 2d, time by features)
        random_state: (int, None, or np.random.RandomState) Initial random seed (default: None)

    Returns:

        shifted_data: (np.array) phase randomized data
    """
    random_state = check_random_state(random_state)

    data = np.array(data)
    fft_data = fft(data, axis=0)

    if data.shape[0] % 2 == 0:
        pos_freq = np.arange(1, data.shape[0] // 2)
        neg_freq = np.arange(data.shape[0] - 1, data.shape[0] // 2, -1)
    else:
        pos_freq = np.arange(1, (data.shape[0] - 1) // 2 + 1)
        neg_freq = np.arange(data.shape[0] - 1, (data.shape[0] - 1) // 2, -1)

    if len(data.shape) == 1:
        phase_shifts = random_state.uniform(0, 2 * np.pi, size=(len(pos_freq)))
        fft_data[pos_freq] *= np.exp(1j * phase_shifts)
        fft_data[neg_freq] *= np.exp(-1j * phase_shifts)
    else:
        phase_shifts = random_state.uniform(
            0, 2 * np.pi, size=(len(pos_freq), data.shape[1])
        )
        fft_data[pos_freq, :] *= np.exp(1j * phase_shifts)
        fft_data[neg_freq, :] *= np.exp(-1j * phase_shifts)
        
    return np.real(ifft(fft_data, axis=0))


# Define range of subject ids
subj_ids = range(1002, 1030)

# Iterate through subjects
list_pupil = []

for sub in subj_ids:
    
    filename = os.path.join(path, str(sub) + "_final_interp_ET.mat")
   
    try:
        
        mat = sio.loadmat(filename)
        pupilSize = mat['pupilFinal'].flatten()
        df = pd.DataFrame(
            {'pupilDownsampled': pupilSize}
        )
        
    except FileNotFoundError: # Skip subject if file doesn't exist
        continue

    # Create 2Hz dataframe
    colName = f'pupilDownsampled'
    subName = f'{sub}'

    # Extract data from subject
    thisSubj = df[colName]
    thisSubj.name = subName # Rename

    # Append to list
    list_pupil.append(thisSubj)

    pupilSize_by_sub = pd.concat([pd.Series(x) for x in list_pupil], axis=1)    

total_nans = pd.DataFrame(pupilSize_by_sub).isnull().sum()
print(total_nans)

# Initialize arrays
isc_loo_values = {}
nSub = pupilSize_by_sub.shape[1]

for i in range(nSub):
    
    # Save the one-to-average correlation for each subject
    isc_loo_values[pupilSize_by_sub.columns[i]] = isc_loo(pupilSize_by_sub, i)

# One-to-average ISC
isc_df = pd.DataFrame([isc_loo_values], index=None)

# Fisher-z transform, average, inverse fisher-z transform
isc_loo_z = np.arctanh(list(isc_loo_values.values()))
true_mean_z = np.nanmean(isc_loo_z)
true_mean_r = np.tanh(true_mean_z) # True one-to-average ISC

print('True mean r value: ', true_mean_r)


# Bootstrapping HERE

# Permute bootstrapped samples
nIt = 5000
boot_ISC_mean = np.full([nIt,1], np.nan)

for iteration in range(nIt):

    if iteration % 100 == 0:
        print('Iteration =', iteration)

    for sub_idx in range(nSub):

        # This subject's time series data
        thisSubj = pupilSize_by_sub.iloc[:, sub_idx]

        # Interpolate all NaNs for phase randomization
        # This also pads edge cases (head/tail NaNs) with first/last occurring value
        nSample = len(thisSubj)
        x = np.arange(0, len(thisSubj), 1) # x-coordinate of query points
        nan_indices = np.isnan(thisSubj) 
        thisSubj_interp = np.interp(x, x[~nan_indices], thisSubj[~nan_indices])

        # Everyone else's time series data
        everyoneElse = pupilSize_by_sub.drop(pupilSize_by_sub.columns[[sub_idx]], axis=1)

        # Phase randomize this subject's (interpolated) data
        thisSubj_rand = phase_randomize(thisSubj_interp)

        # Average everyone else's data
        avg = everyoneElse.mean(axis=1, skipna=True)

        # Create a temporary df to store thisSubj_rand and avg
        df_temp = pd.DataFrame({'thisSubj_rand': thisSubj_rand, 'avg': avg})

        # Correlate this subject's phase randomized data with the average of everyone else's
        boot_ISC_loo = df_temp.corr(method='pearson').iloc[0,1]

        boot_ISC_mean[iteration] = np.tanh(np.nanmean(np.arctanh(boot_ISC_loo)))

# Difference between actual and bootstrapped means
boot_ISC_demean = boot_ISC_mean - true_mean_r

# p-value
p_value = np.mean(true_mean_r < boot_ISC_demean) + 1 / nIt
print('P-value: ', p_value)
print(f'ISC: {true_mean_r}, p-value: {p_value}')

isc_final_df = pd.DataFrame(
    {'P-value': p_value,
    'True-Mean-R': true_mean_r},
    index=[0]
)

filename = os.path.join(save_path, str(sub) + "_isc_values.csv")
isc_final_df.to_csv(filename)




    