# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: May 8, 2024
# Description: The script aligns pupil data to stimulus presentation and excludes noisy participants
# A participant is excluded if 25% of the raw samples 
# (1) contain amplitudes of <1.5mm or (2) the change in amplitude is >0.075mm relative to the directly preceding sample (Murphy et al., 2014)
