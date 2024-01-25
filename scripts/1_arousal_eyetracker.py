# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: Jan 25, 2024
# Description: Task script adopted from Kannon Bhattacharyya and Zishan Su.
# The script continuously records eyetracking data while showing visual 
# stimulus and recording verbal responses.

#--------------------------------- Import ---------------------------------#
from __future__ import absolute_import, division, print_function
import psychopy
import pandas as pd
import os
import time
import psychtoolbox as ptb
from psychopy import visual, core, event, iohub, data, gui, logging, sound
from psychopy.iohub import launchHubServer
from psychopy.iohub.util import hideWindow, showWindow
from psychopy.sound import Microphone
from psychopy.hardware.keyboard import Keyboard
from psychopy.constants import PLAYING, PAUSED
from pylsl import StreamInfo, StreamOutlet

paranoia_length = 1320 # Length of stim in seconds

# psychopy.useVersion('2022.2.2')

#--------------------------------- Toggle ---------------------------------#

# =========================================================
# Toggle tracker: 
# 1=Eyelink, 2=Mouse (with calibration), 0=Skip calibration
# =========================================================
ET = 1

# ====================================
# Toggle kill switch:
# 0=No kill switch, 1=Yes kill switch
# ====================================
KS = 1

# =============================
# Toggle voice recording:
# 0=Don't record, 1=Record
# =============================
REC = 1

#------------------------------- Initialize -------------------------------#

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = 'v2022.2.5'
expName = 'paranoia'
expInfo = {'participant': ''}

# Get participant ID via dialog
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit() # If user pressed cancel
expInfo['date'] = data.getDateStr() # Add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Setup the Window
win = visual.Window(
    size = [1920, 1080], fullscr=True,
    # screen=1,
    units='pix',
    allowGUI=False,
    colorSpace='rgb255',
    monitor='55w_60dist', 
    color="gray"
)
win.mouseVisible = False

# Setup central circle
crossCentralBlack = visual.TextStim(
    win=win, 
    name='crossCentralBlack',
    text='+',
    font='Arial',
    pos=[0, 0],
    height=46 ,
    bold=True,
    color='black',
    units='pix',
    colorSpace='rgb'
)

# ==================
# Data save settings
# ==================

# Path to save data
path = os.path.join(_thisDir, '..', 'data')

# Data file name
filename = os.path.join(path, '%s_%s_%s' % (expInfo['participant'], expName, expInfo['date']))

# Save a log file for detail verbose info
logFile = logging.LogFile(filename + '.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)

# Store participant ID and timestamps 
df = {'id': [expInfo['participant']]}
dfTimeStamps = pd.DataFrame(df)  

# Save df to csv
dfTimeStamps.to_csv(filename + '_timestamps.csv', index=False)

#------------------------- Experiment Settings ----------------------------#

# =================
# Microphone setup
# =================

if REC == 1: 
    recordingDevicesList = Microphone.getDevices()
    device=recordingDevicesList[1] # Double check that this corresponds to the external mic!
    mic = Microphone(streamBufferSecs=3000.0,
                        sampleRateHz=48000,
                        device=recordingDevicesList[1],
                        channels=1,
                        maxRecordingSize=300000,
                        audioRunMode=0
    )


# ================
# Eyetracker setup
# ================

if ET == 1:
    TRACKER = 'eyelink'
    eyetracker_config = dict(name='tracker')
    devices_config = {}
    eyetracker_config['model_name'] = 'EYELINK 1000 DESKTOP'
    eyetracker_config['runtime_settings'] = dict(sampling_rate=500, track_eyes='RIGHT')
    devices_config['eyetracker.hw.sr_research.eyelink.EyeTracker'] = eyetracker_config
    eyetracker_config['calibration'] = dict(type='FIVE_POINTS')
    
    win.setMouseVisible(False)

io = launchHubServer(window=win, **devices_config)

# ===========
# iohub setup
# ===========
keyboard = io.getDevice('keyboard')
kb = Keyboard(waitForStart=True) # JP: clock?
tracker = io.getDevice('tracker')

# =====================
# Quit experiment setup
# =====================

keys = kb.getKeys(['p'])
if "p" in keys:
    core.quit()

#------------------------------- Functions --------------------------------#

# Start ET calibration
def start_calibration():
    
    if ET == 1:
        # Calibration instructions
        calibration = visual.TextStim(
            win=win, 
            text='Please follow the circle on the next screen. \n\n Please wait for the experimenter to start.',
            font='Arial',
            pos=[0, 0], height=36,color='black', units='pix', colorSpace='named',
            wrapWidth=win.size[0] * .9
        )
        
        # Draw text stimulus and wait until "enter" is pressed
        calibration.draw()
        win.flip()
        event.waitKeys(keyList=["return"])
        
        # Print calibration result (if calibration was passed)
        hideWindow(win)
        result = tracker.runSetupProcedure()
        print("Calibration returned: ", result)
        showWindow(win)
    
    # No calibration in debug mode
    elif ET == 0:
        core.wait(0)
    
    return

#----------------------------- Instructions -------------------------------#

# Instruction: Welcome
startInstructions = visual.TextStim(
    win=win, 
    name = 'instrStart',
    text='Welcome! Press ENTER to start the experiment.',
    font='Arial',
    pos=[0, 0], height=36, color='black', units='pix', colorSpace='named',
    wrapWidth=win.size[0] * .9
)

# Instruction: Wait
waitInstructions = visual.TextStim(
    win=win, 
    name = 'instrWait',
    text='Please wait for the experimenter.',
    font='Arial',
    pos=[0, 0], height=36, color='black', units='pix', colorSpace='named',
    wrapWidth=win.size[0] * .9
)

# Instruction: Video intro
paranoiaIntroInstructions = visual.TextStim(
    win=win,
    name='instrVideoIntro',
    text="In this study, you will be listening to a story. \n The story is just over 20 minutes long in total. \n Following the story, you will be asked to recount what you heard. \n\n\n\n\n\nPress ENTER to continue",
    font='Arial',
    pos=[0, 0], height=36, color='black', units='pix', colorSpace='rgb',
    wrapWidth=win.size[0] * 0.9
)

# Instruction: Video ready
paranoiaReadyInstructions = visual.TextStim(
    win=win, 
    name='instrVideoReady',
    text='While listening to the story,\nplease try to keep your head as still as possible and refrain from moving. \nPlease stare at the cross in the center of the screen for the duration of the piece.\n\n\n Press ENTER when you are ready to begin.',
    font='Arial',
    pos=[0, 0], height=36, color='black', units='pix', colorSpace='named',
    wrapWidth=win.size[0] * .9
)

# Instruction: Break
breakInstructions = visual.TextStim(
    win=win,
    name='instrBreak',
    text="You are done the listening portion of the experiment.\n Please feel free to take a short break. \n\n\nWhenever you are ready, press ENTER to continue",
    font='Arial',
    pos=[0, 0], height=36, color='black', units='pix', colorSpace='rgb',
    wrapWidth=win.size[0] * 0.7
)

# Instruction: Record intro
recordIntroInstructions1 = visual.TextStim(
    win=win,
    name='instrRecordIntro1',
    text="Now, we would like you to recount, in your own words, \nthe events of the story in the original order they were experienced in, with as much detail as possible. \n\nSpeak for at least 10 min if possible -- but the longer the better. \nPlease verbally indicate when you are finished by saying, for example, \"I'm done.\" \n\n Press ENTER to continue.",
    font='Arial',
    pos=[0, 0], height=36, color='black', units='pix', colorSpace='rgb',
    wrapWidth=win.size[0] * 0.3
)

recordIntroInstructions2 = visual.TextStim(
    win=win,
    name='instrRecordIntro2',
    text="Completeness and detail are more important than temporal order. \nIf at any point you realized that you missed something, feel free to return to it. \n\n\nPress ENTER to begin audio recording \n\n(The microphone will automatically turn on after you press Enter; please do NOT touch/move the microphone. \nThere will be a black cross on the screen, keep your eyes on it during recording. \nWhen you are finished speaking, press Enter again to stop recording.)",
    font='Arial',
    pos=[0, 0], height=36, color='black', units='pix', colorSpace='rgb',
    wrapWidth=win.size[0] * 0.3
)

# Instruction: Finish
finishInstructions = visual.TextStim(
    win=win,
    name='instrFinish',
    text="Thank you for your participation! \n\n\n\n\n Please let the experimenter know you have finished!",
    font='Arial',
    pos=[0, 0], height=36, color='black', units='pix', colorSpace='rgb',
    wrapWidth=win.size[0] * 0.9
)

#--------------------------- Start Experiment -----------------------------#

# Get stimulus
paranoia = sound.Sound(_thisDir + os.sep + '../stimuli/paranoia.wav', stereo=True)

# Prep for ET calibration
startInstructions.draw()
win.flip()
event.waitKeys(keyList=['return'])

# ==================
# Run ET Calibration
# ==================
if ET == 1:
    start_calibration()

# Show instruction: Wait
waitInstructions.draw()
win.flip()
event.waitKeys(keyList=["return"])

# This marks the start of the main experiment. 
mainExpClock = core.Clock() # Timer for tracking time stamps

# Show instruction: Video intro
paranoiaIntroInstructions.draw()
win.flip()
keys = event.waitKeys(keyList=["return"])

# Show instruction: Video ready
paranoiaReadyInstructions.draw()
win.flip()
keys = event.waitKeys(keyList=["return"])

# ==============================
# Send story start message to ET
# ==============================
if ET == 1:
    tracker.sendMessage("STORY_START")
    dfTimeStamps.loc[0, 'startETStory'] = mainExpClock.getTime()

# ==============
# Begin stimulus
# ==============

# Show the black cross
crossCentralBlack.draw()
win.flip()

# Record start time
dfTimeStamps.loc[0, 'storyStart'] = mainExpClock.getTime()
dfTimeStamps.to_csv(filename + '_timestamps.csv', index=False)  # Save partial data

# Play audio
paranoia.play()
paused = False
pause_count = 0

# Keep the dot until the audio finishes playing
while paranoia.status == PLAYING or paused:
    
    # Check for key presses
    keys = event.getKeys(keyList=['k', 'p'])
    
    # To end audio
    if 'k' in keys: # K for kill
        if KS == 1:
            paranoia.stop(reset=True)
            break
    elif 'p' in keys: # P for pause
        if not paused:
            # Pause the audio
            paranoia.pause()
            dfTimeStamps.loc[pause_count, 'storyPause'] = mainExpClock.getTime()
            dfTimeStamps.to_csv(filename + '_timestamps.csv', index=False)  # Save partial data
            paused = True
        else:
            paranoia.play()
            dfTimeStamps.loc[pause_count, 'storyRestart'] = mainExpClock.getTime()
            dfTimeStamps.to_csv(filename + '_timestamps.csv', index=False)  # Save partial data
            paused = False
            pause_count +=1
            
    # Redraw the dot to keep on screen
    crossCentralBlack.draw()
    win.flip()
    
    # Keep window open until audio finishes playing 
    core.wait(0.1)
            
 
# record end time
dfTimeStamps.loc[0,'storyEnd'] = mainExpClock.getTime()
dfTimeStamps.to_csv(filename + '_timestamps.csv', index=False)  # save partial data

# ============================
# Send story end message to ET
# ============================
if ET == 1:
    tracker.sendMessage("STORY_END")
    dfTimeStamps.loc[0, 'endETStory'] = mainExpClock.getTime()

# show instruction: break
breakInstructions.draw()
win.flip()
keys = event.waitKeys(keyList=["return"])

# show instruction: record intro
recordIntroInstructions1.draw()
win.flip()
keys = event.waitKeys(keyList=["return"])

recordIntroInstructions2.draw()
win.flip()
keys = event.waitKeys(keyList=["return"])

# ==================================
# Send recording start message to ET
# ==================================
if ET == 1:
    tracker.sendMessage("REC_START")
    dfTimeStamps.loc[0, 'startETVoiceRec'] = mainExpClock.getTime()

# start recording
mic.start()

# record start time
dfTimeStamps.loc[0,'recordStart'] = mainExpClock.getTime()
dfTimeStamps.to_csv(filename + '_timestamps.csv', index=False)  # save partial data

# show central white dot during recording
crossCentralBlack.draw()
win.flip()
keys = event.waitKeys(keyList=["return"])

# stop recording
mic.stop()

# record end time
dfTimeStamps.loc[0,'recordEnd'] = mainExpClock.getTime()
dfTimeStamps.to_csv(filename + '_timestamps.csv', index=False)  # save data

# ==================================
# Send recording end message to ET
# ==================================
if ET == 1:
    tracker.sendMessage("REC_END")
    dfTimeStamps.loc[0, 'endETVoiceRec'] = mainExpClock.getTime()

# save audio
audioClip = mic.getRecording()
audioClip.save(filename + '.wav')

# show instruction: finish
finishInstructions.draw()
win.flip()
keys = event.waitKeys(keyList=["g"])


# --------------------- Post-Experiment Settings --------------------------#
# Stop recording ET
if ET == 1:
    tracker.setRecordingState(False)

win.close()

# Disconnect ET
if ET == 1:
    tracker.setConnectionState(False)

# Quit IO Hub
io.quit()

# Explort participant ET data
if ET == 1:
    edf_root = ''
    edf_file = edf_root + '/' + filename + '.EDF'
    os.rename('et_data.EDF', edf_file)

# Close experiment
core.quit()




