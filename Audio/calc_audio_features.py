'''
Functions to take a .wav audio file and convert into audio features
(ZCR, MFCCs, Chroma)
'''

#Basic packages
import timeit
import os

#Third party packages
import deepdish as dd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn import cluster

#PyAudioAnalysis package by Theodoros Giannakopoulos
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioFeatureExtraction as aF

from SQL.addrows import add_audio_features


def load_data_table(filename = '/media/graham/OS/Linux Content/Youtube/teacher_audio_data.h5'):
    '''load HDF5 data'''
    d = dd.io.load(HDF5_file)
    return d

def find_audio_file(yt_id, folder = '/media/graham/OS/Linux Content/Youtube/Audio/'):
    '''produce the path to the .wav file'''
    loc = file_loc = folder + yt_id + '.wav'
    return loc
    
    
def load_waveform(yt_id, folder = '/media/graham/OS/Linux Content/Youtube/Audio/'):
    '''load the .wav file corresponding to a given Youtube ID'''
    file_loc = folder + yt_id + '.wav'
    [Fs, x] = aIO.readAudioFile(file_loc)
    return Fs, x

def get_mono(x):
    '''average over the two audio channels to produce a mono signal'''
    x_mono = x.mean(axis=1)
    return x_mono

def get_time_vec(Fs, x):
    '''with a given sampling rate, produce a vector of times to use for plotting'''
    T = len(x) / Fs
    timevec = np.linspace(0, T, len(x))
    return timevec

def undersample(Fs,x,N):
    '''divide the audio waveform into bins of size N in order to reduce the data size'''
    n_bins = int(np.floor(len(x)/N))
    x_sampl = x[0:N*n_bins].reshape((n_bins,N)).mean(axis=1)

    T = len(x) / Fs
    time = np.linspace(0, T, len(x_sampl))
    Fs_sampl = Fs / N
    return Fs_sampl, x_sampl, time

def get_features(Fs,x,start,stop, window = 1.0, step = 1.0):
	'''calculate basic audio features such as zero-crossing-rate, MFCCs,
	and chroma vector using the pyAudioAnalysis package

	calculation typically takes around one second per minute of audio'''

	F = aF.stFeatureExtraction(x[start*Fs:stop*Fs], Fs, window*Fs, step*Fs)

	#Create a time vector appropriate for plotting the features F
	time_F = np.linspace(start, stop, F.shape[1])

	return F, time_F

def expand_features(features, Fs, dt, num_frames):
    '''repeat the values of the audio features in each time bin to help with smooth plotting
    Fs is the sampling rate that is desired
    dt is the total time for which features are captured
    num_frames is the number of times each audio feature is sampled in the given time'''
    F_plot = np.repeat(features,dt*Fs/num_frames, axis=1)
    return F_plot

def low_pass_filter(Fs, x, tau = 2, order=3):
	'''defines a butterworth filter with the specified order
	with a time constant of tau seconds (determining the cut-off 
	frequency / -3db point )'''
	f_abs = 1.0/tau
	f_crit = f_abs/Fs  #critical frequency in the range 0-1 of the Nyquist frequency
	b, a = signal.butter(order, f_crit)
	
	#check if this is a matrix of features
	if x.ndim > 1:
		filt = np.zeros(x.shape)
		for i in range(x.shape[1]):
			filt[:,i] = signal.filtfilt(b, a, x[:,i])
	else:
		filt = signal.filtfilt(b, a, x[:])

	return filt

def get_minute_labels(timevec):
    '''for a given time vector, return a list of markers and labels'''
    m_max = np.ceil(timevec[-1]/60)
    m_min = np.floor(timevec[0]/60)
    m_step = round((m_max-m_min)/12)
    if m_step == 0: m_step = 1
    m_labels = np.arange(m_min,m_max+1,m_step)
    s_values = m_labels*60
    m_labels = [('%d:00' % m) for m in m_labels]
    return m_labels, s_values


def PCA_reduce(features, featuremask):
    '''use PCA analysis in sklearn to whittle down the relevant features'''
    pca = PCA(n_components = 3)
    pca.fit(features[featuremask])
    
    return pca.transform(features[featuremask])

#try PCA on these audio features
def prep_features(features, feature_mask):
    '''select desired features and normalize them for PCA'''
    #also need to transpose because PCA works on axis=1
    features_masked = features[feature_mask, :].transpose()
    return features_masked

def get_labels():
	'''import some labelled data to check classification'''
	labels = pd.read_csv('/home/graham/Insight2017/YoutubeVids/IrelandTranscript.csv')
	t_start = labels['start'].tolist()
	t_stop = labels['stop'].tolist()
	t_type = labels['type'].tolist()

	return t_start, t_stop, t_type

def create_label_vecs(timevec, t_start, t_stop, t_type):
    '''with lists of times in which a given speaker starts and stops speaking
    this function produces numpy vectors T_times and S_time that label each 
    timestep in the time vector "timevec" with boolean values corresponding
    to whether the specific speaker is talking at that timestep'''
    T_times = np.zeros(timevec.shape).astype(int)
    S_times = np.zeros(timevec.shape).astype(int)

    for start, stop, typ in zip(t_start, t_stop, t_type):
        if typ == 'T':
            new_times = (timevec > start) & (timevec < stop)
            T_times = T_times + new_times
        elif typ == 'S':
            new_times = (timevec > start) & (timevec < stop)
            S_times = S_times + new_times
    T_times = T_times.astype(bool)
    S_times = S_times.astype(bool)
    return T_times, S_times	

def get_jumps(A):
	'''given a numpy array with boolean values, this function
	find all of the elements of the array for which the value changes from
	the previous array element (the 'jumps' in the array value)

	e.g. this is used to find the precise seconds at which the speaker changes based on
	a numpy vector which contains speaker classification for all of the seconds
	in an audio file '''

	#create shifted version of A
	B = np.roll(A,-1)

	#find the array elements in which the original and shifted
	# 	versions of A do not agree... these are the 'jumps'
	start = np.roll(np.logical_and((np.not_equal(A,B)), B),1)    
	stop = np.logical_and((np.not_equal(A,B)), A)
	return start, stop
 

def get_minute_labels(timevec):
    '''for a given time vector, return a list of markers and labels
    in the mm:ss format, to annotate plots'''
    m_max = np.ceil(timevec[-1]/60)
    m_min = np.floor(timevec[0]/60)
    m_step = round((m_max-m_min)/12)
    if m_step == 0: m_step = 1
    m_labels = np.arange(m_min,m_max+1,m_step)
    s_values = m_labels*60
    m_labels = [('%d:00' % m) for m in m_labels]
    return m_labels, s_values


#-----------------------------------
#	Main functions to process a file
#-----------------------------------

def process_audio(yt_id = 'y2OFsG6qkBs'):

	#analyse the audio file
	Fs, x = load_waveform(yt_id)
	#some audio files are downloaded as mono, others
	# 	must be converted
	if x.ndim == 2:
		x = get_mono(x)
	timevec = get_time_vec(Fs,x)
	start = 1
	stop = int(timevec[-1])
	video_length = stop

	#Analyse the file
	window_size = 1.0
	step_size = 0.5
	Features, FeatureTime = get_features(Fs,x,start, stop, window = window_size, step = step_size)

	add_audio_features(yt_id, FeatureTime, Features)

	return Features, FeatureTime

#run on a list of youtube video files for which an audio file is available
if __name__ == '__main__':
	video_list = [
				#'IHo_Fvx1V5I',
				 # 'y2OFsG6qkBs',
				# 'kW_rOyL7xuc',
				# 'AeioFIXDvhM',
				# 'G1dx947MAmk',
				# 'LIIU7ZuzBi4',
				'17wnvSd_Ndo',
				# 'l6L2tUbQ4iM',
				# 'oEQyAuz_hzs',
				# 'lwdfoZ1Z3s8',
				# 'fM3PqRcQ27o',
				# 'GD7GNO08Epg'
				]

	for video_id in video_list:
		print('Processing video %s...' % video_id)
		process_audio(video_id)