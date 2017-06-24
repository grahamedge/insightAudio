#Third party packages
import deepdish as dd
import numpy as np
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioFeatureExtraction as aF
from scipy import signal
import pandas as pd
import timeit
import os

from scipy.spatial.distance import euclidean

from SQL.load_rows import load_audio_data
from SQL import settingsAWS

from Audio import summarize_cluster_labels as scl
from Audio import audio_functions as audiofunc

#Data analysis stuff
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn import cluster

from SQL.addrows import add_cluster_labels, create_cluster_row, add_one_cluster_label
from SQL.load_rows import load_cluster_labels, load_a_cluster_label


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

def undersample(x,N):
    '''divide the audio waveform into bins of size N in order to reduce the data size'''
    n_bins = int(np.floor(len(x)/N))
    x_sampl = x[0:N*n_bins].reshape((n_bins,N)).mean(axis=1)

    T = len(x) / Fs
    time = np.linspace(0, T, len(x_sampl))
    Fs_sampl = Fs / N
    return Fs_sampl, x_sampl, time

def get_features(Fs,x,start,stop, window = 1.0, step = 1.0):

    #start_time = timeit.default_timer()
    F = aF.stFeatureExtraction(x[start*Fs:stop*Fs], Fs, window*Fs, step*Fs);
    #elapsed_time = timeit.default_timer() - start_time
    #print('Basic feature extraction took %d seconds' % elapsed_time)

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
    filt = np.zeros(x.shape)
    for i in range(x.shape[1]):
        filt[:,i] = signal.filtfilt(b, a, x[:,i])
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
	#import the labelled start and stop times
	labels = pd.read_csv('/home/graham/Insight2017/YoutubeVids/IrelandTranscript.csv')
	t_start = labels['start'].tolist()
	t_stop = labels['stop'].tolist()
	t_type = labels['type'].tolist()

	return t_start, t_stop, t_type

def create_label_vecs(timevec, t_start, t_stop, t_type):
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


def apply_PCA(feature_matrix, n_components = 5):
	'''Apply PCA to reduce the size of feature space'''
	pca = PCA(n_components = n_components)
	#normalize to allow PCA to work
	feature_normed = normalize(feature_matrix, axis=0)
	pca.fit(feature_normed)

	#DONT NORMALIZE when transforming the variables...
	#	why does this work better?
	Features_reduced = pca.transform(feature_matrix)
	return Features_reduced

def get_jumps(A):

	#create shifted version of A
	B = np.roll(A,-1)

	start = np.roll(np.logical_and((np.not_equal(A,B)), B),1)    
	stop = np.logical_and((np.not_equal(A,B)), A)
	return start, stop

def cluster_audio(Features, FeatureTime, n_pca_components = 5, n_clusters = 2):

	desired_features = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
	feature_mask = [feature in desired_features for feature in range(Features.shape[0])]
	Features_masked = prep_features(Features,feature_mask)

	#Apply PCA
	Features_reduced = apply_PCA(Features_masked, n_components = n_pca_components)	
	#Features_reduced = Features_masked
	#Features_reduced = normalize(Features_masked, axis=0)
	#Features_reduced = add_time_feature(FeatureTime, Features_reduced)

	#Add nearby time features
	Features_reduced = add_nearby_time_features(Features_reduced,
											 n_time_steps = 9)

	#Apply cluster analysis
	clus = cluster.AgglomerativeClustering(n_clusters = n_clusters, affinity = 'l2', linkage = 'complete')
	clus.fit(Features_reduced)
	cluster_labels = clus.labels_.astype(bool)

	return cluster_labels, Features_reduced

def get_cluster_centroid(features, cluster_labels):
	'''from some cluster labels, determine statistics of the
	cluster shape and centre in feature space'''
	clusters = np.unique(cluster_labels)
	n_features = features.shape[1]
	centroids = np.zeros((len(clusters), n_features))

	#integer for the cluster ID
	for cluster in clusters:
		#integer for the feature
		for feature in range(n_features):
			centroids[cluster, feature] = features[cluster_labels==cluster, feature].mean()

	return centroids



def cluster_audio_with_rejection(Features, FeatureTime, 
	n_pca_components = 5, size_threshold = 0.08):
	'''performs clustering looking for 3 features, in order to find
	2 that are useful, and one that should be merged with the other two'''

	desired_features = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
	feature_mask = [feature in desired_features for feature in range(Features.shape[0])]
	Features_masked = prep_features(Features,feature_mask)

	#Apply PCA
	Features_reduced = apply_PCA(Features_masked, n_components = n_pca_components)	
	#Features_reduced = Features_masked
	#Features_reduced = normalize(Features_masked, axis=0)
	#Features_reduced = add_time_feature(FeatureTime, Features_reduced)

	#Add nearby time features
	Features_reduced = add_nearby_time_features(Features_reduced,
											 n_time_steps = 9)

	#Apply cluster analysis
	clus = cluster.AgglomerativeClustering(n_clusters = 3, affinity = 'l2', linkage = 'complete')
	clus.fit(Features_reduced)
	cluster_labels = clus.labels_.astype(int)
	n_labels = np.unique(cluster_labels)

	#check whether a small number of outliers have been found
	cluster_sizes = np.asarray([np.sum(cluster_labels==n) for n in n_labels])
	if cluster_sizes.min() < size_threshold*len(cluster_labels):
		minority_label = n_labels[cluster_sizes.argmin()]
		majority_labels = n_labels[n_labels != minority_label]
		minorities = np.zeros((len(cluster_labels),2))
		minorities[:,0] = np.arange(0,len(cluster_labels))
		minorities[:,1] = (cluster_labels == n_labels[cluster_sizes.argmin()])
		centroids = get_cluster_centroid(Features_reduced, cluster_labels)
		majority_centroids = centroids[majority_labels,:]

		for idx, minority in minorities[minorities[:,1]==True,:]:
			u = Features_reduced[int(idx),:]
			#assuming only two majority clusters here!
			v1 = majority_centroids[0,:]
			v2 = majority_centroids[1,:]
			if euclidean(u, v1) < euclidean(u, v2):
				cluster_labels[int(idx)] = majority_labels[0]
			else:
				cluster_labels[int(idx)] = majority_labels[1]

		if 0 not in np.unique(cluster_labels):
			cluster_labels[cluster_labels==2] = 0
		else:
			cluster_labels[cluster_labels==2] = 1

		#may have added lots of flicker... so filter
		cluster_labels = smooth_cluster_predictions(cluster_labels)		
				
	else:
		#no tiny cluster here... so just use 2 clusters
		clus = cluster.AgglomerativeClustering(n_clusters = 2, affinity = 'l2', linkage = 'complete')
		clus.fit(Features_reduced)
		cluster_labels = clus.labels_.astype(int)

	return cluster_labels, Features_reduced	

def analyse_cluster_performance(cluster_labels, T_times):
	accuracy = 100*np.mean(np.logical_not(cluster_labels) == T_times)
	accuracy_b = 100*np.mean(cluster_labels == T_times)
	TTR = 1 - float(np.sum(cluster_labels)) / len(cluster_labels)
	
	if accuracy < accuracy_b:
		accuracy = accuracy_b
		TTR = 1-TTR

	base_accuracy = 100*np.mean(np.ones(T_times.shape).astype(bool) == T_times)

	truth_TTR = float(np.sum(T_times)) / len(T_times)

	cluster_starts, cluster_stops = get_jumps(cluster_labels)
	n_interactions = np.sum(cluster_starts)

	truth_n_interactions = np.asarray([typ == 'S' for typ in t_type]).sum()

	print('The classification accuracy is %d' % accuracy)
	print('The trivial accuracy is %d' % base_accuracy)
	print('Calculated TTR is %0.2f' % TTR)
	print('Actual TTR is %0.2f' % truth_TTR)
	print('Detected %d student-teacher interactions' % n_interactions)
	print('Expected %d student-teacher interactions' % truth_n_interactions)    

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

def visualize_classification_vs_time(times, clusters, teacher_times, student_times):

	start = 0
	stop = times[-1]
	plot_times = (times >= start) & (times <= stop)

	minute_labels, minute_values = get_minute_labels(times)

	fig = plt.figure(figsize = (20,6), dpi = 60)
	ax = plt.subplots(111) 
	ax.plot(times[plot_times], student_times[plot_times], '--k')
	ax.set_xticks(minute_values)
	l = ax.set_xticklabels(minute_labels, rotation = 45)

	plt.show()

	# fig, ax = plt.figure(figsize = (20,6), dpi = 60)
	# plt.subplot(211)
	# plt.plot(times[plot_times], student_times[plot_times], '--k')
	# ax.set_xticks(minutes)
	# l = ax.set_xticklabels(minute_labels, rotation = 45)
	# plt.subplot(212)
	# plt.plot(times[plot_times], clusters[plot_times], '--b')
	# ax.set_xticks(minutes)
	# l = ax.set_xticklabels(minute_labels, rotation = 45)

	# plt.tight_layout()
	# plt.show()

def visualize_classification_clusters(clusters, features, teacher_times, student_times):

	#2D projection of feature space
	pca = PCA(n_components = 2)
	plot_features = pca.fit_transform(normalize(features, axis=0))

	student_class = clusters.astype(bool)
	teacher_class = np.logical_not(clusters)

	fontsize = 20
	titlesize = fontsize+4

	plt.figure(figsize = (16,8))
	plt.subplot(121)
	plt.plot(plot_features[teacher_times,0],
		plot_features[teacher_times,1],'.r')
	plt.plot(plot_features[student_times,0],
		plot_features[student_times,1], '.k')
	plt.xlabel('Audio Feature A', fontsize = fontsize)
	plt.ylabel('Audio Feature B', fontsize = fontsize)
	plt.title('Labelled Data', fontsize = titlesize)

	plt.subplot(122)
	plt.plot(plot_features[teacher_class,0],
		plot_features[teacher_class,1],'.r', label = 'Teacher')
	plt.plot(plot_features[student_class,0],
		plot_features[student_class,1], '.k', label = 'Student')
	plt.xlabel('Audio Feature A', fontsize = fontsize)
	plt.ylabel('Audio Feature B', fontsize = fontsize)
	plt.title('Unsupervised Clusters', fontsize = titlesize)

	plt.tight_layout()
	plt.show()

def add_time_feature(times, features):
	f = np.vstack((features.transpose(),times)).transpose()

	return f

def add_nearby_time_features(features, n_time_steps = 9 ):
	'''to smooth out some of the fast noise in the classes,
	add new features to each time bin which are scaled versions of
	nearby time bins... this is possibly similar to classifying
	each time point individually and later applying median filter

	n_time_steps is the total number points to include BOTH forward
	and backward in time, and should be an odd number'''

	n_times = features.shape[0]
	n_features = features.shape[1]
	expanded_features = np.zeros((n_times, 
				(n_time_steps)*n_features))

	n_steps = np.floor(n_time_steps/2)
	step_list = np.linspace(-n_steps, n_steps, n_time_steps).astype(int)

	sigma = 5	
	weight_list = np.exp(-1.0 * step_list*step_list / sigma**2)

	for i, step, weight in zip(range(n_time_steps), step_list, weight_list):
		shifted_features = np.roll(features, step, axis = 0)

		expanded_features[:,i*n_features:(i+1)*n_features] = weight*shifted_features

	return expanded_features


def smooth_cluster_predictions(cluster_labels, smooth_window = 5):

	cluster_labels_smoothed = signal.medfilt(cluster_labels, smooth_window)
	return cluster_labels_smoothed	

#-------------------------------
#	Run here
#-------------------------------

def plot_speaker_vs_time(yt_id):
	audio_df = load_audio_data(yt_id)
	Feature_Time = audio_df['time'].as_matrix()
	Features = audio_df.ix[:,2:36].as_matrix().transpose()

def calc_cluster_labels(yt_id = 'y2OFsG6qkBs'):

	#load audio features from SQL
	audio_df = load_audio_data(yt_id)
	Feature_Time = audio_df['time'].as_matrix()
	Features = audio_df.ix[:,2:36].as_matrix().transpose()
	print('Total number of feature rows: %d' % len(Feature_Time))

	#Cluster the audio features
	cluster_labels, fit_features = cluster_audio(Features, Feature_Time, n_pca_components = 4)
	cluster_labels = cluster_labels.astype(int)

	# cluster_labels = smooth_cluster_predictions(cluster_labels)

	return Feature_Time, cluster_labels

def calc_cluster_labels_with_rejection(yt_id = 'y2OFsG6qkBs'):

	#load audio features from SQL
	audio_df = load_audio_data(yt_id)
	Feature_Time = audio_df['time'].as_matrix()
	Features = audio_df.ix[:,2:36].as_matrix().transpose()
	print('Total number of feature rows: %d' % len(Feature_Time))

	#Cluster the audio features
	cluster_labels, fit_features = cluster_audio_with_rejection(Features, Feature_Time, n_pca_components = 4)

	# cluster_labels = smooth_cluster_predictions(cluster_labels)

	return Feature_Time, cluster_labels

def process_audio(yt_id = 'y2OFsG6qkBs'):

	#load audio features from SQL
	audio_df = load_audio_data(yt_id)
	Feature_Time = audio_df['time'].as_matrix()
	Features = audio_df.ix[:,2:36].as_matrix().transpose()
	print('Total number of feature rows: %d' % len(Feature_Time))

	#Cluster the audio features
	cluster_labels, fit_features = cluster_audio(Features, Feature_Time, n_pca_components = 4)
	cluster_labels = cluster_labels.astype(int)

	#Add the cluster labels into the SQL database
	add_cluster_labels(yt_id, Feature_Time, Features, cluster_labels)

	#Add summary statistics into a separate summary database
	#	scl.summarize_video(yt_id)

if __name__ == '__main__':
	# video_list = [
	# 			 'IHo_Fvx1V5I', DONE
	# 			 'y2OFsG6qkBs', DONE
	# 			 'kW_rOyL7xuc', DONE
	# 			 'AeioFIXDvhM',  #time_differences is empty!
	# 			 'G1dx947MAmk', # no audio features yet
	# 			 'LIIU7ZuzBi4', DONE
	# 			 '17wnvSd_Ndo', DONE
	# 			 'l6L2tUbQ4iM', # no audio features yet
	# 			 'oEQyAuz_hzs', DONE
	# 			 'lwdfoZ1Z3s8', DONE
	# 			 'fM3PqRcQ27o', DONE
	# 			 'GD7GNO08Epg' # no audio features yet
	# 			 ]	

	video_list = [
				'dqPjgQwoXLQ'
			]			 

	for video_id in video_list:

		print('Processing video %s...' % video_id)
		process_audio(video_id)

		# T, C, = calc_cluster_labels_with_rejection(video_id)
