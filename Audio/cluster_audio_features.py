'''
Function to perform unsupervised clustering analysis on the
audio features of a waveform. Audio features are fetched from a SQL
database, and cluster results can be saved to the same database
'''

#Basic libraries
import timeit
import os

#Third party packages
import numpy as np
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioFeatureExtraction as aF
from scipy import signal
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn import cluster

#Homebuilt
from SQL import settingsAWS
from Audio import summarize_cluster_labels as scl
from Audio import audio_functions as audiofunc
from SQL.addrows import add_cluster_labels, create_cluster_row, add_one_cluster_label
from SQL.load_rows import load_cluster_labels, load_a_cluster_label, load_audio_data


def undersample(x,N):
    '''divide the audio waveform into bins of size N in order to reduce the data size'''
    n_bins = int(np.floor(len(x)/N))
    x_sampl = x[0:N*n_bins].reshape((n_bins,N)).mean(axis=1)

    T = len(x) / Fs
    time = np.linspace(0, T, len(x_sampl))
    Fs_sampl = Fs / N
    return Fs_sampl, x_sampl, time


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
	'''given a numpy array with boolean values, this function
	find all of the elements of the array for which the value changes from
	the previous array element (the 'jumps' in the array value)

	e.g. this is used to find the precise seconds at which the speaker changes based on
	a numpy vector which contains speaker classification for all of the seconds
	in an audio file '''

	#create shifted version of A
	B = np.roll(A,-1)

	start = np.roll(np.logical_and((np.not_equal(A,B)), B),1)    
	stop = np.logical_and((np.not_equal(A,B)), A)
	return start, stop

def cluster_audio(Features, FeatureTime, n_pca_components = 5, n_clusters = 2):
	'''given a list of audio features, this function produces a smaller
	set of features using Principal Component Analysis (PCA), and then uses 
	hierarchical clustering to try to discover structure in the reduced
	feature space

	after reducing the number of features with PCA, this function also adds
	time-shifted versions of features to each timestep in the file - an expansion
	of the data that is intended to make nearby timesteps look more similar to reduce
	the prevalence of 'digital flicker' noise in the classification

	returns a list of cluster labels for each timestep, as well as the list
	of features in the reduced feature space produced by PCA'''

	desired_features = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
	feature_mask = [feature in desired_features for feature in range(Features.shape[0])]
	Features_masked = prep_features(Features,feature_mask)

	#Apply PCA
	Features_reduced = apply_PCA(Features_masked, n_components = n_pca_components)	

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
	'''with a numpy array of predicted cluster labels, as well as an array
	of true speaker classifications from labelled data, this function calculates
	the per-second classification accuracy of the algorithm and prints some results
	to the terminal '''

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
	'''with a list of predicted classifications, this function plots a cartoon of the
	audio waveform and color-codes it with the cluster predictions for easy interpretation'''
	
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


def visualize_classification_clusters(clusters, features, teacher_times, student_times):
	'''takes a list of predicted cluster labels "clusters" as well as the true
	speaker labels "teacher_times" and "student_times", and uses them to visualize
	the performance of clustering in the audio feature space specified by the numpy
	array "features"

	since there are more audio features than can be shown in a 2D or 3D plot, options are:
	- plot only a couple of the audio features
	- use t-SNE on the known speaker labels to find the best low-dimensional	
		representation for plotting
	- pretend that we don't know the true labels and use PCA to reduce to a
		low-dimensional representation for plotting that may or may not actually
		be the best way to separate the clusters

	currently, the third option (PCA into 2 features) is used'''


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
	'''uses median filtering to reduce noise in the cluster predictions'''

	cluster_labels_smoothed = signal.medfilt(cluster_labels, smooth_window)
	return cluster_labels_smoothed	

#-------------------------------
#	Run here
#-------------------------------

def calc_cluster_labels(yt_id = 'y2OFsG6qkBs'):
	'''load the audio features from the SQL database and find clusters'''

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
	'''Load audio features from the SQL database and find clusters,
	checking for anomalously small clusters and attempting to remove them'''


	#load audio features from SQL
	audio_df = load_audio_data(yt_id)
	Feature_Time = audio_df['time'].as_matrix()
	Features = audio_df.ix[:,2:36].as_matrix().transpose()
	print('Total number of feature rows: %d' % len(Feature_Time))

	#Cluster the audio features
	cluster_labels, fit_features = cluster_audio_with_rejection(Features, Feature_Time, n_pca_components = 4)

	# cluster_labels = smooth_cluster_predictions(cluster_labels)

	return Feature_Time, cluster_labels

def add_cluster_info_to_sql(yt_id = 'y2OFsG6qkBs'):
	'''basic function to load audio data from SQL, apply the
	clustering algorithm, and then add the cluster information 
	to the database'''

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

if __name__ == '__main__':

	video_list = [
				'dqPjgQwoXLQ'
			]			 

	for video_id in video_list:

		print('Calculating clusters video %s...' % video_id)
		add_cluster_info_to_sql(video_id)
