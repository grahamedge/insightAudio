#Third party packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import timeit
import os

from SQL.load_rows import load_audio_data
from SQL import settings

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn import cluster

from SQL.addrows import add_cluster_labels, create_cluster_row, add_one_cluster_label
from SQL.load_rows import load_cluster_labels, load_a_cluster_label


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

def visualize_classification_vs_time_with_truth(times, clusters, teacher_times, student_times):

	start = 0
	stop = times[-1]
	plot_times = (times >= start) & (times <= stop)

	minute_labels, minute_values = get_minute_labels(times)

	fig = plt.figure(figsize = (20,6), dpi = 60)
	ax = plt.subplot(111) 
	ax.plot(times[plot_times], student_times[plot_times], '--k')
	ax.set_xticks(minute_values)
	l = ax.set_xticklabels(minute_labels, rotation = 45)

	plt.show()

def visualize_classification_vs_time(times, clusters):

	start = 0
	stop = times[-1]
	plot_times = (times >= start) & (times <= stop)

	minute_labels, minute_values = get_minute_labels(times)

	plt.figure(figsize = (20,6), dpi = 60)
	ax = plt.subplot(111) 
	ax.plot(times[plot_times], clusters[plot_times], '--k')
	ax.set_xticks(minute_values)
	l = ax.set_xticklabels(minute_labels, rotation = 45)

	plt.show()

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

#---------
#Run here:
#---------

yt_id = 'y2OFsG6qkBs'

cluster_df = load_cluster_labels(yt_id)

times = cluster_df['time'].as_matrix()
cluster_labels = cluster_df['cluster_label_raw'].as_matrix()
# 
# visualize_classification_vs_time(times, cluster_labels)	

audio_df = load_audio_data(yt_id)
Feature_Time = audio_df['time'].as_matrix()
Features = audio_df.ix[:,2:36].as_matrix().transpose()


t_start, t_stop, t_type = get_labels()

T_times, S_times = create_label_vecs(times, t_start, t_stop, t_type)

visualize_classification_clusters(cluster_labels, Features, T_times, S_times)