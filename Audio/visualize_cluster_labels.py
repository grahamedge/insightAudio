'''
Functions to plot and otherwise visualize the performance
of the clustering algorithm
'''

#Basic libraries
import timeit
import os

#Third party packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn import cluster
import mpld3

#Homebuilt
from SQL import settings
from SQL.addrows import add_cluster_labels, create_cluster_row
from SQL.load_rows import load_cluster_labels, load_a_cluster_label, load_audio_data


def get_labels():
	'''Import some labelled data from a local file'''
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
  

def get_minute_labels(timevec):
    '''for a given time vector, return a list of markers and labels
    in mm:ss format (try to return at least 4 labels that span the full length)'''
    m_max = np.ceil(timevec[-1]/60)
    m_min = np.floor(timevec[0]/60)
    if (m_max-m_min)>15:
        m_step = 2
    else: m_step = 1
    if m_step == 0: m_step = 1
    m_labels = np.arange(m_min,m_max+1,m_step)
    s_values = m_labels*60
    m_labels = [('%d:00' % m) for m in m_labels]
    return m_labels, s_values

def get_minute_labels_II(timevec):
	'''for a given time vector, return a list of markers and labels
	in mm:ss format (for short videos, step every 10s)'''
	s_max = np.ceil(timevec[-1]/10)*10
	s_min = np.floor(timevec[0]/10)*10
	s_step = 10
	s_values = np.arange(s_min,s_max+1,s_step)
	m_labels = [('%d:%02d' % (np.floor(s/60), s%60)) for s in s_values]
	return m_labels, s_values    

def visualize_classification_vs_time_with_truth(times, clusters, teacher_times, student_times):
	'''plot the cluster labels over time, alongside the true values over time'''

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
	'''with a list of predicted classifications, this function plots a cartoon of the
	audio waveform and color-codes it with the cluster predictions for easy interpretation'''

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

def plot_clustered_waveform_without_wav(yt_id = 'y2OFsG6qkBs', start=1, stop = -1):

	#load audio features from DB
	audio_df = load_audio_data(yt_id)
	Feature_Time = audio_df['time'].as_matrix()
	Features = audio_df.ix[:,2:36].as_matrix().transpose()
	if stop == -1:
		stop = int(Feature_Time[-1])
	elif stop > int(Feature_Time[-1]):
		stop = int(Feature_Time[-1])

	#load cluster data
	cluster_df = load_cluster_labels(yt_id)
	cluster_times = cluster_df['time'].as_matrix()
	cluster_labels = cluster_df['cluster_label_raw'].as_matrix()

	#find the waveform times that match cluster labels
	teacher_labels = np.logical_not(cluster_labels.astype(bool))
	student_labels = np.logical_not(teacher_labels)
	teacher_times = cluster_times[teacher_labels]
	student_times = cluster_times[student_labels]

	# #pick the times from t that match the times from cluster_times where
	# cluster_labels_us = match_time_labels(time_us, cluster_times, cluster_labels)
	# teacher_labels_us = cluster_labels_us.astype(bool)
	# student_labels_us = np.logical_not(teacher_labels_us)

	#plot the waveform in a tractable way
	plot_start = start
	plot_stop = stop
	plot_times = (Feature_Time >= plot_start) & (Feature_Time <= plot_stop)

	#the last data points is treated poorly by filtering steps, so don't plot

	if stop-start > 120:
		minute_labels, minute_values = get_minute_labels(Feature_Time[plot_times])
	else:
		minute_labels, minute_values = get_minute_labels_II(Feature_Time[plot_times])

	# x_filtered = audio_func.low_pass_filter(1.0, Features[0,:], tau = 1, order=1)
	x_filtered = Features[0,:]
	x_max = x_filtered.max()
	x_teacher = np.zeros(x_filtered.shape)
	x_student = np.zeros(x_filtered.shape)
	x_teacher[teacher_labels] = x_filtered[teacher_labels]
	x_student[student_labels] = x_filtered[student_labels]

	fig = plt.figure(figsize = (8,3))
	ax = plt.subplot(111) 
	plt.fill_between(Feature_Time[plot_times],-x_teacher[plot_times]/x_max,x_teacher[plot_times]/x_max, facecolor='red')
	plt.fill_between(Feature_Time[plot_times],-x_student[plot_times]/x_max,x_student[plot_times]/x_max, facecolor='black')
	plt.ylim([-1,1])
	plt.xlabel('time (s)', fontsize = 14)
	plt.ylabel('Amplitude', fontsize = 14)
	ax.set_xticks(minute_values)
	ax.set_yticks([0])
	l = ax.set_xticklabels(minute_labels, rotation = 45, fontsize = 14	)
	l = ax.set_yticklabels([''], fontsize = 14)
	# fig.patch.set_visible(False)
	# ax.axis('off')

	plt.tight_layout()
	plt.show()

	return fig


def visualize_classification_vs_time_html(times, clusters):
	'''same functionality as visualize_classification_clusters
	but uses the mpld3 function to produce javascript for a web interface'''

	start = 0
	stop = times[-1]
	plot_times = (times >= start) & (times <= stop)

	minute_labels, minute_values = get_minute_labels(times)

	fig = plt.figure(figsize = (8,2))
	ax = plt.subplot(111) 
	ax.fill_between(times[plot_times],0, clusters[plot_times])
	ax.set_xticks(minute_values)
	ax.set_yticks([0])
	l = ax.set_xticklabels(minute_labels, rotation = 90, fontsize = 14	)
	l = ax.set_yticklabels([''], fontsize = 18)

	# plt.fill_between(times[plot_times],0, clusters[plot_times])

	fig_html = mpld3.fig_to_html(fig)

	return fig_html	

def plot_clustered_waveform_html(yt_id = 'y2OFsG6qkBs', start=1, stop = -1):

	#load audio features from DB
	audio_df = load_audio_data(yt_id)
	Feature_Time = audio_df['time'].as_matrix()
	Features = audio_df.ix[:,2:36].as_matrix().transpose()
	if stop == -1:
		stop = int(Feature_Time[-1])
	elif stop > int(Feature_Time[-1]):
		stop = int(Feature_Time[-1])

	#load cluster data
	cluster_df = load_cluster_labels(yt_id)
	cluster_times = cluster_df['time'].as_matrix()
	cluster_labels = cluster_df['cluster_label_raw'].as_matrix()

	#find the waveform times that match cluster labels
	teacher_labels = np.logical_not(cluster_labels.astype(bool))
	student_labels = np.logical_not(teacher_labels)
	teacher_times = cluster_times[teacher_labels]
	student_times = cluster_times[student_labels]

	# #pick the times from t that match the times from cluster_times where
	# cluster_labels_us = match_time_labels(time_us, cluster_times, cluster_labels)
	# teacher_labels_us = cluster_labels_us.astype(bool)
	# student_labels_us = np.logical_not(teacher_labels_us)

	#plot the waveform in a tractable way
	plot_start = start
	plot_stop = stop
	plot_times = (Feature_Time >= plot_start) & (Feature_Time <= plot_stop)

	#the last data points is treated poorly by filtering steps, so don't plot

	if stop-start > 120:
		minute_labels, minute_values = get_minute_labels(Feature_Time[plot_times])
	else:
		minute_labels, minute_values = get_minute_labels_II(Feature_Time[plot_times])

	# x_filtered = audio_func.low_pass_filter(1.0, Features[0,:], tau = 1, order=1)
	x_filtered = Features[0,:]
	x_max = x_filtered.max()
	x_teacher = np.zeros(x_filtered.shape)
	x_student = np.zeros(x_filtered.shape)
	x_teacher[teacher_labels] = x_filtered[teacher_labels]
	x_student[student_labels] = x_filtered[student_labels]

	fig = plt.figure(figsize = (8,3))
	ax = plt.subplot(111) 
	plt.fill_between(Feature_Time[plot_times],-x_teacher[plot_times]/x_max,x_teacher[plot_times]/x_max, facecolor='red')
	plt.fill_between(Feature_Time[plot_times],-x_student[plot_times]/x_max,x_student[plot_times]/x_max, facecolor='black')
	plt.ylim([-1,1])
	plt.xlabel('', fontsize = 14)
	plt.ylabel('', fontsize = 14)
	ax.set_xticks(minute_values)
	ax.set_yticks([0])
	l = ax.set_xticklabels(minute_labels, rotation = 45, fontsize = 14	)
	l = ax.set_yticklabels([''], fontsize = 14)
	# fig.patch.set_visible(False)
	ax.axis('off')

	plt.tight_layout()

	fig_html = mpld3.fig_to_html(fig)

	return fig_html		

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
	if plot_features.shape[1] > 34:
		#need the time axis on index 0 of the array
		plot_features = np.transpose(plot_features)

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

def set_teacher_cluster(cluster_labels):
	'''takes the cluster labels as integers (0 or 1, 
	corresponding to teacher / not-teacher) and ensures
	that: 0 = teacher
		  1 = not-teacher	 
	by assuming that teacher talks most!  '''

	if np.sum(cluster_labels) < 0.5:
		cluster_labels = np.logical_not(cluster_labels.astype(bool)).astype(int)

	return cluster_labels


def smooth_cluster_predictions(cluster_labels, smooth_window = 5):

	cluster_labels_smoothed = signal.medfilt(cluster_labels, smooth_window)
	return cluster_labels_smoothed	

def check_cluster_performance(yt_id):

	cluster_df = load_cluster_labels(yt_id)
	times = cluster_df['time'].as_matrix()
	cluster_labels = cluster_df['cluster_label_raw'].as_matrix()

	visualize_classification_vs_time(times, cluster_labels)


#---------
#Run here:
#---------

# yt_id = 'y2OFsG6qkBs'

# cluster_df = load_cluster_labels(yt_id)

# times = cluster_df['time'].as_matrix()
# cluster_labels = cluster_df['cluster_label_raw'].as_matrix()
# # 
# # visualize_classification_vs_time(times, cluster_labels)	

# audio_df = load_audio_data(yt_id)
# Feature_Time = audio_df['time'].as_matrix()
# Features = audio_df.ix[:,2:36].as_matrix()

# t_start, t_stop, t_type = get_labels()

# T_times, S_times = create_label_vecs(times, t_start, t_stop, t_type)

# visualize_classification_clusters(cluster_labels, Features, T_times, S_times)