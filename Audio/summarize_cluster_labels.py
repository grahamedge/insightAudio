'''
Functions to take a labelled waveform and produce summary statistics
such as the number of questions, the ratio of time each speaker talks,
voice variation of each speaker, etc...
'''

#Basic packages
import timeit
import os

#Third party packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import mpld3

#Homebuilt
import calc_audio_features as audio_func
from SQL.addrows import add_results, create_results_row
from SQL.load_rows import load_cluster_labels, load_audio_data, load_intensity


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
	#some short videos need to be labelled in 30s increments
	if m_max < 6:
		if m_max < 3:
			s_step = 30.0
		else:
			s_step = 60.0
	else:
		s_step = 60.0*round((m_max-m_min)/6)

	s_values = np.arange(m_min*60.0, m_max*60.0, s_step)
	m_labels = np.floor(s_values/60).astype(int)
	s_labels = (s_values % 60).astype(int)
	mmss_labels = [('%d:%02d' % (m,s)) for m,s in zip(m_labels, s_labels)]
	return mmss_labels, s_values


def get_jumps(A):

	#create shifted version of A
	B = np.roll(A,-1)

	start = np.roll(np.logical_and((np.not_equal(A,B)), B),1)
	stop = np.logical_and((np.not_equal(A,B)), A)
	return start, stop	


def smooth_cluster_predictions(cluster_labels, smooth_window = 5):

	cluster_labels_smoothed = signal.medfilt(cluster_labels, smooth_window)
	return cluster_labels_smoothed	

def set_teacher_cluster(cluster_labels):
	'''takes the cluster labels as integers (0 or 1, 
	corresponding to teacher / not-teacher) and ensures
	that: 0 = teacher
		  1 = not-teacher	 
	by assuming that teacher talks most!  '''

	if np.sum(cluster_labels) < 0.5:
		cluster_labels = np.logical_not(cluster_labels.astype(bool)).astype(int)

	return cluster_labels	

def analyse_cluster_performance(times, cluster_labels):
	
	metrics = {}

	# get the TTR (time bins are equal, so just need to sum)
	TTR = 1 - float(np.sum(cluster_labels)) / len(cluster_labels)
	metrics['teacher_talk_ratio'] = TTR

	cluster_starts, cluster_stops = get_jumps(cluster_labels)
	
	#find the start times and stop times for the non-teacher speaker
	start_times = times[cluster_starts]
	stop_times = times[cluster_stops]
	n_interactions = np.sum(cluster_starts)
	metrics['interaction_times'] = start_times

	interaction_lengths = stop_times - start_times
	metrics['number_of_interactions'] = n_interactions
	metrics['interaction_lengths'] = interaction_lengths

	metrics = get_interaction_distribution(interaction_lengths, metrics)
	
	return metrics

def get_interaction_distribution(interaction_lengths, metrics):

	cutoff = 3.0 #max length of interaction (seconds) to consider short
	n_short_interactions = np.sum(interaction_lengths < cutoff)
	metrics['n_short_interactions'] = n_short_interactions

	long_cutoff = 20.0
	n_long_interactions = np.sum(interaction_lengths > long_cutoff)
	metrics['n_long_interactions'] = n_long_interactions

	return metrics

def histogram_interactions(interaction_lengths):
	'''Look at the way the detected interactions are distributed'''
	plt.hist(interaction_lengths, bins = [0.5, 1, 1.5, 2, 3, 4, 5, 7, 9, 12, 15, 20, 25, 30, 35, 40])
	plt.title("Interaction Lengths")
	plt.xlabel("Time (s)")
	plt.ylabel("Occurrences")

	plt.show()

def find_questions(interaction_times, interaction_lengths):
	'''Try to find a period in the file that looks like questions'''

	period_length = 25

	#find reasonably length interactions
	short_interactions = (interaction_lengths < 3.0)

	#find the best cluster of them
	n = 1
	short_times = interaction_times[short_interactions]
	too_far = False
	while not too_far:
		shifted_times = np.roll(short_times,-n)
		time_differences = shifted_times[0:-n] - short_times[0:-n]
		long_enough = time_differences < period_length
		if not np.any(long_enough):
			too_far = True
		else:
			question_index = time_differences.argmin()
			question_period_time = short_times[question_index]
			question_period_length = time_differences[question_index]
			n = n+1

	if n == 1:
		question_index = time_differences.argmin()
		question_period_time = short_times[question_index]
		question_period_length = time_differences[question_index]

	#need the index in the actual time array, not the short time arrays
	time_idx = np.where(
		interaction_times == short_times[question_index])[0]

	question_start = int(question_period_time) - 5
	question_end = int(question_period_time) + int(question_period_length)	 + 2	

	question_indices = range(time_idx,time_idx+n)

	return question_start, question_end, question_indices

def find_longest_question(interaction_times, interaction_lengths):
	'''finds the longest period of time in which the confidence
	is high for student talking'''
	longest_question = interaction_times[interaction_lengths.argmax()]

	#return start and end times
	start = int(longest_question)-15
	stop = int(longest_question) + int(interaction_lengths[interaction_lengths.argmax()]) + 10
	index = interaction_lengths.argmax()

	return start, stop, index

def get_question_periods(metrics):

	question_start_a, question_stop_a, indices_a = find_questions(
		metrics['interaction_times'], 
		metrics['interaction_lengths'])
	question_start_b, question_stop_b, indices_b = find_questions(
		np.delete(metrics['interaction_times'], indices_a), 
		np.delete(metrics['interaction_lengths'], indices_a))
	question_start_c, question_stop_c, indices_c = find_questions(
		np.delete(metrics['interaction_times'], indices_a + indices_b ), 
		np.delete(metrics['interaction_lengths'], indices_a + indices_b ))					

	long_question_start_a, long_question_stop_a, index_a = find_longest_question(
		metrics['interaction_times'], metrics['interaction_lengths'])
	long_question_start_b, long_question_stop_b, index_b = find_longest_question(
		np.delete(metrics['interaction_times'], index_a),
		np.delete(metrics['interaction_lengths'], index_a))				

	metrics['question_start_a'] = question_start_a
	metrics['question_start_b'] = question_start_b
	metrics['question_start_c'] = question_start_c
	metrics['question_stop_a'] = question_stop_a
	metrics['question_stop_b'] = question_stop_b
	metrics['question_stop_c'] = question_stop_c
	metrics['long_question_start_a'] = long_question_start_a
	metrics['long_question_start_b'] = long_question_start_b
	metrics['long_question_stop_a'] = long_question_stop_a
	metrics['long_question_stop_b'] = long_question_stop_b

	return metrics

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def match_time_labels(dest_times, cluster_times, cluster_labels):
	'''takes the set of time points cluster_times, and the associated
	cluster labels cluster_labels, and returns the cluster labels for each
	time in the differently sampled time vector dest_times'''
	dest_labels = np.zeros(dest_times.shape)

	#fill in all of the times which are 
	for i, time in enumerate(dest_times):
		dest_labels[i] = cluster_labels[find_nearest(cluster_times, time)]

	return dest_labels

def plot_clustered_waveform_without_wav(yt_id = 'y2OFsG6qkBs'):

	#load audio features from DB
	audio_df = load_audio_data(yt_id)
	Feature_Time = audio_df['time'].as_matrix()
	Features = audio_df.ix[:,2:36].as_matrix().transpose()
	start = 1
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

	minute_labels, minute_values = get_minute_labels(Feature_Time[plot_times])

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
	ax.axis('off')

	plt.tight_layout()

	return fig

def plot_clustered_waveform(yt_id = 'y2OFsG6qkBs'):

	Fs, x = audio_func.load_waveform(yt_id)
	try:
		x = audio_func.get_mono(x)
	except TypeError:
		print('Audio not properly loaded, make sure that audio data is accessible!')
	t = audio_func.get_time_vec(Fs,x)
	start = 1
	stop = int(t[-1])

	n_min = int(t[-1]/60)
	N_undersample = 10000*n_min
	Fs_us, x_us, time_us = audio_func.undersample(Fs, x, N=N_undersample)

	#load cluster data
	cluster_df = load_cluster_labels(yt_id)
	cluster_times = cluster_df['time'].as_matrix()
	cluster_labels = cluster_df['cluster_label_raw'].as_matrix()

	#find the waveform times that match cluster labels
	teacher_times = cluster_times[np.logical_not(cluster_labels.astype(bool))]
	student_times = cluster_times[cluster_labels]

	#pick the times from t that match the times from cluster_times where
	cluster_labels_us = match_time_labels(time_us, cluster_times, cluster_labels)
	teacher_labels_us = cluster_labels_us.astype(bool)
	student_labels_us = np.logical_not(teacher_labels_us)

	#plot the waveform in a tractable way
	plot_start = start
	plot_stop = stop
	plot_times = (time_us >= plot_start) & (time_us <= plot_stop)

	#the last data points is treated poorly by filtering steps, so don't plot

	minute_labels, minute_values = get_minute_labels(time_us[plot_times])

	x_filtered = audio_func.low_pass_filter(Fs_us, x_us, tau = 1, order=3)
	x_max = x_filtered.max()
	x_teacher = np.zeros(x_filtered.shape)
	x_student = np.zeros(x_filtered.shape)
	x_teacher[teacher_labels_us] = x_filtered[teacher_labels_us]
	x_student[student_labels_us] = x_filtered[student_labels_us]

	fig = plt.figure(figsize = (8,3))
	ax = plt.subplot(111) 
	plt.fill_between(time_us[plot_times],-x_teacher[plot_times]/x_max,x_teacher[plot_times]/x_max, facecolor='red')
	plt.fill_between(time_us[plot_times],-x_student[plot_times]/x_max,x_student[plot_times]/x_max, facecolor='black')
	plt.ylim([-1,1])
	plt.xlabel('time (s)', fontsize = 14)
	plt.ylabel('Amplitude', fontsize = 14)
	ax.set_xticks(minute_values)
	ax.set_yticks([0])
	l = ax.set_xticklabels(minute_labels, rotation = 45, fontsize = 14	)
	l = ax.set_yticklabels([''], fontsize = 14)
	# fig.patch.set_visible(False)
	ax.axis('off')

	plt.tight_layout()

	return fig

def plot_waveform(yt_id = 'y2OFsG6qkBs'):

	Fs, x = audio_func.load_waveform(yt_id)
	try:
		x = audio_func.get_mono(x)
	except TypeError:
		print('Audio not properly loaded, make sure that audio data is accessible!')
	t = audio_func.get_time_vec(Fs,x)
	start = 1
	stop = int(t[-1])

	Fs_us, x_us, time_us = audio_func.undersample(Fs, x,N=10000)

	#plot the waveform in a tractable way
	plot_start = start
	plot_stop = stop
	plot_times = (time_us >= plot_start) & (time_us <= plot_stop)

	minute_labels, minute_values = get_minute_labels(time_us[plot_times])

	x_filtered = audio_func.low_pass_filter(Fs_us, x_us, tau = 1, order=3)
	x_max = x_filtered.max()

	fig = plt.figure(figsize = (8,3))
	ax = plt.subplot(111) 
	plt.fill_between(time_us[plot_times],-x_filtered[plot_times]/x_max,x_filtered[plot_times]/x_max)
	plt.ylim([-1,1])
	plt.xlabel('time (s)', fontsize = 14)
	plt.ylabel('Amplitude', fontsize = 14)
	ax.set_xticks(minute_values)
	ax.set_yticks([0])
	l = ax.set_xticklabels(minute_labels, rotation = 45, fontsize = 14	)
	l = ax.set_yticklabels([''], fontsize = 14)
	ax.axes.get_yaxis().set_visible(False)
	plt.tight_layout()

	return fig	

#---------
#Run here:
#---------

def summarize_video(yt_id = 'y2OFsG6qkBs'):

	cluster_df = load_cluster_labels(yt_id)

	times = cluster_df['time'].as_matrix()
	cluster_labels = cluster_df['cluster_label_raw'].as_matrix()

	cluster_labels = set_teacher_cluster(cluster_labels)

	metrics = analyse_cluster_performance(times, cluster_labels)
	metrics['video_length'] = times[-1]

	metrics = get_question_periods(metrics)

	#save all the metrics to the results table
	add_results(yt_id, metrics)

def plot_speaker_vs_time(yt_id):
	cluster_df = load_cluster_labels(yt_id)

	times = cluster_df['time'].as_matrix()
	cluster_labels = cluster_df['cluster_label_raw'].as_matrix()

	fig_html = visualize_classification_vs_time_html(times, cluster_labels)

	return fig_html

def plot_speaker_vs_time_test(yt_id):
	cluster_df = load_cluster_labels(yt_id)

	times = cluster_df['time'].as_matrix()
	cluster_labels = cluster_df['cluster_label_raw'].as_matrix()

	fig = visualize_classification_vs_time(yt_id, times, cluster_labels)

	return fig


if __name__ == '__main__':

	# yt_id = 'y2OFsG6qkBs'

	# summarize_video(yt_id)

	# html = plot_clustered_waveform_html('dqPjgQwoXLQ')
	# # print(html)

	# visualize_classification_clusters(clusters, features, teacher_times, student_times)

	fig = plot_clustered_waveform_without_wav('dqPjgQwoXLQ')
	plt.show()