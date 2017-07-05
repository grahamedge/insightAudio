
#Built in libaries
import os
import codecs

#third-party libraries
import ffmpy
import pycaption
import numpy as np
import youtube_dl

#Database functions
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, DateTime, String, Float, Time

#homebuilt libraries
from Youtube import youtube_file_handling as yt
import calc_audio_features as calc
import cluster_audio_features as clus
import summarize_cluster_labels as summ
import visualize_cluster_labels as vis
import audio_functions as af

#Database settigns
from SQL import settingsAWS
from SQL.addrows import create_audio_row, add_audio_row, add_audio_features
from SQL.load_rows import check_audio_data, check_cluster_data

yt_id = 'GD7GNO08Epg'
re_process_audio = False
reclustering = False

t_start = 0
t_stop = 120

#choose a video
youtube_video_folder 	= '/media/graham/OS/Linux Content/Youtube/Videos/'
youtube_audio_folder 	= '/media/graham/OS/Linux Content/Youtube/Audio/'

#check whether the file is already downloaded
fname = youtube_video_folder + yt_id
if os.path.isfile(str(fname + '.mkv')):
	# .mkv file exists
	container_type = '.mkv'
	title = yt.download_video_metadata(yt_id)
	print('MKV file exists!')
elif os.path.isfile(str(fname + '.mp4')):
	# .mp4 file exists
	container_type = '.mp4'
	title = yt.download_video_metadata(yt_id)
	print('MP4 file exists!')
else:
	# file does not exist yet
	print('Downloading file...')
	video_exists, title = yt.download_video(yt_id)
	container_type = '.mkv'

#check whether .wav has been already created
wav_file = youtube_audio_folder + yt_id + '.wav'
if os.path.isfile(wav_file):
	# .wav already exists
	print('File already converted!')
	audio_exists = True
else:
	# .wav file not created yet
	print('Converting WAV file...')
	audio_exists = yt.convert_to_wav(yt_id, vid_type = container_type, 
		video_loc = youtube_video_folder, 
		audio_loc = youtube_audio_folder)

#check whether audio is already processed!
results = check_audio_data(yt_id)
if (results["count"].values == 0 or re_process_audio == True):
	#File has not been added to the audio_features database

	#Get audio features and add to database
	print('Processing audio...')
	Features, FeatureTime = calc.process_audio(yt_id)

#check whether audio has been clustered!
results = check_cluster_data(yt_id)
if (results["count"].values == 0 or reclustering == True):

	#Cluster the features
	print('Clustering audio...')
	clus.add_cluster_info_to_sql(yt_id)

#Viualize clustering
# fig = vis.plot_clustered_waveform_without_wav(yt_id, t_start, t_stop)

# #Produce summary info for the video
summ.summarize_video(yt_id)