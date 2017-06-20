'''
Function to take a list of Youtube video IDs and create the data table
that will be useful for Teacher Effectiveness Assessment
'''

#standard libraries
import os
import codecs
#third-party libraries
import ffmpy
import deepdish as dd
import pycaption
import numpy as np
import youtube_dl
#homebuilt libraries
import youtube_file_handling as yt


youtube_id_list = [
	'17wnvSd_Ndo',
	'AeioFIXDvhM',
	'fM3PqRcQ27o',
	'G1dx947MAmk',
	'GD7GNO08Epg',
	'IHo_Fvx1V5I',
	'kW_rOyL7xuc',
	'l6L2tUbQ4iM',
	'LIIU7ZuzBi4',
	'lwdfoZ1Z3s8',
	'oEQyAuz_hzs',
	'y2OFsG6qkBs']

youtube_video_folder 	= '/media/graham/OS/Linux Content/Youtube/Videos/'
youtube_audio_folder 	= '/media/graham/OS/Linux Content/Youtube/Audio/'
HDF5_file = '/media/graham/OS/Linux Content/Youtube/teacher_audio_data.h5'

if not os.path.isfile(HDF5_file):
	#create empty hdf5 file
	d = {}
	dd.io.save(HDF5_file, d)

for yt_id in youtube_id_list:

	data = {}

	#check whether the file is already downloaded
	fname = youtube_video_folder + yt_id
	if os.path.isfile(str(fname + '.mkv')):
		# .mkv file exists
		container_type = '.mkv'
		title = yt.download_video_metadata(yt_id)
	elif os.path.isfile(str(fname + '.mp4')):
		# .mp4 file exists
		container_type = '.mp4'
		title = yt.download_video_metadata(yt_id)
	else:
		# file does not exist yet
		video_exists, title = yt.download_video(yt_id)
		container_type = 'mkv'

	data['title'] = title

	#check whether subtitle file exists
	fname = youtube_video_folder + yt_id + '.en.vtt'
	if os.path.isfile(fname):
		text, seconds = yt.read_vtt_captions(yt_id)
		#save some video information to the data structure
		data['caption text'] = text
		data['caption seconds'] = seconds
		data['caption wps'] = yt.calc_wps(text, seconds)
	else:
		print('No caption file found for ID %s!' % yt_id)

	#check whether .wav has been already created
	fname = youtube_audio_folder + yt_id + '.wav'
	if os.path.isfile(fname):
		# .wav already exists
		audio_exists = True
	else:
		# .wav file not created yet
		audio_exists = yt.convert_to_wav(yt_id, vid_type = container_type, 
			video_loc = youtube_video_folder, 
			audio_loc = youtube_audio_folder)

	#process audio
	# audio_features = calc_audio_features(wav_file)

	#save all data into the dictionary
	d = dd.io.load(HDF5_file)
	d[yt_id] = data
	dd.io.save(HDF5_file, d)

