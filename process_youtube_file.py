'''
takes a youtube ID and adds the audio data into SQL

text info is stored in the SQL database with composite primary keys:
	- 'youtube_id' the video identifier
	- 'time' a float for the number of seconds elapsed in the video
'''

import os
import codecs
#third-party libraries
import ffmpy
import pycaption
import numpy as np
import youtube_dl

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, DateTime, String, Float, Time
#homebuilt libraries
from Youtube import youtube_file_handling as yt
from SQL.addrows import create_audio_row, add_audio_row, add_audio_features
from SQL import settings

from Audio import audio_functions as af


#start the connection
engine = create_engine(settings.DATABASE_URI, echo=True)
Session = sessionmaker(bind=engine)
dbsession = Session()

#choose a video
youtube_video_folder 	= '/media/graham/OS/Linux Content/Youtube/Videos/'
youtube_audio_folder 	= '/media/graham/OS/Linux Content/Youtube/Audio/'

# youtube_id_list = [
	# '17wnvSd_Ndo',
	# 'AeioFIXDvhM',
	# 'fM3PqRcQ27o',
	# 'G1dx947MAmk',
	# 'GD7GNO08Epg',
	# 'IHo_Fvx1V5I',
	# 'kW_rOyL7xuc',
	# 'l6L2tUbQ4iM',
	# 'LIIU7ZuzBi4',
	# 'lwdfoZ1Z3s8',
	# 'oEQyAuz_hzs',
	# 'y2OFsG6qkBs']

easy_youtube_list = [
			#'Cj1TJTl-XE0',
			'5MJR74vZgX0t',
			'5qsVB5ZmrD4',
			'33-IPvOHc6M'
			]


# yt_id_list = youtube_id_list[-1]	

for yt_id in easy_youtube_list:

	data = {}

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

	#check whether file has been added to database
	#process audio
	print('Processing audio...')
	audio_features, time_labels = af.calc_audio_features(yt_id)
	print('%d new rows to add' % len(time_labels))

	# #save all data to the SQL Database
	print('Sending to database...')
	add_audio_features(yt_id, time_labels, audio_features)
