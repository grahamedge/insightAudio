'''
takes a youtube ID and adds the text data into the SQL database using
SQLAlchemy

if the video captions are not downloaded yet, they will be downloaded

if the captions have not been analyzed yet to get metrics like word-per-second,
	this analysis will be called

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
from SQL.addrows import create_audio_row, add_audio_row
from SQL import settings

from Audio import calc_audio_features


#start the connection
engine = create_engine(settings.DATABASE_URI, echo=True)
Session = sessionmaker(bind=engine)
dbsession = Session()

#choose a video
youtube_video_folder 	= '/media/graham/OS/Linux Content/Youtube/Videos/'
youtube_audio_folder 	= '/media/graham/OS/Linux Content/Youtube/Audio/'

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

yt_id_list = youtube_id_list[-1]	

for yt_id in yt_id_list:

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

	#check whether subtitle file exists
	fname = youtube_video_folder + yt_id + '.en.vtt'
	if os.path.isfile(fname):
		text_list, second_list = yt.read_vtt_captions(yt_id)
		#save some video information to the data structure
		wps_list = yt.calc_wps(text_list, second_list)
		length_list = yt.calc_length(second_list)
	else:
		print('No caption file found for ID %s!' % yt_id)

	#Add each row of the text data to the database
	for text, second, length, wps in zip(text_list[0:10], second_list[0:10], length_list[0:10], wps_list[0:10]):
		print((yt_id,second,str(text),length,wps))
		#add_text_row(yt_id, second, text, length, wps)		

	# #check whether .wav has been already created
	# fname = youtube_audio_folder + yt_id + '.wav'
	# if os.path.isfile(fname):
	# 	# .wav already exists
	# 	audio_exists = True
	# else:
	# 	# .wav file not created yet
	# 	audio_exists = yt.convert_to_wav(yt_id, vid_type = container_type, 
	# 		video_loc = youtube_video_folder, 
	# 		audio_loc = youtube_audio_folder)

	#process audio
	audio_features, time_labels = calc_audio_features(wav_file)



	#save all data to the SQL Database
	