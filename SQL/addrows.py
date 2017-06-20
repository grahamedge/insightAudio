from datetime import datetime
from models import AudioInfo, VideoSummary

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import psycopg2
import pandas as pd
import settings

import timeit

# engine = create_engine(settings.DATABASE_URI, echo=False)
# Session = sessionmaker(bind=engine)
# dbsession = Session()

user = 'graham'
host = 'localhost'
dbname = 'teaching_videos'
#db = create_engine('postgres://%s%s%s' % (user, host, dbname))

engine = create_engine(settings.DATABASE_URI, echo=False)
Session = sessionmaker(bind=engine)
dbsession = Session()

con = None
con = psycopg2.connect(database = dbname, user = user)
cur = con.cursor()

def create_audio_row(yt_id, time, features):
	row = AudioInfo(
        youtube_id = yt_id,
        time = time,
        zero_crossing = features[0],
        energy = features[1],
        energy_entropy = features[2],
        spectral_centroid = features[3], 
        spectral_spread = features[4],
        spectral_entropy = features[5],
        spectral_flux	  = features[6],
    	spectral_rolloff  = features[7],
    	mfcc_1 			  = features[8],
    	mfcc_2			  = features[9],
	    mfcc_3			  = features[10],
	    mfcc_4			  = features[11],
	    mfcc_5			  = features[12],
	    mfcc_6			  = features[13],
	    mfcc_7			  = features[14],
	    mfcc_8			  = features[15],
	    mfcc_9			  = features[16],
	    mfcc_10			  = features[17],
	    mfcc_11			  = features[18],
	    mfcc_12			  = features[19],
	    mfcc_13			  = features[20],
	    chroma_1		  = features[21],
	    chroma_2		  = features[22],
	    chroma_3		  = features[23],
	    chroma_4		  = features[24],
	    chroma_5		  = features[25],
	    chroma_6		  = features[26],
	    chroma_7		  = features[27],
	    chroma_8		  = features[28],
	    chroma_9		  = features[29],
	    chroma_10		  = features[30],
	    chroma_11		  = features[31],
	    chroma_12		  = features[32],
	    chroma_dev		  = features[33] )
	return row

def create_complete_row(yt_id, time, features, cluster_label):
	row = AudioInfo(
        youtube_id = yt_id,
        time = time,
        zero_crossing = features[0],
        energy = features[1],
        energy_entropy = features[2],
        spectral_centroid = features[3], 
        spectral_spread = features[4],
        spectral_entropy = features[5],
        spectral_flux	  = features[6],
    	spectral_rolloff  = features[7],
    	mfcc_1 			  = features[8],
    	mfcc_2			  = features[9],
	    mfcc_3			  = features[10],
	    mfcc_4			  = features[11],
	    mfcc_5			  = features[12],
	    mfcc_6			  = features[13],
	    mfcc_7			  = features[14],
	    mfcc_8			  = features[15],
	    mfcc_9			  = features[16],
	    mfcc_10			  = features[17],
	    mfcc_11			  = features[18],
	    mfcc_12			  = features[19],
	    mfcc_13			  = features[20],
	    chroma_1		  = features[21],
	    chroma_2		  = features[22],
	    chroma_3		  = features[23],
	    chroma_4		  = features[24],
	    chroma_5		  = features[25],
	    chroma_6		  = features[26],
	    chroma_7		  = features[27],
	    chroma_8		  = features[28],
	    chroma_9		  = features[29],
	    chroma_10		  = features[30],
	    chroma_11		  = features[31],
	    chroma_12		  = features[32],
	    chroma_dev		  = features[33],
	    cluster_label_raw = cluster_label )
	return row	

def create_results_row(yt_id, results_dict):
	row = VideoSummary(
        youtube_id = yt_id,
        teacher_talk_ratio = results_dict['teacher_talk_ratio'],
        n_interactions = results_dict['number_of_interactions'],
        video_length = results_dict['video_length'],
        n_short_interactions = results_dict['n_short_interactions'],
        n_long_interactions = results_dict['n_long_interactions'],
        question_start_a = results_dict['question_start_a'],
		question_start_b = results_dict['question_start_b'] ,
		question_start_c = results_dict['question_start_c'] ,
		question_stop_a = results_dict['question_stop_a'],
		question_stop_b = results_dict['question_stop_b'] ,
		question_stop_c = results_dict['question_stop_c'] ,
		long_question_start_a = results_dict['long_question_start_a'],
		long_question_start_b = results_dict['long_question_start_b'],
		long_question_stop_a = results_dict['long_question_stop_a'],
		long_question_stop_b = results_dict['long_question_stop_b']
        )
	return row

def create_cluster_row(yt_id, time, cluster_label, label_type = 'raw'):
	'''adds the cluster label to the database, 
	with different types of labels possible:
		- raw labels
		- filtered versions of the labels
		- possible other types of clustering algorithms'''
	if label_type == 'raw':
		row = AudioInfo(
			youtube_id = yt_id,
			time = time,
			cluster_label_raw = cluster_label)	
	elif label_type == 'filtered':
		row = AudioInfo(
			youtube_id = yt_id,
			time = time,
			cluster_label_filtered = cluster_label)
	else:
		print('Unrecognized cluster label type!')
	return row

def add_audio_row(yt_id, time, features):
	row = create_audio_row(yt_id, time, features)
	dbsession.add(row)
	dbsession.commit()	

def add_audio_features(yt_id, times, feature_matrix):

	for idx, time in enumerate(times):
		row = create_audio_row(yt_id, time, feature_matrix[:,idx])
		# add_audio_row(yt_id, time, audio_features[:,idx])
		dbsession.add(row)
	dbsession.commit()		

def add_cluster_labels(yt_id, times, feature_matrix, cluster_labels):

	#Remove all the audio data for the specified youtube file
	sql_query = "DELETE FROM audio_features WHERE youtube_id = %(yt_id)s"
	data = {'yt_id':yt_id}
	cur.execute(sql_query, data)
	con.commit()

	#Then insert it again with all columns already matched in order
	for idx, time in enumerate(times):
		row = create_complete_row(yt_id, time, feature_matrix[:,idx], cluster_labels[idx])
		dbsession.add(row)
	dbsession.commit()

def add_results(yt_id, results_dict):

	row = create_results_row(yt_id, results_dict)
	dbsession.add(row)
	dbsession.commit()

def add_one_cluster_label(yt_id, time, label):
	sql_query = "UPDATE audio_features SET cluster_label_raw = %(label)s WHERE youtube_id = %(yt_id)s AND ABS(audio_features.time-%(t)s) < 0.1;"
	data = {'yt_id':yt_id, 'label': int(label), 't': time}
	cur.execute(sql_query, data)
	con.commit()