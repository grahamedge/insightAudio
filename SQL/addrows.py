'''
Basic functionality for adding different types of data to PostgreSQL database
'''

#Built in libraries
from datetime import datetime

#Third party libraries
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import psycopg2
import pandas as pd

#PostgreSQL settings
import settings 	#for locally hosted database
import settingsAWS	#for Amazon RDS

#Homebuilt
from models import AudioInfo, VideoSummary


#This should really go in an __init__ file elsewhere!
engine = create_engine(settingsAWS.DATABASE_URI, echo=False)
Session = sessionmaker(bind=engine)


#Functions
#---------
def create_audio_row(yt_id, time, features):
	'''it is occasionally convenient to add only the features of a processed
	audio file to the database, without performing any cluster analysis,
	so this function produces rows with only audio features'''
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
	'''Creates full rows with all audio features and 
	cluster labels to be added to the database'''
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
	'''Creates rows for the second database storing only the sumary statistics
	for each audio file'''
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
	'''Add a single row (single timestep) of audio information to the data base'''
	
	dbsession = Session()

	row = create_audio_row(yt_id, time, features)
	dbsession.add(row)
	dbsession.commit()	

	dbsession.close()

def add_audio_features(yt_id, times, feature_matrix):
	'''Given a list of times and the associated features for the full audio file,
	this function adds them all to the database using repeated SQL operations'''

	dbsession = Session()

	#First remove any previous versions of audio features from this file
	sql_query = text("DELETE FROM audio_features WHERE youtube_id = :yt_id")
	with engine.connect() as con:
		con.execute(sql_query, yt_id = yt_id)

	#Then add the newly calculated features
	for idx, time in enumerate(times):
		row = create_audio_row(yt_id, time, feature_matrix[:,idx])
		# add_audio_row(yt_id, time, audio_features[:,idx])
		dbsession.add(row)
	dbsession.commit()	

	dbsession.close()	

def add_cluster_labels(yt_id, times, feature_matrix, cluster_labels):
	'''Audio features may have been added to the database already without
	any labels from the discovered clusters. This function removes any such partial data
	and then saves the full information for time, features, and cluster_label into 
	the database.

	Deleting the existing info and re-adding is faster than adding the ~3000 rows
	one-by-one and having to match the time column each time. By adding full rows
	we avoid the matching step.'''

	dbsession = Session()

	#Remove all the audio data for the specified youtube file
	sql_query = text("DELETE FROM audio_features WHERE youtube_id = :yt_id")

	with engine.connect() as con:
	    con.execute(sql_query, yt_id = yt_id)

	#Then insert it again with all columns already matched in order
	for idx, time in enumerate(times):
		row = create_complete_row(yt_id, time, feature_matrix[:,idx], cluster_labels[idx])
		dbsession.add(row)
	dbsession.commit()

	dbsession.close()

def add_results(yt_id, results_dict):
	'''Adds the summary statistics stored in "results_dict" to the
	results database'''

	dbsession = Session()

	row = create_results_row(yt_id, results_dict)
	dbsession.add(row)
	dbsession.commit()

	dbsession.close()