'''
Basic function to load particular kinds of information
from the PostgreSQL database for analysis or plotting
'''

#Built in libraries
from datetime import datetime

#Third party libraries
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import psycopg2

#Homebuilt
from models import AudioInfo

#PostgreSQL settings
import settingsAWS	#database on Amazon RDS
# import settings 	#locally hosted database

#Create the engine and session factory
engine = create_engine(settingsAWS.DATABASE_URI, echo=False)
Session = sessionmaker(bind=engine)

def get_table_info():
	'''returns the column names from the audio_features table'''

	sql_query = "SELECT column_name FROM information_schema.columns WHERE table_name = 'audio_features';"

	with engine.connect() as con:
		query_results = pd.read_sql_query(sql_query, con)

	print query_results	

def load_audio_data(yt_id):
	'''loads all of the audio information from the audio_features
	database, returning all values in a pandas dataframe'''

	sql_query = "SELECT * FROM audio_features WHERE youtube_id = %(yt_id)s ORDER BY audio_features.time;" # Note: no quotes
	data = {'yt_id': yt_id}

	with engine.connect() as con:
		query_results = pd.read_sql_query(sql_query, con, params = data)

	return query_results

def load_intensity(yt_id):
	'''loads a specified feature for all seconds on a particular youtube file
	 - useful if only a little information is desired to generate a plot of the waveform

	 returns the features as a pandas dataframe'''

	sql_query = "SELECT zero_crossing FROM audio_features WHERE youtube_id = %(yt_id)s ORDER BY audio_features.time;" # Note: no quotes
	data = {'yt_id': yt_id}

	with engine.connect() as con:
		query_results = pd.read_sql_query(sql_query, con, params = data)

	return query_results	

def load_cluster_labels(yt_id):
	'''for a specified youtube file, returns all timesteps
	and their associated cluster labels as a pandas dataframe'''

	sql_query = "SELECT audio_features.time, cluster_label_raw FROM audio_features WHERE youtube_id = %(yt_id)s ORDER BY audio_features.time;" # Note: no quotes
	data = {'yt_id': yt_id}

	with engine.connect() as con:
		query_results = pd.read_sql_query(sql_query, con, params = data)

	return query_results

def load_a_cluster_label(yt_id, time):
	'''loads the cluster label corresponding to a specified time 
	in a specified youtube file. time is matched to the nearest
	100ms since it is stored as a float

	returns the matched time and associated cluster label
	as a pandas dataframe'''

	sql_query = "SELECT audio_features.time, cluster_label_raw FROM audio_features WHERE youtube_id = %(yt_id)s AND ABS(audio_features.time - %(t)s) < 0.1;" # Note: no quotes
	data = {'yt_id': yt_id, 't': time}

	with engine.connect() as con:
		query_results = pd.read_sql_query(sql_query, con, params = data)

	return query_results

def load_video_summary(yt_id):
	'''load the summary statistics for a specified video from the summary database

	returns a pandas dataframe'''

	sql_query = "SELECT * from video_summary WHERE youtube_id = %(yt_id)s "
	data = {'yt_id': yt_id}	

	with engine.connect() as con:
		query_results = pd.read_sql_query(sql_query, con, params = data)
	return query_results

def add_text_row(yt_id, time, text, length, wps):
	'''Add rows of text information to the database, based
	on analysis of the video captions'''

	dbsession = Session()

	row = create_text_row(yt_id, time, text, length, wps)
	dbsession.add(row)
	dbsession.commit()	

	dbsession.close()