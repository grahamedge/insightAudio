from datetime import datetime
from models import AudioInfo

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import psycopg2

import settings

con = None
con = psycopg2.connect(database = settings.DATABASE['NAME'], user = settings.DATABASE['USER'])

def get_table_info():
	sql_query = "SELECT column_name FROM information_schema.columns WHERE table_name = 'audio_features';"
	query_results = pd.read_sql_query(sql_query, con)

	print query_results	

def load_audio_data(yt_id):

	sql_query = "SELECT * FROM audio_features WHERE youtube_id = %(yt_id)s ORDER BY audio_features.time;" # Note: no quotes
	data = {'yt_id': yt_id}

	query_results = pd.read_sql_query(sql_query, con, params = data)

	return query_results

def load_intensity(yt_id):

	sql_query = "SELECT zero_crossing FROM audio_features WHERE youtube_id = %(yt_id)s ORDER BY audio_features.time;" # Note: no quotes
	data = {'yt_id': yt_id}

	query_results = pd.read_sql_query(sql_query, con, params = data)

	return query_results	

def load_cluster_labels(yt_id):

	sql_query = "SELECT audio_features.time, cluster_label_raw FROM audio_features WHERE youtube_id = %(yt_id)s ORDER BY audio_features.time;" # Note: no quotes
	data = {'yt_id': yt_id}

	query_results = pd.read_sql_query(sql_query, con, params = data)

	return query_results

def load_a_cluster_label(yt_id, time):

	sql_query = "SELECT audio_features.time, cluster_label_raw FROM audio_features WHERE youtube_id = %(yt_id)s AND ABS(audio_features.time - %(t)s) < 0.1;" # Note: no quotes
	data = {'yt_id': yt_id, 't': time}

	query_results = pd.read_sql_query(sql_query, con, params = data)

	return query_results

def load_video_summary(yt_id):
	sql_query = "SELECT * from video_summary WHERE youtube_id = %(yt_id)s "
	data = {'yt_id': yt_id}	

	query_results = pd.read_sql_query(sql_query, con, params = data)
	return query_results

def add_text_row(yt_id, time, text, length, wps):
	row = create_text_row(yt_id, time, text, length, wps)
	dbsession.add(row)
	dbsession.commit()	