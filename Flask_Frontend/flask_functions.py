'''
Some additional functions that are useful to call in views.py

Graham Edge
June 22 2017
'''

#third party libraries
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import psycopg2
import pandas as pd

#import database info
from SQL import settingsAWS

engine = create_engine(settingsAWS.DATABASE_URI, echo=False)
Session = sessionmaker(bind=engine)
dbsession = Session()

# user = 'graham'
# host = 'localhost'
# dbname = 'teaching_videos'
# db = create_engine('postgres://%s%s%s' % (user, host, dbname))
# con = None
# con = psycopg2.connect(database = dbname, user = user)

def empty_results_dict():
	#find an example youtube file in the database
	sql_query = "SELECT youtube_id from video_summary"
	with engine.connect() as con:
		query_results = pd.read_sql_query(sql_query, con)
	youtube_id_dict = query_results.to_dict(orient = 'list')
	sample_yt_id = youtube_id_dict['youtube_id'][0]

	#get the data format for this
	sql_query = "SELECT * from video_summary WHERE youtube_id = %(yt_id)s"
	data = {'yt_id': sample_yt_id}	
	with engine.connect() as con:
		query_results = pd.read_sql_query(sql_query, con, params = data)
	results_dict = query_results.to_dict(orient = 'list')

	#empty the fields to produce a similar, but data-empty dictionary
	for key in results_dict:
		results_dict[key] = [0]

	return results_dict