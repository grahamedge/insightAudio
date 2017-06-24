#third party libraries
import pandas as pd
import psycopg2
import mpld3

#third party functions
from flask import render_template
from flask import request
from Flask_Frontend import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

#database info
from SQL import settingsAWS

#homebuilt functions
from Audio.summarize_cluster_labels import get_minute_labels, plot_clustered_waveform_html
import flask_functions as ff


#Create database engine
engine = create_engine(settingsAWS.DATABASE_URI, echo=False)
Session = sessionmaker(bind=engine)
dbsession = Session()

# user = 'graham'
# host = 'localhost'
# dbname = 'teaching_videos'
# db = create_engine('postgres://%s%s%s' % (user, host, dbname))
# con = None
# con = psycopg2.connect(database = dbname, user = user)

@app.route('/')
@app.route('/index')
def index():
	return render_template('video_input.html')

@app.route('/video_output')
def serve_video_features():
	
	# #get the list of all processed youtube videos
	# sql_query = "SELECT youtube_id from video_summary"
	# query_results = pd.read_sql_query(sql_query, con)
	# youtube_id_dict = query_results.to_dict(orient = 'list')
	
	#check for one video that is selected:
	try:
		yt_id = request.args.get('yt_id')
		sql_query = "SELECT * from video_summary WHERE youtube_id = %(yt_id)s"
		data = {'yt_id': str(yt_id)}	
		with engine.connect() as con:
			query_results = pd.read_sql_query(sql_query, con, params = data)
		if query_results.empty:
			empty_dict = True
			results_dict = ff.empty_results_dict()
		else:
			empty_dict = False
			results_dict = query_results.to_dict(orient = 'list')
	except:
		#error retrieving video data from database...
		#	most likely the video has not been processed!
		empty_dict = True
		results_dict = ff.empty_results_dict()

	#get the parameters of the selected video
	if empty_dict:
		for key in results_dict.keys():
			results_dict[key] = 0
	else:
		for key in results_dict.keys():
			results_dict[key] = results_dict[key][0]

	try:
		t_start = int(request.args.get('t_start'))
		try:
			t_stop = int(request.args.get('t_stop'))
			embed_url = ('https://www.youtube.com/embed/'+str(yt_id) +
						'?start=' + str(t_start) + 
						'&end=' + str(t_stop) + 
						'&autoplay=1')
		except:
			embed_url = ('https://www.youtube.com/embed/'+str(yt_id) +
						'?start=' + str(t_start) + 
						'&autoplay=1')
			t_stop = 0
	except:
		embed_url = 'https://www.youtube.com/embed/'+str(yt_id)
		t_start = 0
		t_stop = 0

	fig_html = plot_clustered_waveform_html(yt_id)


	#could also have the plot cue up when the short interactions
	#	are clicked on!!!

	#return a rendered html with the dictionary as an input
	return render_template('summary_table.html', 
					 results = results_dict, url = embed_url,
					 t_start = t_start, name = 'graham',
					 fig_html = fig_html)

@app.route('/bonus_videos')
def serve_bonus_video():
	
	#get the list of all processed youtube videos
	sql_query = "SELECT youtube_id from video_summary"
	with engine.connect() as con:
		query_results = pd.read_sql_query(sql_query, con)
	youtube_id_dict = query_results.to_dict(orient = 'list')
	
	#check for one video that is selected:
	try:
		yt_id = request.args.get('yt_id')

		sql_query = "SELECT * from video_summary WHERE youtube_id = %(yt_id)s"
		data = {'yt_id': yt_id}	
		with engine.connect() as con:
			query_results = pd.read_sql_query(sql_query, con, params = data)
		results_dict = query_results.to_dict(orient = 'list')
	except:
		results_dict = empty_results_dict()

	embed_url = 'https://www.youtube.com/embed/'+str(yt_id)

	fig_html = plot_clustered_waveform(yt_id)

	#return a rendered html with the dictionary as an input
	return render_template('bonus_videos.html', yt_ids = youtube_id_dict,
					 url = embed_url,
					 name = 'graham',
					 fig_html = fig_html)	