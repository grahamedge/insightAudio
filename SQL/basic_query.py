'''
Test script to play around with basic SQL queries and check database connections
'''

#Third party libraries
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
import pandas as pd
import numpy as np
import psycopg2

#Database settings
import settingsAWS


engine = create_engine(settingsAWS.DATABASE_URI, echo=False)
Session = sessionmaker(bind=engine)
dbsession = Session()

# table_name = "audio_features"
yt_id = 'dqPjgQwoXLQ'

with engine.connect() as con:
	# query = "SELECT count(*) FROM %(table)s;"
	query = text("SELECT count(*) FROM audio_features WHERE youtube_id = :yt_id")
	rs = con.execute(query, yt_id = yt_id)
	for row in rs:
		print row