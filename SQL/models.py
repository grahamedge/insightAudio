'''
Base classes for two PostgreSQL databases, which respectively
store the second-by-second audio features and the summary statistics
of the processed youtube files
'''

import settings
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, DateTime, String, Float, Time

engine = create_engine(settings.DATABASE_URI, echo=True)
Session = sessionmaker(bind=engine)
dbsession = Session()

Base = declarative_base()

class AudioInfo(Base):
    __tablename__ = 'audio_features'

    youtube_id        = Column(String, primary_key=True)
    time 		      = Column(Float, primary_key=True)
    #audio features
    zero_crossing	  = Column(Float)
    energy			  = Column(Float)
    energy_entropy	  = Column(Float)
    spectral_centroid = Column(Float)
    spectral_spread	  = Column(Float)
    spectral_entropy  = Column(Float)
    spectral_flux	  = Column(Float)
    spectral_rolloff  = Column(Float)
    mfcc_1			  = Column(Float)
    mfcc_2			  = Column(Float)
    mfcc_3			  = Column(Float)
    mfcc_4			  = Column(Float)
    mfcc_5			  = Column(Float)
    mfcc_6			  = Column(Float)
    mfcc_7			  = Column(Float)
    mfcc_8			  = Column(Float)
    mfcc_9			  = Column(Float)
    mfcc_10			  = Column(Float)
    mfcc_11			  = Column(Float)
    mfcc_12			  = Column(Float)
    mfcc_13			  = Column(Float)
    chroma_1		  = Column(Float)
    chroma_2		  = Column(Float)
    chroma_3		  = Column(Float)
    chroma_4		  = Column(Float)
    chroma_5		  = Column(Float)
    chroma_6		  = Column(Float)
    chroma_7		  = Column(Float)
    chroma_8		  = Column(Float)
    chroma_9		  = Column(Float)
    chroma_10		  = Column(Float)
    chroma_11		  = Column(Float)
    chroma_12		  = Column(Float)
    chroma_dev		  = Column(Float)
    #clustering algorithm output
    cluster_label_raw = Column(Integer)
    cluster_label_filtered = Column(Integer)

class VideoSummary(Base):
    __tablename__ = 'video_summary'

    youtube_id        	= Column(String, primary_key=True)
    title 		      	= Column(String)
    teacher_name	  	= Column(String)
    video_length 		= Column(Float)
    class_grade		  	= Column(Integer)
    class_subject	  	= Column(String)
    teacher_talk_ratio 	= Column(Float)
    n_interactions 	  	= Column(Integer)
    n_short_interactions = Column(Integer)
    n_long_interactions = Column(Integer)
    question_start_a = Column(Integer)
    question_start_b = Column(Integer)
    question_start_c = Column(Integer)
    question_stop_a = Column(Integer)
    question_stop_b = Column(Integer)
    question_stop_c = Column(Integer)
    long_question_start_a = Column(Integer)
    long_question_start_b = Column(Integer)
    long_question_stop_a = Column(Integer)
    long_question_stop_b = Column(Integer)
    cluster_confidence 	= Column(Float)
      
