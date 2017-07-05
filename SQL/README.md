# Details:

These functions are used to create and interact with two tables in a 
PostgreSQL database. The two tables are 'audio_features' and 'video_summary'.

'audio_features' is meant to store second-by-second audio information for each
Youtube file that is processed. Since two different Youtube videos may (and likely will)
have overlap in terms of the possible times considered, this database uses two primary keys: 
'youtube_id' (string) and 'time' (float) in order to uniquely specify a sample of audio.

'video_summary' is meant to store summary information about each youtube video such as
the length of the video, or the number of times that the speaker switches during the video.
Since there is no second-by-second information stored here, only a single primary key is needed: 'youtube_id'.

#PostgreSQL Details:

Actual PostgreSQL details are withheld here. To run these functions a file 'settingsAWS.py' should be created in the SQL directory with the contents:

```python
DATABASE = {
        'NAME': "database_name",
        'USER': "user_name",
        "PASSWORD": "password_goes_here",
        "HOST": "database_server_address",
        "PORT": "database_port",
    }

DATABASE_URI = 'postgres://%s:%s@%s:%s/%s' % (DATABASE['USER'],
        DATABASE['PASSWORD'],DATABASE['HOST'],DATABASE['PORT'],DATABASE['NAME'])
```

For debugging purposes, two such settings files were used during this project. One pointing to a PostgreSQL database hosted locally on my laptop (settings.py) and one pointing to an Amazon Web Services RDS server (settingsAWS).
