#Third party packages
from SQL.load_rows import load_audio_data
from SQL import settings

from SQL.addrows import add_results, create_results_row
from SQL.load_rows import load_video_summary

#---------
#Run here:
#---------

yt_id = 'y2OFsG6qkBs'

summary_df = load_video_summary(yt_id)

summary_dict = summary_df.to_dict(orient = 'list')
print(summary_dict.keys())

print(summary_dict['n_long_interactions'])