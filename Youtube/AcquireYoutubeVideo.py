from __future__ import unicode_literals


import youtube_dl
import ffmpy

'''
NOTES

Currently this downloads both audio and video, packages them into .mkv, and then saves

But later we just extract the audio as .wav, and will discard the video

Thus we should just download audio only, and convert to .wav inside of youtube-dl,
	shortening the script and reducing the disk space / bandwidth used

'''

#Get file info
#-------------------
link_id = 'oEQyAuz_hzs'
video_save_loc = '/media/graham/OS/Linux Content/Youtube/Videos/' 	#also subtitle loc
audio_save_loc = '/media/graham/OS/Linux Content/Youtube/Audio/'

#Download video file
#-------------------
save_format = video_save_loc+'%(id)s.%(ext)s'
ydl_opts = {
    'outtmpl': save_format,
    'merge_output_format': 'mkv',
    'writeautomaticsub': True
    }
ydl = youtube_dl.YoutubeDL(ydl_opts)

with ydl:
    link = 'https://www.youtube.com/watch?v='+link_id
    result = ydl.extract_info(
        link,
        download=True # We just want to extract the info
    )

if 'entries' in result:
    # Can be a playlist or a list of videos
    video = result['entries'][0]
else:
    # Just a video
    video = result

video_url = video['title']
print('\nVideo title is %s' % video_url)


#Extract audio and save to .wav file
in_file = video_save_loc + link_id + '.mkv'
out_file = audio_save_loc + link_id + '.wav'
print('Saving audio to %s' % out_file)
ff = ffmpy.FFmpeg(
    inputs={in_file: None},
    outputs={out_file: None}
    )
ff.run()