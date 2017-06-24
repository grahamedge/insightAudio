# Insight Audio Analysis Project

This is a project to identify different speakers in an audio file - a task known as 'speaker diarization'. Specifically we want to analyse the audio data from teacher demonstration videos that are posted to Youtube, to see if we can detect the differences between 'teacher talking' and 'teacher not talking'. The latter case might correspond to students talking, or might equally well correspond to long periods of silence.

The process followed for the project is:

1. Download youtube videos: each video is specified by its unique Youtube ID such as 'dQw4w9WgXcQ' for the video 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', and the audio files are downloaded and processed using functions in the 'Youtube' directory

2. Convert the audio waveform into useable information: functions in the directory 'Audio' are used to extract meaning from the sound wave using fourier analysis, the Mel Frequency Cepstrum, and cluster analysis

3. Save the audio analysis results in a database: functions in the directory 'SQL' handle the transfer of information to and from a PostgreSQL database hosted on Amazon RDS

4. View the analysis results in a web interface: using Flask a simple web interface has been created to view the audio analysis results along with the original video files, based on the functions contained in the 'Flask_Frontend' directory
