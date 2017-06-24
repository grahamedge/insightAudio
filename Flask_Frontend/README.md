# Details

The Flask functions and html templates that produce a simple web interface for the project. It is too computationally intensive to download and process long audio files on-the-fly, and so all of the data that can be visualized in the web interface has been pre-processed. The raw audio files are not accessible to the web application, only the second-by-second audio data (in the audio_features database) and the summary statistics for each video (in the video_summary database) are available. These data are stored on a PostgreSQL database on Amazon RDS.

Once one of the pre-processed Youtube files has been selected, the audio features and summary statistics are accessed by Flask by using the SQL Query functions in the SQL directory. Once the data are accessed, visualizations of the audio waveform may be produced using functions from the Audio directory.

To display the actual Youtube file along with the audio classification results, an iframe is used to embed the video from Youtube inside the web interface. Manipulation of the HTML query strings used to embed the video allow for control over the start and stop times of the video.