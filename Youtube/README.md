# Details

Some basic scripts to download Youtube files based on a list of provided Youtube IDs. The command line program youtube-dl is used to automate the downloads, and is imported from within python as youtube_dl.

To convert the audio data into the .wav format suitable for further analysis, the command line program ffmpeg is used. It can be called from within python using the ffmpy package.

It is often also possible to download Youtube's automatically generated closed captions, which represent a rich source of information for analysis. The caption data can be acquired by youtube-dl, and is often in the .vtt format. Functions have also been written to convert the .vtt file into more straightforward strings and timestamps with the help of the package pycaption.