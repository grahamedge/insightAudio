# Details

These functions are used to convert raw .wav audio into useful features, perform unsupervised clustering on those features, visualize the results of the clustering, and calculate some summary statistics about the audio waveforms.

The most important files are:

- calc_audio_features: basic audio processing from .wav files and calculation of the Mel Frequency Ceptral Coefficients (MFCCs)

- cluster_audio_features: loading of the MFCCs, and speaker identification using hierarchical clustering

- summarize_cluster_labels: calculation of summary statistics based on the clustering of an audio file

- visualize_cluster_labels: functions for the visualization of the detected clusters in the time domain, as well as in the space of the MFCC features