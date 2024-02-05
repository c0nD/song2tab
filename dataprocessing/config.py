import os

# Define the base directory for the data
BASE_DATA_DIR = 'z:/song2tab/dataprocessing/data'

# Define other constants and configurations
HOP_LENGTH = 512
SAMPLE_RATE = 44100

# define paths to specific directories if needed
ANNOTATION_DIR = os.path.join(BASE_DATA_DIR, 'annotation')
AUDIO_DIR = os.path.join(BASE_DATA_DIR, 'audio_mono-mic')
