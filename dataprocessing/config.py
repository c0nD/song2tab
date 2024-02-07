import os
import torch

# Get the directory where config.py is located
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CONFIG_DIR, '..', '..')
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, 'song2tab', 'dataprocessing', 'data')
BASE_DATA_DIR = os.path.normpath(BASE_DATA_DIR)

# Define other constants and configurations
HOP_LENGTH = 512
SAMPLE_RATE = 44100
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define paths to specific directories if needed
ANNOTATION_DIR = os.path.join(BASE_DATA_DIR, 'annotation')
AUDIO_DIR = os.path.join(BASE_DATA_DIR, 'audio_mono-mic')
