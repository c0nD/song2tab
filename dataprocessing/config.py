import os
import torch

# Get the directory where config.py is located
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(CONFIG_DIR, 'cache', 'guitarset')
PROJECT_ROOT = os.path.join(CONFIG_DIR, '..', '..')
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, 'song2tab', 'dataprocessing', 'data')
BASE_DATA_DIR = os.path.normpath(BASE_DATA_DIR)

# Define other constants and configurations
HOP_LENGTH = 512
SAMPLE_RATE = 22050
BATCH_SIZE = 48
NUM_EPOCHS = 12
NUM_FRAMES = 200
ITERATIONS = 1250
NUM_FOLDS = 4
CHECKPOINTS = 50
LEARNING_RATE = 0.001
SEED = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define paths to specific directories if needed
ANNOTATION_DIR = os.path.join(BASE_DATA_DIR, 'annotation')
AUDIO_DIR = os.path.join(BASE_DATA_DIR, 'audio_mono-mic')