import torch
from torch.utils.data import DataLoader
from torch.optim import Adadelta
import os, sys
from amt_tools.datasets import GuitarSet
from amt_tools.models import TabCNN
from amt_tools.features import CQT, VQT, WaveformWrapper
from amt_tools.train import train
from amt_tools.transcribe import ComboEstimator, TablatureWrapper, StackedMultiPitchCollapser
from amt_tools.evaluate import ComboEvaluator, LossWrapper, MultipitchEvaluator, TablatureEvaluator, SoftmaxAccuracy
from amt_tools.evaluate import validate, append_results, average_results
from amt_tools.tools import GuitarProfile, seed_everything
import numpy as np
import librosa


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


def load_model(model_path, device='cpu'):
    """
    Load the trained TabCNN model from a specified .pt file
    Loads the model as full object, not just the state_dict
    """
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model
    

def process_audio(audio_path, model, device='cpu'):
    """
    Process an audio file with the TabCNN model to get tablature predictions.
    """
    # Load the audio file
    
    audio, sr = librosa.load(audio_path, sr=22050)
    
    data_proc = CQT(sample_rate=SAMPLE_RATE,
                    hop_length=HOP_LENGTH,
                    n_bins=192,
                    bins_per_octave=24)
    
    feats = data_proc.process_audio(audio)
    
    selected_frames = feats[:, :9]  # Select the first 9 frames
    
    feats_tensor = torch.tensor(selected_frames, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        results = model(feats_tensor)
        
    return results
    
    

def main():
    model_path = 'Z:/song2tab/models/experimentation/experiments/tab_cnn/models/fold-3/model-1250.pt'
    audio_path = '00_BN1-129-Eb_comp_hex_cln.wav'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, device=device)
    results = process_audio(audio_path, model, device=device)
    
    print(results)
    
    
if __name__ == '__main__':
    main()