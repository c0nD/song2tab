from config import BASE_DATA_DIR, HOP_LENGTH, SAMPLE_RATE
from amt_tools.datasets import GuitarSet
from amt_tools import tools
import os

def process_guitarset(base_dir):
    """
    Process the GuitarSet dataset and prepare it for training.
    
    Args:
        base_dir (str): The path to the GuitarSet directory.
    """
    
    # Define params for processing
    splits = GuitarSet.available_splits()
    profile = tools.GuitarProfile(num_frets=19)
    # Number of consecutive frames within each example fed to the model
    num_frames = 200
    
    guitarset = GuitarSet(base_dir=base_dir,
                          splits=splits,
                          num_frames=num_frames,
                          profile=profile,
                          hop_length=HOP_LENGTH,
                          sample_rate=SAMPLE_RATE)
    
    # Process the tracks
    for split in splits:
        tracks = guitarset.get_tracks(split)
        for track in tracks:
            data = guitarset.load(track)
            
            # Now 'data' should contain the processed data for the given track
            # Here we can handle the processed data, like saving or passing it to your neural network
            # Save the processed data (optional)
            save_processed_data(base_dir, track, data)
    
            
def save_processed_data(base_dir, track, data):
    """
    Save the processed data for a track.
    
    Args:
        base_dir (str): The base directory for the data.
        track (str): The name of the track.
        data (dict): The processed data.
    """
    
    processed_dir = os.path.join(base_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    save_path = os.path.join(processed_dir, f"{track}.npz")
    
    tools.save_dict_npz(save_path, data)
    print(f"Processed data for track '{track}' saved to '{save_path}'.")
    

if __name__ == "__main__":
    process_guitarset(BASE_DATA_DIR)