import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dataprocessing.guitarset_dataset import GuitarSetDataset
from dataprocessing.config import BASE_DATA_DIR, DEVICE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
from models.production.basic_cnn import BasicCNN

def train_model():
    npz_files = [os.path.join(BASE_DATA_DIR, 'processed', filename)
                 for filename in os.listdir(os.path.join(BASE_DATA_DIR, 'processed'))
                 if filename.endswith('.npz')]

    dataset = GuitarSetDataset(npz_files=npz_files)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = DEVICE  # will load GPU if available

    model = BasicCNN(num_pitches=121, num_strings=6, model_complexity=1, frame_width=1, device=device)
    model.change_device(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for inputs, labels in dataloader:
            # Give data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero param gradients
            optimizer.zero_grad()
            
            # Pre-process inputs using model's pre_proc (if any pre-processing is needed)
            batch = {'input': inputs}
            batch = model.pre_proc(batch)
            
            # Forward pass
            outputs = model(batch['input'])
            
            # Post-process outputs using model's post_proc (if any post-processing is needed)
            batch['output'] = outputs
            batch['tablature'] = labels
            batch = model.post_proc(batch)
            
            loss = batch['loss']
            
            # Backward pass + optimize
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if (i + 1) % 100 == 0:  # prints every 100 mini-batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}: Loss {running_loss / 100:.4f}')
                running_loss = 0.0

    print('Finished Training')

if __name__ == '__main__':
    train_model()