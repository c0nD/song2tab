import torch
import torch.nn as nn
import torch.nn.functional as F

from amt_tools.models.common import TranscriptionModel


class BasicCNN(TranscriptionModel):
    
    # Basic CNN model
    def __init__(self, dim_in, profile, num_pitches, num_strings=6, model_complexity=1, frame_width=1, device='cpu'):
        # Initialize the TranscriptionModel with the provided parameters
        super(BasicCNN, self).__init__(dim_in, profile, in_channels=1, model_complexity=model_complexity, frame_width=frame_width, device=device)
        
        # Number of classes is the total number of pitches plus one for the silent state
        self.num_classes = num_pitches * num_strings + 1
        
        # Number of filters for each convolutional layer, adjusted by model complexity
        nf1 = 32 * model_complexity
        nf2 = 64 * model_complexity
        nf3 = nf2
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, nf1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(nf1, nf2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(nf2, nf3, kernel_size=3, stride=1, padding=1)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(nf3 * 6 * 6, 1000)
        self.fc2 = nn.Linear(1000, self.num_classes)
        
    
    # Define the forward pass of the network (i.e. how to compute the output from the input)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
    def pre_proc(self, batch):
        return super().pre_proc(batch)
    
    
    def post_proc(self, batch):
        output = batch['output']
        
        # Calc loss if ground truth is given
        if 'tablature' in batch:
            tablature_ref = batch['tablature']
            tablature_est = output
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(tablature_est, tablature_ref)
            output['loss'] = loss
        
        return output