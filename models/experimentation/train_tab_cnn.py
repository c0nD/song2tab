import torch
from torch.utils.data import DataLoader
from torch.optim import Adadelta
import os, sys

from amt_tools.datasets import GuitarSet
from amt_tools.models import TabCNN
from amt_tools.features import CQT
from amt_tools.train import train
from amt_tools.transcribe import ComboEstimator, TablatureWrapper, StackedMultiPitchCollapser
from amt_tools.evaluate import ComboEvaluator, LossWrapper, MultipitchEvaluator, TablatureEvaluator, SoftmaxAccuracy
from amt_tools.evaluate import validate, append_results, average_results
from amt_tools.tools import GuitarProfile, seed_everything

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dataprocessing.config import SAMPLE_RATE, BATCH_SIZE, HOP_LENGTH, NUM_FRAMES, ITERATIONS, CHECKPOINTS, GPU_ID, SEED, NUM_FOLDS, CACHE_DIR, BASE_DATA_DIR
from models.production.basic_cnn import BasicCNN

# Set the root directory for saving the experiment files
ROOT_DIR = 'experiments/tab_cnn'
os.makedirs(ROOT_DIR, exist_ok=True)

# Initialize the default guitar profile
profile = GuitarProfile(num_frets=19)

# Create a CQT feature extraction module
data_proc = CQT(sample_rate=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                n_bins=192,
                bins_per_octave=24)

# Initialize the estimation and evaluation pipelines
validation_estimator = ComboEstimator([
    TablatureWrapper(profile=profile),
    StackedMultiPitchCollapser(profile=profile)
])

validation_evaluator = ComboEvaluator([
    LossWrapper(),
    MultipitchEvaluator(),
    TablatureEvaluator(profile=profile),
    SoftmaxAccuracy()
])

# Set the cache directory
gset_cache = CACHE_DIR + '/guitarset'

# Function to run cross-validation training
def run_training():
    # Seed everything
    seed_everything(SEED)

    # Initialize results dictionary
    results = dict()

    for k in range(NUM_FOLDS):
        # Define training and testing splits
        train_splits = GuitarSet.available_splits()
        test_splits = [train_splits.pop(k)]

        # Load the training dataset
        gset_train = GuitarSet(base_dir=BASE_DATA_DIR,
                               splits=train_splits,
                               hop_length=HOP_LENGTH,
                               sample_rate=SAMPLE_RATE,
                               num_frames=NUM_FRAMES,
                               data_proc=data_proc,
                               profile=profile,
                               reset_data=SEED==0,  # Reset data only on first fold if needed
                               save_loc=gset_cache)

        train_loader = DataLoader(dataset=gset_train,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)

        # Load the testing dataset
        gset_test = GuitarSet(base_dir=BASE_DATA_DIR,
                              splits=test_splits,
                              hop_length=HOP_LENGTH,
                              sample_rate=SAMPLE_RATE,
                              num_frames=None,
                              data_proc=data_proc,
                              profile=profile,
                              store_data=True)

        # Initialize the TabCNN model
        tabcnn = TabCNN(dim_in=data_proc.get_feature_size(),
                        profile=profile,
                        in_channels=data_proc.get_num_channels(),
                        device=GPU_ID)
        tabcnn.change_device(GPU_ID)
        tabcnn.train()

        # Define the optimizer
        optimizer = Adadelta(tabcnn.parameters(), lr=1.0)

        # Training the model
        model_dir = os.path.join(ROOT_DIR, 'models', f'fold-{k}')
        tabcnn = train(model=tabcnn,
                       train_loader=train_loader,
                       optimizer=optimizer,
                       iterations=ITERATIONS,
                       checkpoints=CHECKPOINTS,
                       log_dir=model_dir,
                       val_set=gset_test,
                       estimator=validation_estimator,
                       evaluator=validation_evaluator,
                       resume=True,
                       )

        # Evaluate the model
        validation_evaluator.set_save_dir(os.path.join(ROOT_DIR, 'results'))
        fold_results = validate(tabcnn, gset_test, evaluator=validation_evaluator, estimator=validation_estimator)
        results = append_results(results, fold_results)
        validation_evaluator.reset_results()

    # Log the average results
    overall_results = average_results(results)
    print('Overall Results:', overall_results)

# Run the training
run_training()
