import torch
CONFIG = {
    "learning_rate": 1e-4,
    "batch_size": 1,
    "num_epochs": 100,
    "train_set_size": 0.9,
    "image_height": 512,
    "image_width": 512,
    "image_depth": 60,
    "keep_background_fraction": 0.1, # Fraction of background slices to keep, only used in 2D

    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "input_channels": 1,
    "output_channels": 3,
    "feature_sizes": [16, 32, 64, 128, 256],

}


#base_path = "/datasets/tdt4265/mic/open/HNTS-MRG"
# base_path = "/home/mariusliebig/Documents/DYP/HNTS-MRG"
base_path = "/cluster/projects/vc/data/mic/open/HNTS-MRG"
