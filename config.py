CONFIG = {
        "learning_rate": 5e-4,
        "batch_size": 16,
        "num_epochs": 30,
        "train_set_size": 0.8,
        "image_height": 512,
        "image_width": 512,
        "keep_background_fraction": 0.2,
        "early_stop_count": 6,

        "num_classes": 3,
        "weights": [0.05, 0.475, 0.475],
        "dice_weight": 0.5,
        "ce_weight": 0.5,

    }


base_path = "/datasets/tdt4265/mic/open/HNTS-MRG/train"
# base_path = "/home/mariusliebig/Documents/DYP/HNTS-MRG"
# base_path = "/cluster/projects/vc/data/mic/open/HNTS-MRG"


test_path = "/datasets/tdt4265/mic/open/HNTS-MRG/test"
# test_path = "/home/mariusliebig/Documents/DYP/HNTS-MRG/test"
# test_path = "/cluster/projects/vc/data/mic/open/HNTS-MRG/test"