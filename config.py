import numpy as np
# CONFIGURATION VARIABLES & USAGE
# These variables are used to configure the data, model, and training process.
# After modifying these variables, save and run this file to determine if the variables are valid.
# Then, run "data".ipynb to generate the new dataset
# Finally, run one of the ipynb files in the "models" folder to train a model.

# Decision Patch Width: Corresponds to the width of the decision region and label subsamples.
decision_width = 8

# Context Patch Width: Corresponds to the width of input subsamples. Must be greater than or equal to decision_width.
context_width = int(1.5 * decision_width)

# Samples: Number of patches used for training. Samples are evenly selected from each image, and patches are randomly selected from images.
samples = 2000
batch_size = 32

# Num Epochs: Number of epochs for training.
num_epochs = 100

# VGG-16 Model Architecture
# Convolutional Layers: Number of filters in each convolutional layer.
convlayers = [64, 128]

# Fully Connected Layers: Number of neurons in each fully connected layer.
netlayers = [256]

# U-Net Model Architecture
enclayers = [64, 128, 256]

if context_width // (2**(len(convlayers))) == 0:
    raise AttributeError("context_width too small for model architecture")

# Imports
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
import rasterio as rio

# Constants

classNames = {
    0: 'Background',
    1: 'Seagrass',
    2: 'Coral',
    3: 'Macroalgae',
    4: 'Sand',
    5: 'Land',
    6: 'Ocean'
}

classColors = {
    0: (0, 0, 0),
    1: (134, 164, 117),
    2: (255, 127, 80),
    3: (101, 138, 42),
    4: (203, 189, 147),
    5: (139, 98, 76),
    6: (127, 205, 255),
}

classColorsNormalized = { x: (r / 255, g / 255, b / 255) for x, (r, g, b) in classColors.items() }

def show_class_colors():
    # shows a legend containing the class colors and names
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    for i, (class_name, color) in enumerate(classColors.items()):
        plt.subplot(1, len(classColors), i + 1)
        plt.imshow(np.zeros((1, 1, 3), dtype=np.uint8) + np.array(color) / 255)
        plt.axis('off')
        plt.title(classNames[class_name], fontsize=12) 
    plt.tight_layout()
    plt.show()
