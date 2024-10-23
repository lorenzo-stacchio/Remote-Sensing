from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import torchvision
import torch
import tqdm
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed):
    """
    Set the seed for reproducibility.

    Args:
    seed (int): The seed to set.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)

    # Set seed for CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Ensure that CUDA operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

