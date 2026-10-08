import torch
import numpy as np
import random
import gc

def reset_env(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    gc.collect()
    torch.cuda.empty_cache()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")