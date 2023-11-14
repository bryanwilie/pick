import torch
import random
import numpy as np


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True
    

def duplicate_samples(list_of_text, dup_factor):
    text_duplicated = []
    for text in list_of_text:
        text_duplicated.extend([text]*dup_factor)
    return text_duplicated