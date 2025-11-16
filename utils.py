import random
import numpy as np
import torch
import argparse
import os
import ast


def set_seed(seed=42):
    """
    Set random seed
    """
    random.seed(seed)         
    np.random.seed(seed)         
    torch.manual_seed(seed)         

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)             
        torch.cuda.manual_seed_all(seed)         
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def unnormalize(img_tensor, device):
    """Unnormalize the CLIP normalized image tensor to the original pixel space"""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    return img_tensor * std + mean

def str2bool(v):
    """Convert string to boolean value"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(v):
    """Convert string to list"""
    if isinstance(v, list):
        return v
    try:
        # try to parse the list using ast.literal_eval safely
        return ast.literal_eval(v)
    except:
        # if parsing fails, try to split by comma and convert to integers
        try:
            return [int(x.strip()) for x in v.split(',')]
        except:
            raise argparse.ArgumentTypeError('List value expected.')