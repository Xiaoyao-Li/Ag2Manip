from typing import Dict, List
import torch.nn as nn
from omegaconf import DictConfig

from utils.registry import Registry

MODEL = Registry('Model')

def create_model(cfg: DictConfig, *args: List, **kwargs: Dict) -> nn.Module:
    """ Create a `torch.nn.Module` object from configuration.

    Args:
        cfg: configuration object, model configuration
        args: arguments to initialize the model
        kwargs: keyword arguments to initialize the model
    
    Return:
        A Module object that has loaded the designated model.
    """
    return MODEL.get(cfg.model.name)(cfg.model, *args, **kwargs)
