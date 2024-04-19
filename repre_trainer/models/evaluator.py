import os
import torch
import pickle
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict
from omegaconf import DictConfig

from utils.registry import Registry
from loguru import logger

EVALUATOR = Registry('Evaluator')

def create_evaluator(cfg: DictConfig) -> nn.Module:
    """ Create a evaluator for quantitative evaluation
    Args:
        cfg: configuration object
    
    Return:
        A evaluator
    """
    return EVALUATOR.get(cfg.name)(cfg)

@EVALUATOR.register()
class RewardRanker():
    def __init__(self, cfg: DictConfig) -> None:
        """ Evaluator class for reward ranking test
        
        Args:
            cfg: evaluator configuration
        """
        self.log_step = cfg.log_step

    