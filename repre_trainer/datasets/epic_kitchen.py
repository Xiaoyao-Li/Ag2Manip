from typing import Any, Tuple, Dict
import os
import json
import glob
from tqdm import tqdm
import pickle
import trimesh

import torch
import torchvision
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig

from datasets.base import DATASET

@DATASET.register()
class EpicKitchen(Dataset):
    """ Dataset for representation learning, 
        training with EPIC-KITCHEN agent-aware/agent-agonostic dataset
    """
    _STR_FRAME_LENGTH = 10
    _train_split = []
    _test_split = []
    _all_split = []
    
    def __init__(self, cfg: DictConfig, phase: str, slurm: bool, **kwargs: Dict) -> None:
        super(EpicKitchen, self).__init__()
        self.phase = phase
        self.slurm = slurm
        if self.phase == 'train':
            self.split = self._train_split
        elif self.phase == 'test':
            self.split = self._test_split
        elif self.phase == 'all':
            self.split = self._all_split
        else:
            raise Exception(f"Unsupported phase: {self.phase}")
        self.data_type = cfg.data_type
        self.model_type = cfg.model_type
        self.data_dir = cfg.data_dir_slurm if self.slurm else cfg.data_dir_local
        self.resolution = (cfg.resolution_height, cfg.resolution_width)
        self.aug_sidewindow_size = (1 - cfg.aug_window_size) / 2
        self.to_tensor = torchvision.transforms.ToTensor()
        self.preprocess = torch.nn.Sequential(
                    torchvision.transforms.Resize(self.resolution, antialias=True),)
        
        #* for specify getitem func.
        self.item_type = cfg.item_type.lower()
        #* load data
        self._pre_load_data()

    def _pre_load_data(self) -> None:
        """ Load metadata from json files
        """
        # self.indices = []
        self.info = json.load(open(os.path.join(self.data_dir, 'info.json'), 'r'))
        self.metadata = pd.read_csv(os.path.join(self.data_dir, 'EPIC100_annotations.csv'))
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: Tuple) -> Tuple:
        if self.item_type.lower() in ['vip']:
            # return self._getitem_vip(index)
            #! use r3m sample way
            return self._getitem_vip(index)
        elif self.item_type.lower() in ['r3m', 'ag2manip']:
            return self._getitem_r3m(index)
        else:
            raise NotImplementedError

    def _getitem_r3m(self, index: Any) -> Tuple:
        mdata = self.metadata.iloc[index]
        start_frame = mdata['start_frame']
        stop_frame = mdata['stop_frame']
        part_id = mdata['participant_id']
        video_id = mdata['video_id']

        #* do augmentation and observation sampling
        # clip_length = stop_frame - start_frame + 1
        # start_ind = np.random.randint(start_frame,
        #                                 start_frame + int(clip_length * self.aug_sidewindow_size) + 1)
        # stop_ind = np.random.randint(stop_frame - int(clip_length * self.aug_sidewindow_size),
        #                                 stop_frame + 1)
        sample_indices = np.random.permutation(np.arange(start_frame, stop_frame + 1))[:3]
        s0_ind_r3m, s1_ind_r3m, s2_ind_r3m = np.sort(sample_indices)

        img_s0 = self._load_frame(part_id, video_id, s0_ind_r3m)
        img_s1 = self._load_frame(part_id, video_id, s1_ind_r3m)
        img_s2 = self._load_frame(part_id, video_id, s2_ind_r3m)
        imgs = torch.stack([img_s0, img_s1, img_s2], dim=0)
        imgs = self.preprocess(imgs)

        #* dict a data sample
        data = {
            'imgs': imgs,
            's0_ind': s0_ind_r3m,
            's1_ind': s1_ind_r3m,
            's2_ind': s2_ind_r3m
        }
        return data
    
    def _getitem_vip(self, index: Any) -> Tuple:
        mdata = self.metadata.iloc[index]
        #? do random crop???
        start_frame = mdata['start_frame']
        stop_frame = mdata['stop_frame']
        part_id = mdata['participant_id']
        video_id = mdata['video_id']

        #* do augmentation and observation sampling
        clip_length = stop_frame - start_frame + 1
        start_ind = np.random.randint(start_frame, 
                                      start_frame + int(clip_length * self.aug_sidewindow_size) + 1)
        stop_ind = np.random.randint(stop_frame - int(clip_length * self.aug_sidewindow_size), 
                                     stop_frame + 1)
        s0_ind_vip = np.random.randint(start_ind, stop_ind)
        s1_ind_vip = np.random.randint(s0_ind_vip + 1, stop_ind + 1)

        #* load images
        #! should be start_ind and stop_ind
        img_start = self._load_frame(part_id, video_id, start_ind)
        img_goal = self._load_frame(part_id, video_id, stop_ind)
        img_s0 = self._load_frame(part_id, video_id, s0_ind_vip)
        img_s1 = self._load_frame(part_id, video_id, s1_ind_vip)
        imgs = torch.stack([img_start, img_goal, img_s0, img_s1], dim=0)
        imgs = self.preprocess(imgs)

        #* dict a data sample
        data = {
            'imgs': imgs,
            'start_ind': start_ind,
            'stop_ind': stop_ind,
            's0_ind': s0_ind_vip,
            's1_ind': s1_ind_vip,}
        return data

    def _load_frame(self, part_id: str, video_id: str, frame_id: int) -> torch.Tensor:
        if self.data_type == 'rgb':
            vid = os.path.join(self.data_dir, part_id, 'rgb_frames', video_id, f"frame_{frame_id:010d}.jpg")
        elif self.data_type == 'agentago':
            vid = os.path.join(self.data_dir, part_id, 'agentago_frames', video_id, f"frame_{frame_id:010d}.jpg")
        else:
            raise NotImplementedError
        return self.to_tensor(Image.open(vid).convert('RGB'))

    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)
