import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from repres.base.base_repre import BaseRepre

class AG2MANIP(BaseRepre):

    def __init__(self, cfg_repre) -> None:
        super(AG2MANIP, self).__init__()
        self.goal_image = cfg_repre["goal_image"]
        self.device = cfg_repre["device"]
        self.batchsize = cfg_repre["batchsize"]

        self.d_emb = cfg_repre["d_emb"]
        self.backbone_type = cfg_repre['backbone_type']
        self.similarity_type = cfg_repre['similarity_type']

        if self.goal_image.dtype != torch.float32:
            raise TypeError("cfg_repre.goal_image.dtype must be torch.float32")
            self.goal_image = torch.tensor(self.goal_image, dtype=torch.float32)
        
        self.normlayer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.backbone_type == 'resnet50':
            self.backbone = torchvision.models.resnet50(pretrained=False)
            self.backbone.fc = nn.Linear(2048, self.d_emb)
        else:
            raise NotImplementedError
        
        #* load pre-trained ckpts
        if cfg_repre['ckpt_dir']:
            print(f'Require a pre-trained ckpt dir for representation model {self.__class__.__name__}')
        self.ckpt_dir = cfg_repre['ckpt_dir']
        print(f'Loading ckpt from {self.ckpt_dir}')
        checkpoint = torch.load(os.path.join(self.ckpt_dir, 'model.pth'))['model']
        self.load_state_dict(checkpoint)
        self.to(self.device)
        self.eval()

        #* compute goal image embedding
        self.goal_image = self.goal_image.to(self.device)
        self.goal_emb = self.embedding(self.goal_image.unsqueeze(0).permute(0, 3, 1, 2)) # (1, 1000)

    @torch.no_grad()
    def forward(self, x):
        """
            x: [to torch.float32] (batch_size, 256, 256, 3)
        """
        x = x.to(self.device)
        if x.dtype != torch.float32:
            raise TypeError("x.dtype must be torch.float32")

        x = x.permute(0, 3, 1, 2) # (batch_size, 3, 256, 256)
        embs = []
        for i in range(0, x.shape[0], self.batchsize):
            embs.append(self.embedding(x[i:i+self.batchsize]))
        embs = torch.cat(embs, dim=0)
        value = self.similarity(embs, self.goal_emb)

        return value
    
    @torch.no_grad()
    def embedding(self, imgs: torch.Tensor) -> torch.Tensor:
        """ Embedding function
        """
        if imgs.shape[1:] != (3, 256, 256):
            preprocess = nn.Sequential(
                transforms.Resize(256, antialias=True),
                self.normlayer,
            )
        else:
            preprocess = nn.Sequential(
                self.normlayer,
            )
        imgs = preprocess(imgs)
        embs = self.backbone(imgs)
        return embs

    def similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Similarity function #! nagative similarity 
        """
        if self.similarity_type == 'l2':
            d = -torch.linalg.norm(x - y, dim=-1)
            return -d
        elif self.similarity_type == 'cosine':
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
            d = torch.einsum('...i,...i->...', x, y)
            return -d
        else:
            raise NotImplementedError