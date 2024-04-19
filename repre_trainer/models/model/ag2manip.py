from typing import Dict
from einops import rearrange
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from omegaconf import DictConfig

from models.base import MODEL

@MODEL.register()
class AG2MANIP(nn.Module):
    # a copy for r3m model architecture
    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        super(AG2MANIP, self).__init__()
        self.d_emb = cfg.d_emb
        self.backbone_type = cfg.backbone_type
        self.similarity_type = cfg.similarity_type
        self.num_negatives = cfg.num_negatives
        self.loss_weight = cfg.loss_weight

        self.normlayer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.backbone_type == 'resnet50':
            self.backbone = torchvision.models.resnet50(pretrained=False)
            self.backbone.fc = nn.Linear(2048, self.d_emb)
        else:
            raise NotImplementedError

    def forward(self, data: Dict) -> torch.Tensor:
        """ Forward
        Args:
            data: input data dict
                {
                'imgs': imgs,  [B, T, C, H, W] (e.g., [32, 3, 3, 256, 256]), must be float32
                's0_ind': s0_ind,
                's1_ind': s1_ind,
                's2_ind': s2_ind}
        return:
            dict {
                'loss': full_loss, 
                'metrics': metrics for logs}
        """
        imgs = data['imgs']
        s0_ind = data['s0_ind']
        s1_ind = data['s1_ind']
        s2_ind = data['s2_ind']

        if imgs.shape[2:] != (3, 256, 256):
            preprocess = nn.Sequential(
                transforms.Resize(256, antialias=True),
                self.normlayer,
            )
        else:
            preprocess = nn.Sequential(
                self.normlayer,
            )
        imgs = preprocess(imgs)
        B, T = imgs.shape[:2]
        imgs = imgs.reshape(B*T, *imgs.shape[2:])
        embs = self.backbone(imgs)
        embs = embs.reshape(B, T, *embs.shape[1:])
        emb_s0 = embs[:, 0]
        emb_s1 = embs[:, 1]
        emb_s2 = embs[:, 2]

        #* compute metrics and full loss
        full_loss = 0
        metrics = dict()

        #* 1. Embdedding Norm loss
        loss_l1 = torch.linalg.norm(embs, ord=1, dim=-1).mean()
        loss_l2 = torch.linalg.norm(embs, ord=2, dim=-1).mean()
        full_loss += self.loss_weight.l1norm * loss_l1
        full_loss += self.loss_weight.l2norm * loss_l2
        metrics['loss_l1'] = loss_l1.item()
        metrics['loss_l2'] = loss_l2.item()

        #* 2. TCN Loss
        sim_0_1 = self.similarity(emb_s0, emb_s1)
        sim_1_2 = self.similarity(emb_s1, emb_s2)
        sim_0_2 = self.similarity(emb_s0, emb_s2)

        # negative samples
        sim_s0_neg = []
        sim_s2_neg = []
        perm = [i for i in range(B)]
        for _ in range(self.num_negatives):
            perm = [(i_perm + 1) % B for i_perm in perm]
            emb_s0_shuf = emb_s0[perm]
            emb_s2_shuf = emb_s2[perm]
            sim_s0_neg.append(self.similarity(emb_s0_shuf, emb_s0))
            sim_s2_neg.append(self.similarity(emb_s2_shuf, emb_s2))
        sim_s0_neg = torch.stack(sim_s0_neg, dim=-1)
        sim_s2_neg = torch.stack(sim_s2_neg, dim=-1)

        tcn_loss_1 = -torch.log(1e-6 + (torch.exp(sim_1_2) / (1e-6 + torch.exp(sim_0_2) + torch.exp(sim_1_2) + torch.exp(sim_s2_neg).sum(-1))))
        tcn_loss_2 = -torch.log(1e-6 + (torch.exp(sim_0_1) / (1e-6 + torch.exp(sim_0_1) + torch.exp(sim_0_2) + torch.exp(sim_s0_neg).sum(-1))))
        
        tcn_loss = ((tcn_loss_1 + tcn_loss_2) / 2.0).mean()
        metrics['loss_tcn'] = tcn_loss.item()
        metrics['alignment'] = (1.0 * (sim_0_2 < sim_1_2) * (sim_0_1 > sim_0_2)).float().mean().item()

        #* compute full loss
        full_loss += self.loss_weight.tcn * tcn_loss
        metrics['full_loss'] = full_loss.item()

        return {'loss': full_loss, 'metrics': metrics}
    
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
        """ Similarity function
        """
        if self.similarity_type == 'l2':
            d = -torch.linalg.norm(x - y, dim=-1)
            return d
        elif self.similarity_type == 'cosine':
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
            d = torch.einsum('...i,...i->...', x, y)
            return d
        else:
            raise NotImplementedError
