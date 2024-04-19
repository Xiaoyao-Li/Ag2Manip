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
class VIP(nn.Module):
    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        super(VIP, self).__init__()
        self.d_emb = cfg.d_emb
        self.backbone_type = cfg.backbone_type
        self.reward_type = cfg.reward_type
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
                'imgs': imgs,  [B, T, C, H, W] (e.g., [32, 4, 3, 256, 256]), must be float32
                'start_ind': start_ind,
                'stop_ind': stop_ind,
                's0_ind': s0_ind,
                's1_ind': s1_ind}
        return:
            dict {
                'loss': full_loss, 
                'metrics': metrics for logs}
        """
        #* forward process
        imgs = data['imgs']
        start_ind = data['start_ind']
        stop_ind = data['stop_ind']
        s0_ind = data['s0_ind']
        s1_ind = data['s1_ind']

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
        embs = embs.reshape(B, T, embs.shape[-1])  # [B, T, d_emb]
        emb_start = embs[:, 0]
        emb_goal = embs[:, 1]
        emb_s0 = embs[:, 2]
        emb_s1 = embs[:, 3]

        #* compute metrics and full_loss
        full_loss = 0
        metrics = dict()

        #* 1. Embedding Norm Loss
        loss_l1 = torch.linalg.norm(embs, ord=1, dim=-1).mean()
        loss_l2 = torch.linalg.norm(embs, ord=2, dim=-1).mean()
        full_loss += self.loss_weight.l1norm * loss_l1
        full_loss += self.loss_weight.l2norm * loss_l2
        metrics['loss_l1'] = loss_l1.item()
        metrics['loss_l2'] = loss_l2.item()

        #* 2. VIP Loss
        v_o = self.similarity(emb_start, emb_goal)
        v_s0 = self.similarity(emb_s0, emb_goal)
        v_s1 = self.similarity(emb_s1, emb_goal)
        # compute reward (sparse version)
        reward = self.reward(start_ind, stop_ind, s0_ind, s1_ind)
        loss_vip = (1 - self.loss_weight.gamma) * ( - v_o.mean()) \
                    + torch.log(1e-6 + torch.mean(torch.exp( - (reward + self.loss_weight.gamma * v_s1 - v_s0))))

        #* 3. Additional negative observations
        v_s0_neg = []
        v_s1_neg = []
        perm = [i for i in range(B)]
        for _ in range(self.num_negatives):
            perm = [(i_perm + 1) % B for i_perm in perm]
            emb_s0_shuf = emb_s0[perm]
            emb_s1_shuf = emb_s1[perm]
            v_s0_neg.append(self.similarity(emb_s0_shuf, emb_goal))
            v_s1_neg.append(self.similarity(emb_s1_shuf, emb_goal))
        if self.num_negatives > 0:
            v_s0_neg = torch.cat(v_s0_neg)
            v_s1_neg = torch.cat(v_s1_neg)
            reward_neg = - torch.ones_like(v_s0_neg, device=v_s0_neg.device)
            loss_vip += torch.log(1e-6 + torch.mean(torch.exp( - (reward_neg + self.loss_weight.gamma * v_s1_neg - v_s0_neg))))
        metrics['loss_vip'] = loss_vip.item()
        # metrics['alignment'] = (1.0 * (v_s0 > v_o) * (v_s1 > v_s0)).float().mean().item()
        metrics['alignment'] = (0.5 * (v_s0 > v_o) + 0.5 * (v_s1 > v_o)).float().mean().item()

        #* compute full loss
        full_loss += loss_vip
        metrics['full_loss'] = full_loss.item()

        return {'loss': full_loss, 'metrics': metrics}
    
    def embedding(self, imgs: torch.Tensor) -> torch.Tensor:
        """ Embedding function
        Args:
            imgs: input tensor [B, C, H, W]
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
    
    def reward(self, start_ind, stop_ind, s0_ind, s1_ind) -> torch.Tensor:
        """ Reward function
        """
        if self.reward_type == 'sparse':
            reward = (s0_ind == stop_ind).float() - 1
            return reward
        elif self.reward_type == 'dense':
            raise NotImplementedError
        else:
            raise NotImplementedError
