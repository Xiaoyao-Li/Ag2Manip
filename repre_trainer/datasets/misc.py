from typing import Dict, List
import torch
from einops import rearrange

def collate_fn_general(batch: List) -> Dict:
    """ General collate function used for dataloader.
    """
    batch_data = {key: [d[key] for d in batch] for key in batch[0]}
    
    # for key in batch_data:
    #     if torch.is_tensor(batch_data[key][0]):
    #         batch_data[key] = torch.stack(batch_data[key])
    return batch_data

def collate_fn_epic_vip(batch: List) -> Dict:
    """ Collate function used for EPIC-KITCHENS dataset.
    """
    batch_data = {key: [d[key] for d in batch] for key in batch[0]}
    batch_data['imgs'] = torch.stack(batch_data['imgs'])
    batch_data['start_ind'] = torch.tensor(batch_data['start_ind'], dtype=torch.long)
    batch_data['stop_ind'] = torch.tensor(batch_data['stop_ind'], dtype=torch.long)
    batch_data['s0_ind'] = torch.tensor(batch_data['s0_ind'], dtype=torch.long)
    batch_data['s1_ind'] = torch.tensor(batch_data['s1_ind'], dtype=torch.long)

    return batch_data

def collate_fn_epic_r3m(batch: List) -> Dict:
    """ Collate function used for EPIC-KITCHENS dataset.
    """
    batch_data = {key: [d[key] for d in batch] for key in batch[0]}
    batch_data['imgs'] = torch.stack(batch_data['imgs'])
    batch_data['s0_ind'] = torch.tensor(batch_data['s0_ind'], dtype=torch.long)
    batch_data['s1_ind'] = torch.tensor(batch_data['s1_ind'], dtype=torch.long)
    batch_data['s2_ind'] = torch.tensor(batch_data['s2_ind'], dtype=torch.long)

    return batch_data

def collate_fn_epic_clip(batch: List) -> Dict:
    """ Collate function used for EPIC-KITCHEN Clips dataset.
    """
    assert len(batch) == 1, "batch size must be exactly 1"
    batch_data = batch[0]

    return batch_data

def collate_fn_arnold_clip(batch: List) -> Dict:
    """ Collate function used for EPIC-KITCHEN Clips dataset.
    """
    assert len(batch) == 1, "batch size must be exactly 1"
    batch_data = batch[0]

    return batch_data
