import os
import hydra
import torch
import random
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from utils.io import mkdir_if_not_exists
from utils.plot import Ploter
from datasets.base import create_dataset
from datasets.misc import collate_fn_general, collate_fn_epic_vip, collate_fn_epic_r3m
from models.base import create_model
# from models.visualizer import create_visualizer


def train(cfg: DictConfig) -> None:
    """ training portal

    Args:
        cfg: configuration dict
    """
    ## set device
    device = torch.device('cuda', cfg.gpu)

    ## prepare dataset for train and test
    datasets = {
        'train': create_dataset(cfg.task.dataset, 'train', cfg.slurm, case_only=False, ),
    }
    if cfg.task.visualizer.visualize:
        raise NotImplementedError
        datasets['test_for_vis'] = create_dataset(cfg.task.dataset, 'test', cfg.slurm, case_only=True)
    
    if cfg.gpu == 0:
        for subset, dataset in datasets.items():
            logger.info(f'Load {subset} dataset size: {len(dataset)}')
    
    if cfg.model.name.lower() in ['vip', 'livip']:
        collate_fn = collate_fn_epic_vip
    elif cfg.model.name.lower() in ['r3m', 'lir3m']:
        collate_fn = collate_fn_epic_r3m
    else:
        collate_fn = collate_fn_general
    
    train_sampler = DistributedSampler(datasets['train'])
    dataloaders = {
        'train': datasets['train'].get_dataloader(
            sampler=train_sampler,
            batch_size=cfg.task.train.batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.task.train.num_workers,
            pin_memory=True,
        ),
    }
    if 'test_for_vis' in datasets:
       raise NotImplementedError

    ## create model and optimizer
    model = create_model(cfg, slurm=cfg.slurm, device=device)
    model.to(device=device)
    
    params = []
    nparams = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)
            nparams.append(p.nelement())
            if cfg.gpu == 0:
                logger.info(f'add {n} {p.shape} for optimization')
    
    params_group = [
        {'params': params, 'lr': cfg.task.lr},
    ]
    optimizer = torch.optim.Adam(params_group) # use adam optimizer in default

    if cfg.gpu == 0:
        logger.info(f'{len(params)} parameters for optimization.')
        logger.info(f'total model size is {sum(nparams)}.')

    ## create visualizer if visualize in training process
    if cfg.task.visualizer.visualize:
        raise NotImplementedError
        visualizer = create_visualizer(cfg.task.visualizer)
    
    ## convert to parallel
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[cfg.gpu], output_device=cfg.gpu, find_unused_parameters=True)

    ## load if use ckpt
    current_epoch = 0
    step = 0
    if cfg.ckpt is not None:
        logger.info(f'Load checkpoint from {cfg.ckpt_dir}')
        checkpoint = torch.load(os.path.join(cfg.ckpt_dir, 'model.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        current_epoch = checkpoint['epoch'] + 1
        step = checkpoint['step'] + 1

    ## start training
    for epoch in range(current_epoch, cfg.task.train.num_epochs):
        ## set random seed for each epoch sampler
        train_sampler.set_epoch(epoch)
        model.train()
        for it, data in enumerate(dataloaders['train']):
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            optimizer.zero_grad()
            data['epoch'] = epoch
            outputs = model(data)
            outputs['loss'].backward()

            #* gradient clip to solve the gradient explosion problem
            if cfg.task.clip_grad > 0:
                grad = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 
                                                      max_norm=cfg.task.clip_grad)
                outputs['metrics']['grad_norm'] = grad
            outputs['metrics']['max_grad_norm_clip'] = cfg.task.clip_grad
            
            #! requires debug checking
            has_nan_gradients = False
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if torch.isnan(param.grad).any():
                        has_nan_gradients = True
                        break
                elif param.requires_grad and param.grad is None:
                    logger.warning(f'step {step} | No gradient for {name}, set to zero')
                    param.grad = torch.zeros_like(param)
            if has_nan_gradients:
                logger.warning('NaN gradients detected, not updating model parameters')
                optimizer.zero_grad()

            optimizer.step()
            
            ## plot loss
            if cfg.gpu == 0 and (step + 1) % cfg.task.train.log_step == 0:
                total_loss = outputs['loss'].item()
                log_str = f'[TRAIN] ==> Epoch: {epoch+1:3d} | Iter: {it+1:5d} | Step: {step+1:7d} | Loss: {total_loss:.3f}'
                logger.info(log_str)
                output_metrics = outputs['metrics']
                for key in output_metrics:
                    val = output_metrics[key].item() if torch.is_tensor(output_metrics[key]) else output_metrics[key]
                    Ploter.write({
                        f'train/{key}': {'plot': True, 'value': val, 'step': step},
                        'train/epoch': {'plot': True, 'value': epoch, 'step': step},
                    })

            step += 1
        
        ## save ckpt in epoch
        if cfg.gpu == 0 and (epoch + 1) % cfg.save_model_interval == 0:
            save_path = os.path.join(
                cfg.ckpt_dir, 
                f'model_{epoch}.pth' if cfg.save_model_seperately else 'model.pth'
            )

            save_ckpt(
                model=model, optimizer=optimizer, 
                epoch=epoch, step=step, path=save_path,)
            # logger.warning(f'DEBUG saveing ckpt to {save_path}')

        ## test for visualize
        if cfg.task.visualizer.visualize and (epoch + 1) % cfg.task.visualizer.interval == 0:
            raise NotImplementedError
            vis_dir = os.path.join(cfg.vis_dir, f'epoch{epoch+1:0>4d}')
            visualizer.visualize(model, dataloaders['test_for_vis'], vis_dir)

def save_ckpt(model: torch.nn.Module, optimizer: torch.optim, epoch: int, step: int, path: str,) -> None:
    """ Save current model and corresponding data

    Args:
        model: current model
        optimizer: current optimizer
        epoch: best epoch
        step: current step
        path: save path
    """
    saved_state_dict = {}
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        ## if use frozen pretrained scene model, we can avoid saving scene model to save space
        if 'scene_model' in key:
            continue

        saved_state_dict[key] = model_state_dict[key]
    
    logger.info('Saving model!!!' + ('[ALL]'))
    torch.save({
        'model': saved_state_dict,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch, 'step': step,
    }, path)

@hydra.main(version_base=None, config_path="./cfgs", config_name="default")
def main(cfg: DictConfig) -> None:
    ## set rank
    cfg.gpu = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(cfg.gpu)
    # device = torch.device('cuda', cfg.gpu)
    torch.distributed.init_process_group(backend='nccl')

    ## compute modeling dimension according to task
    if os.environ.get('SLURM') is not None:
        cfg.slurm = True # update slurm config
        logger.remove(handler_id=0) # remove default handler

    ## set if ckpts are used
    if cfg.ckpt is not None:
        cfg.exp_dir = os.path.join(cfg.output_dir, cfg.ckpt)

    ## set output logger and tensorboard
    if cfg.gpu == 0:
        logger.add(cfg.exp_dir + '/runtime.log')

        mkdir_if_not_exists(cfg.tb_dir)
        mkdir_if_not_exists(cfg.vis_dir)
        mkdir_if_not_exists(cfg.ckpt_dir)

        writer = SummaryWriter(log_dir=cfg.tb_dir)
        Ploter.setWriter(writer)

        ## Begin training progress
        logger.info('Configuration: \n' + OmegaConf.to_yaml(cfg))
        logger.info('Begin training..')

    train(cfg) # training portal

    ## Training is over!
    if cfg.gpu == 0:
        writer.close() # close summarywriter and flush all data to disk
        logger.info('End training..')

if __name__ == '__main__':
    ## set random seed
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main()
