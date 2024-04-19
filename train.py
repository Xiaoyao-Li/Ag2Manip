import wandb
import numpy as np

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_sarl import process_sarl
from utils.process_offrl import *


def train():
    print(f"Algorithm: {args.algo}")

    if args.algo in ['ppo', 'ddpg', 'sac', 'td3', 'trpo']:
        if args.save_traj:
            cfg['env']['numEnvs'] = 1
        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index=None)

        cfg_train['save_traj'] = args.save_traj
        sarl = eval('process_sarl')(args, env, cfg_train, logdir)

        iterations = cfg_train["learn"]["max_iterations"]
        if args.max_iterations > 0:
            iterations = args.max_iterations

        ## initialize wandb
        if not args.disable_wandb and not args.test:
            task_env, task_name, repre_name = args.task.split("@")
            camera_name = args.camera
            wandb.init(
                project=f'ag2manip',
                name=f'{camera_name}@{repre_name}.seed{env.task.cfg["seed"]}',
                config={
                    'cfg': cfg,
                    'cfg_train': cfg_train,
                    'cfg_repre': cfg_repre,
                    'args': args
                }
            )

        sarl.run(num_learning_iterations=iterations, log_interval=cfg_train["learn"]["save_interval"])
    
    elif args.algo in ["td3_bc", "bcq", "iql", "ppo_collect"]:
        raise NotImplementedError
    
    else:
        raise NotImplementedError


if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir, cfg_repre = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    train()