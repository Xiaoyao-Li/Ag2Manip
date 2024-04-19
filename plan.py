import wandb
import pickle
import numpy as np

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_plan_cfg
from utils.parse_task_plan import parse_task_plan
from algos.planner.base import BASE
from algos.planner.approach import APPROACH


def plan():
    print(f'Planner: Default')

    task, env = parse_task_plan(args, cfg, sim_params)
    planner = APPROACH(env, cfg, save_goal=args.save_goal, save_video=args.save_video)
    planner.run()
    

if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, logdir = load_plan_cfg(args)
    sim_params = parse_sim_params(args, cfg, None)
    set_seed(42, True)
    plan()