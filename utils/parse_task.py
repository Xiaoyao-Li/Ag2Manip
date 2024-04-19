# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# from bidexhands.tasks.shadow_hand_over import ShadowHandOver
# from bidexhands.tasks.shadow_hand_catch_underarm import ShadowHandCatchUnderarm
# from bidexhands.tasks.shadow_hand_two_catch_underarm import ShadowHandTwoCatchUnderarm
# from bidexhands.tasks.shadow_hand_catch_abreast import ShadowHandCatchAbreast
# from bidexhands.tasks.shadow_hand_lift_underarm import ShadowHandLiftUnderarm
# from bidexhands.tasks.shadow_hand_catch_over2underarm import ShadowHandCatchOver2Underarm
# from bidexhands.tasks.shadow_hand_door_close_inward import ShadowHandDoorCloseInward
# from bidexhands.tasks.shadow_hand_door_close_outward import ShadowHandDoorCloseOutward
# from bidexhands.tasks.shadow_hand_door_open_inward import ShadowHandDoorOpenInward
# from bidexhands.tasks.shadow_hand_door_open_outward import ShadowHandDoorOpenOutward
# from bidexhands.tasks.shadow_hand_bottle_cap import ShadowHandBottleCap
# from bidexhands.tasks.shadow_hand_push_block import ShadowHandPushBlock
# from bidexhands.tasks.shadow_hand_swing_cup import ShadowHandSwingCup
# from bidexhands.tasks.shadow_hand_grasp_and_place import ShadowHandGraspAndPlace
# from bidexhands.tasks.shadow_hand_scissors import ShadowHandScissors
# from bidexhands.tasks.shadow_hand_switch import ShadowHandSwitch
# from bidexhands.tasks.shadow_hand_pen import ShadowHandPen
# from bidexhands.tasks.shadow_hand_re_orientation import ShadowHandReOrientation
# from bidexhands.tasks.shadow_hand_kettle import ShadowHandKettle
# from bidexhands.tasks.shadow_hand_block_stack import ShadowHandBlockStack

# # Allegro hand
# from bidexhands.tasks.allegro_hand_over import AllegroHandOver
# from bidexhands.tasks.allegro_hand_catch_underarm import AllegroHandCatchUnderarm

# # Meta
# from bidexhands.tasks.shadow_hand_meta.shadow_hand_meta_mt1 import ShadowHandMetaMT1
# from bidexhands.tasks.shadow_hand_meta.shadow_hand_meta_ml1 import ShadowHandMetaML1
# from bidexhands.tasks.shadow_hand_meta.shadow_hand_meta_mt4 import ShadowHandMetaMT4

# from bidexhands.tasks.hand_base.vec_task import VecTaskCPU, VecTaskGPU, VecTaskPython, VecTaskPythonArm
# from bidexhands.tasks.hand_base.multi_vec_task import MultiVecTaskPython, SingleVecTaskPythonArm
# from bidexhands.tasks.hand_base.multi_task_vec_task import MultiTaskVecTaskPython
# from bidexhands.tasks.hand_base.meta_vec_task import MetaVecTaskPython
# from bidexhands.tasks.hand_base.vec_task_rlgames import RLgamesVecTaskPython

from tasks.base.vec_task import VecTaskCPU, VecTaskGPU, VecTaskPython, VecTaskPythonArm

from importlib import import_module
from utils.config import warn_task_name
import json


def parse_task(args, cfg, cfg_train, sim_params, agent_index):

    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]

    if args.task_type == "C++":
        raise NotImplementedError("C++ task is not supported yet")
    
    elif args.task_type == "Python":
        print("Python")

        task_env, task_name, task_repre = args.task.split("@")
        camera = args.camera

        try:
            task_name_voc = task_name.split("_")
            task_name_voc = [word.capitalize() for word in task_name_voc]
            task_name = "".join(task_name_voc)
            Module = import_module(f"tasks.{task_env}.{task_name}")
            Task = getattr(Module, task_name)
            
            task = Task(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                camera=camera,
                headless=args.headless,)
        except NameError as e:
            print(e)
            warn_task_name()

        env = VecTaskPython(task, rl_device)
        # if args.task == "OneFrankaCabinet" :
        #     env = VecTaskPythonArm(task, rl_device)
        # else :
        #     env = VecTaskPython(task, rl_device)

    elif args.task_type == "MultiAgent":
        raise NotImplementedError("MultiAgent task is not supported yet")

    elif args.task_type == "MultiTask":
        raise NotImplementedError("MultiTask task is not supported yet")

    elif args.task_type == "Meta":
        raise NotImplementedError("Meta task is not supported yet")

    elif args.task_type == "RLgames":
        raise NotImplementedError("RLgames task is not supported yet")
    
    return task, env

