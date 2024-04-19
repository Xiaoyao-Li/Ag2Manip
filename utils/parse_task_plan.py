from tasks.base.vec_task import VecTaskCPU, VecTaskGPU, VecTaskPython, VecTaskPythonArm

from importlib import import_module
from utils.config import warn_task_name
import json

def parse_task_plan(args, cfg, sim_params):

    # create native task and pass custom config
    device_id = args.device_id

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

        env = VecTaskPython(task, device_id)
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

