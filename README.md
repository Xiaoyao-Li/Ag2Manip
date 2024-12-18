# Ag2Manip
Code Repository for **Ag2Manip: Learning Novel Manipulation Skills with Agent-Agnostic Visual and Action Representations**

by [Puhao Li](https://xiaoyao-li.github.io/)<sup> *</sup>, [Tengyu Liu](http://tengyu.ai/)<sup> *</sup>, [Yuyang Li](https://blog.aidenli.net/), [Muzhi Han](https://sites.google.com/view/muzhihan/home), [Haoran Geng](https://geng-haoran.github.io/), [Shu Wang](https://github.com/AssassinWS), [Yixin Zhu](https://yzhu.io/), [Song-Chun Zhu](https://zhusongchun.net/), [Siyuan Huang](https://siyuanhuang.com/)

<p align="center">
    <a href='https://xiaoyao-li.github.io/research/ag2manip/pdf/ag2manip_compressed.pdf'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://arxiv.org/abs/2404.17521'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://xiaoyao-li.github.io/research/ag2manip/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
</p>

![Teaser](./assets/readme/teaser.png)
**Ag2Manip enables various manipulation tasks in scenarios where domain-specific demonstrations are unavailable.** With agent-agnostic visual and action representations, Ag2Manip: (a) learns from human manipulation videos; (b) acquires diverse manipulation skills autonomously in simulation; and (c) supports robust imitation learning of manipulation skills in the real world.

## Abstract
Enhancing the ability of robotic systems to autonomously acquire novel manipulation skills is vital for applications ranging from assembly lines to service robots. Existing methods (*e.g.*, VIP, R3M) rely on learning a generalized representation for manipulation tasks but overlook (i) the domain gap between distinct embodiments and (ii) the sparseness of successful task trajectories within the embodiment-specific action space, leading to misaligned and ambiguous task representations with inferior learning efficiency. Our work addresses the above challenges by introducing **Ag2Manip** (<ins>Ag</ins>ent-<ins>Ag</ins>nostic representations for <ins>Manip</ins>ulation) for learning novel manipulation skills. Our approach encompasses two principal innovations: (i) a novel agent-agnostic visual representation trained on human manipulation videos with embodiments masked to ensure generalizability, and (ii) an agent-agnostic action representation that abstracts the robot’s kinematic chain into an agent proxy with a universally applicable action space to focus on the core interaction between the end-effector and the object. Through our experiments, Ag2Manip demonstrates remarkable improvements across a diverse array of manipulation tasks without necessitating domain-specific demonstrations, substantiating **a significant 325% improvement** in average success rate across 24 tasks from FrankaKitchen, ManiSkill, and PartManip. Further ablation studies underscore the critical role of both representations in achieving such improvements.


## Pipeline
We first process a human demonstration dataset by masking and inpainting the areas occupied by human. Then we train an agent-agnostic visual representation on this dataset. We harness RL to learn manipulation policies in an agent-agnostic action space that abstracts the end-effector into an agent proxy, with a novel reward function emerging from our agent-agnostic visual representation. Lastly, the trajectory devised for the proxy agent is adapted to the robot through Inverse Kinematics.

## Installation
1. Create a new `conda` environment and activate it.
    ```bash
    conda create -n ag2manip python=3.8
    conda activate ag2manip
    ```

2. Install dependent libraries with `pip`.
    ```bash
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
    pip install -r requirements.txt
    ```
    - The code is tested on `pytorch1.13.1` and `cuda11.7`, modify the installation command to install [other versions](https://pytorch.org/get-started/previous-versions/) of `pytorch`.

3. Install [Isaac Gym](https://developer.nvidia.com/isaac-gym) by following the official documentation.

## Data preparation
We choose [Epic-Kitchen](https://epic-kitchens.github.io/2024) as the human demonstration dataset. First, segment the human body from each frame using the [ODISE](https://github.com/NVlabs/ODISE) algorithm. Then employ a video inpainting model, [E2FGVI](https://github.com/MCG-NKU/E2FGVI), to fill in the areas previously occupied by the human.

###  IsaacGym Assets
To access the `assets` for the simulated environments, please head to [Google Drive](https://drive.google.com/drive/folders/1UTjoDfYpgClHg2e1vo6KHC4EPcjC7CIo?usp=drive_link).

## Usage
### Train Visual Representation
1. Train our visual representation model on EPIC-KITCHEN dataset:
   - ``` bash
     cd repre_trainer
   - Run `train_ddp.py` to train our model on multiple GPUs in parallel, or run `train.py` to train on a single GPU.
2. Specify your model save path by modifying `exp_name` in `repre_trainer/cfgs/default.yml`.
3. You can access the pre-trained `ag2manip` visual representation model checkpoints [here](https://drive.google.com/drive/folders/1UTjoDfYpgClHg2e1vo6KHC4EPcjC7CIo?usp=drive_link).

### Train Manipulation Skills with a Proxy Agent
```bash
python train.py
    --plan: store_true, run transfered trajectory, no training.
    --traj_path: str, dummy trajectory path for planning.
    --save_goal: store_true, rather save goal image.
    --save_video: store_true, rather save executed video.
    --agentago: store_true, make video agent-agnostic.
    --seed: int, global random seed(default: 42).
    --save_traj, store_true, run trained policy, no training, save the test result.
    --task, str, specific the task in the environment (args.env) to run.
    --camera, str, specific the camera sensor in the environment to run.
    --disable_wandb, store_true, disable wandb logging.
    --debug_vis, store_true, enable debug visualization.
    --randomize, store_true, apply env reset randomization.
    --test, store_true, run trained policy, no training.
    --play, store_true, run trained policy, the same as test, can be used only by rl_games RL library.
    --resume, int, resume training or start testing from a checkpoint.
    --checkpoint, str, path to the saved weights, only for rl_games RL library.
    --headless, store_true, force display off at all times.
    --logdir, str, log path.
    --num_envs, int, number of environments to create - override config file.
    --episode_length, int, episode length, by default is read from yaml config.
    --seed, int, random seed.
    --algo, str, choose an RL algorithm.
    --model_dir, str, choose a model dir.
```

For example, train a manipulation task `task_name` from `benchmark_name` (frankakitchen, partmanip, maniskill) with our method `ag2manip` in IsaacGym with the following command:
```bash
python train.py --task=benchmark_name@task_name@ag2manip --algo=ppo --seed=42 --cfg_train=cfgs/algo/ppo/manipulation.yaml --disable_wandb --camera=default
```
The best policy will be saved as `model_best.pt` in `logs/ag2manip/task_name@default/ag2manip@ppo.42/`.
   
### Inference with a Proxy Agent
Inference and save a trajectory generated by the trained policy `logs/ag2manip/task_name@default/ag2manip@ppo.42/model_best.pt`:
```bash
python train.py --task=benchmark_name@task_name@ag2manip --model_dir=logs/ag2manip/task_name@default/ag2manip@ppo.42/model_best.pt --test --save_traj --algo=ppo --cfg_train=cfgs/algo/ppo/manipulation.yaml --camera=default --seed=0 --disable_wandb
```
The trajectory will be saved as `logs/ag2manip/task_name@default/ag2manip@ppo.42/absres_best.pkl`.

### Plan with a Franka Robot Arm
Plan with a franka robot arm using the saved trajectory:
```bash
python plan.py --task=benchmark_name@task_name@ag2manip --traj_path=logs/ag2manip/task_name@default/ag2manip@ppo.42/absres_best.pkl --pipeline=cpu --algo=ppo --cfg_train=cfgs/algo/ppo/manipulation.yaml --disable_wandb --camera=default
```

## Citation
If you find this work is helpful, please consider citing us as
```
@article{li2024ag2manip,
  title={Ag2Manip: Learning Novel Manipulation Skills with Agent-Agnostic Visual and Action Representations},
  author={Li, Puhao and Liu, Tengyu and Li, Yuyang and Han, Muzhi and Geng, Haoran and Wang, Shu and Zhu, Yixin and Zhu, Song-Chun and Huang, Siyuan},
  journal={arXiv preprint arXiv:2404.17521},
  year={2024}
}
```


## Contact
If you have any questions about this work, feel free to contact Puhao Li at puhaoli01@gmail.com.
