from algos.rl.ppo import PPO
# from algos.rl.sac import SAC
# from algos.rl.td3 import TD3
# from algos.rl.ddpg import DDPG
# from algos.rl.trpo import TRPO

def process_sarl(args, env, cfg_train, logdir):
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    # is_testing = True
    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        is_testing = True
        chkpt_path = args.model_dir

    if args.max_iterations != -1:
        cfg_train["learn"]["max_iterations"] = args.max_iterations

    logdir = logdir + ".{}".format(env.task.cfg["seed"])

    """Set up the algo system for training or inferencing."""
    model = eval(args.algo.upper())(vec_env=env,
              cfg_train = cfg_train,
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              asymmetric=(env.num_states > 0)
              )

    # ppo.test("/home/hp-3070/logs/demo/scissors/ppo_seed0/model_6000.pt")
    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        model.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        model.load(chkpt_path)

    return model