
def process_td3_bc(args, env, cfg_train, logdir):
    from bidexhands.algorithms.offrl.td3_bc import TD3_BC
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    # is_testing = True
    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        is_testing = True
        chkpt_path = args.model_dir

    """Set up the PPO system for training or inferencing."""
    td3_bc = TD3_BC(vec_env=env,
                device=env.rl_device,
                discount = learn_cfg["discount"],
                tau = learn_cfg["tau"],
                policy_freq = learn_cfg["policy_freq"],
                alpha = learn_cfg["alpha"],
                batch_size = learn_cfg["batch_size"],
                max_timesteps = learn_cfg["max_timesteps"],
                iterations =  learn_cfg["iterations"],
                log_dir=logdir,
                datatype = args.datatype)

    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        td3_bc.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        td3_bc.load(chkpt_path)

    return td3_bc

def process_bcq(args, env, cfg_train, logdir):
    from bidexhands.algorithms.offrl.bcq import BCQ
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    # is_testing = True
    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        is_testing = True
        chkpt_path = args.model_dir

    """Set up the PPO system for training or inferencing."""
    bcq = BCQ(vec_env=env,
                device=env.rl_device,
                discount = learn_cfg["discount"],
                tau = learn_cfg["tau"],
                lmbda = learn_cfg["lmbda"],
                phi = learn_cfg["phi"],
                batch_size = learn_cfg["batch_size"],
                max_timesteps = learn_cfg["max_timesteps"],
                iterations =  learn_cfg["iterations"],
                log_dir=logdir,
                datatype = args.datatype)

    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        bcq.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        bcq.load(chkpt_path)

    return bcq

def process_iql(args, env, cfg_train, logdir):
    from bidexhands.algorithms.offrl.iql import IQL
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    # is_testing = True
    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        is_testing = True
        chkpt_path = args.model_dir

    """Set up the PPO system for training or inferencing."""
    iql = IQL(vec_env=env,
                device=env.rl_device,
                discount = learn_cfg["discount"],
                tau = learn_cfg["tau"],
                expectile = learn_cfg["expectile"],
                beta = learn_cfg["beta"],
                scale = learn_cfg["scale"],
                batch_size = learn_cfg["batch_size"],
                max_timesteps = learn_cfg["max_timesteps"],
                iterations =  learn_cfg["iterations"],
                log_dir=logdir,
                datatype = args.datatype)

    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        iql.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        iql.load(chkpt_path)

    return iql

def process_ppo_collect(args, env, cfg_train, logdir):
    from bidexhands.algorithms.offrl.ppo_collect import PPO, ActorCritic
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    # is_testing = True
    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        is_testing = True
        chkpt_path = args.model_dir

    logdir = logdir + "_seed{}".format(env.task.cfg["seed"])

    """Set up the PPO system for training or inferencing."""
    ppo_collect = PPO(vec_env=env,
              actor_critic_class=ActorCritic,
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              lam=learn_cfg["lam"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
              entropy_coef=learn_cfg["ent_coef"],
              learning_rate=learn_cfg["optim_stepsize"],
              max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
              use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
              schedule=learn_cfg.get("schedule", "fixed"),
              desired_kl=learn_cfg.get("desired_kl", None),
              model_cfg=cfg_train["policy"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              asymmetric=(env.num_states > 0),
              data_size=learn_cfg["data_size"]
              )

    # ppo.test("/home/hp-3070/bi-dexhands/bi-dexhands/logs/shadow_hand_lift_underarm2/ppo/ppo_seed2/model_40000.pt")
    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        ppo_collect.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        ppo_collect.load(chkpt_path)

    return ppo_collect