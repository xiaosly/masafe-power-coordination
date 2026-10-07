"""MAPPO-Lag training settings."""
import argparse


def get_config():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    integer_settings = dict(seed=1, n_training_threads=1, n_rollout_threads=4,
                            num_env_steps=480096, hidden_size=256, layer_N=2,
                            ppo_epoch=1, num_mini_batch=24, save_interval=1000, log_interval=500)
    float_settings = dict(lr=5e-4, critic_lr=5e-4, opti_eps=1e-5, weight_decay=0.0,
                          gain=0.01, std_x_coef=1.0, std_y_coef=0.5, clip_param=0.2,
                          entropy_coef=0.008, value_loss_coef=1.0, max_grad_norm=10.0,
                          huber_delta=10.0, gamma=0.99, gae_lambda=0.95,
                          lamda_lagr1=1.2, lamda_lagr2=0.78, lamda_local=0.5,
                          lagrangian_coef_rate=5e-4)
    for name, default in integer_settings.items():
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in float_settings.items():
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.set_defaults(
        algorithm_name="mappo_lagr", experiment_name="mappo_lag", cuda=False,
        episode_length=24, share_policy=False, use_single_network=False,
        use_centralized_V=True, use_obs_instead_of_state=False,
        stacked_frames=1, use_stacked_frames=False, use_ReLU=True,
        use_popart=True, use_valuenorm=True, use_feature_normalization=True,
        use_orthogonal=True, use_naive_recurrent_policy=False, use_recurrent_policy=False,
        recurrent_N=1, data_chunk_length=10, use_clipped_value_loss=True,
        use_max_grad_norm=True, use_gae=True, use_proper_time_limits=False,
        use_huber_loss=True, use_value_active_masks=False, use_policy_active_masks=True,
        use_linear_lr_decay=False, use_eval=False, n_eval_rollout_threads=1,
        eval_interval=24, eval_episodes=5, use_render=False, render_episodes=5,
        savedata=False, model_dir=None, safety_bound=0.0, local_safety_bound=0.0,
        unified_cost_objective=True, constraint_mode="lagrangian", penalty_coeff=1.0,
    )
    return parser
