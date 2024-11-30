# import sys
# from legged_gym import LEGGED_GYM_ROOT_DIR
# import os
# import sys
# from legged_gym import LEGGED_GYM_ROOT_DIR

# import isaacgym
# from legged_gym.envs import *
# from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

# import numpy as np
# import torch

# from legged_gym.utils import webviewer


# def play(args):
#     web_viewer = webviewer.WebViewer()

#     env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
#     # override some parameters for testing
#     env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
#     env_cfg.terrain.num_rows = 5
#     env_cfg.terrain.num_cols = 5
#     env_cfg.terrain.curriculum = False
#     env_cfg.noise.add_noise = False
#     env_cfg.domain_rand.randomize_friction = False
#     env_cfg.domain_rand.push_robots = False

#     env_cfg.env.test = True

#     # prepare environment
#     env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

#     web_viewer.setup(env)


#     obs = env.get_observations()
#     # load policy
#     train_cfg.runner.resume = True
#     ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
#     policy = ppo_runner.get_inference_policy(device=env.device)

#     print("initialized policy")
    
#     # export policy as a jit module (used to run it from C++)
#     if EXPORT_POLICY:
#         path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
#         export_policy_as_jit(ppo_runner.alg.actor_critic, path)
#         print('Exported policy as jit script to: ', path)

#     for i in range(10*int(env.max_episode_length)):

#         print("stepping")
#         actions = policy(obs.detach())
#         obs, _, rews, dones, infos = env.step(actions.detach())

#         print(f"obs: {obs}")

#         web_viewer.render(fetch_results=True,
#                         step_graphics=True,
#                         render_all_camera_sensors=True,
#                         wait_for_page_load=True)

# if __name__ == '__main__':
#     EXPORT_POLICY = True
#     RECORD_FRAMES = False
#     MOVE_CAMERA = False
#     args = get_args()
#     play(args)
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
import numpy as np
import torch
from legged_gym.utils import webviewer
from isaacgym import gymapi, gymtorch
import time

def create_visualizer_markers(gym, sim, env_handles, num_envs, device):
    """Create visualization markers for all environments before simulation starts."""

    # Create sphere asset once
    sphere_opts = gymapi.AssetOptions()
    sphere_opts.fix_base_link = True
    sphere_radius = 0.05
    sphere_pose = gymapi.Transform()
    sphere_asset = gym.create_sphere(sim, sphere_radius, sphere_opts)

    # Colors for different feet
    colors = [
        gymapi.Vec3(1.0, 0.0, 0.0),  # front left - red
        gymapi.Vec3(0.0, 1.0, 0.0),  # front right - green
        gymapi.Vec3(0.0, 0.0, 1.0),  # rear left - blue
        gymapi.Vec3(1.0, 1.0, 0.0),  # rear right - yellow
    ]

    markers = []
    marker_actor_indices = []
    for env_idx, env_handle in enumerate(env_handles):
        env_markers = []
        env_marker_indices = []
        for i in range(4):
            marker_handle = gym.create_actor(
                env_handle, sphere_asset, sphere_pose, f"marker_{env_idx}_{i}", env_idx, 1
            )
            gym.set_rigid_body_color(
                env_handle, marker_handle, 0, gymapi.MESH_VISUAL, colors[i]
            )
            env_markers.append(marker_handle)
            # Get the actor index in the simulation domain
            marker_actor_index = gym.get_actor_index(
                env_handle, marker_handle, gymapi.DOMAIN_SIM
            )
            env_marker_indices.append(marker_actor_index)
        markers.append(env_markers)
        marker_actor_indices.append(env_marker_indices)
    return markers, marker_actor_indices

def update_marker_positions(root_states, marker_actor_indices, positions):
    """Update marker positions using the tensor API."""

    # positions is of shape (num_envs, num_markers, 3)
    num_envs = positions.shape[0]
    num_markers = positions.shape[1]

    for env_idx in range(num_envs):
        for marker_idx in range(num_markers):
            actor_index = marker_actor_indices[env_idx][marker_idx]
            root_states[actor_index, 0:3] = positions[env_idx, marker_idx, 0:3]

def copysign(a, b):
    # type: (float, torch.Tensor) -> torch.Tensor
    a_tensor = torch.full_like(b, a)
    return torch.abs(a_tensor) * torch.sign(b)

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Disable the viewer for headless execution
    env_cfg.viewer.use_viewer = False  # Disable the viewer
    env_cfg.viewer.use_gpu = True  # Ensure GPU pipeline is used
    env_cfg.sim.enable_gpu_pipeline = True

    # Override config for visualization
    env_cfg.env.num_envs = 10  # Set to match expected number
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True

    # Prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Create markers and get their actor indices before the simulation starts
    markers, marker_actor_indices = create_visualizer_markers(
        env.gym,
        env.sim,
        env.envs,
        env_cfg.env.num_envs,
        env.device
    )

    # Define default foot positions (relative to robot base)
    default_positions = np.array([
        [0.3, 0.2, 0.05],    # front left
        [0.3, -0.2, 0.05],   # front right
        [-0.3, 0.2, 0.05],   # rear left
        [-0.3, -0.2, 0.05],  # rear right
    ])

    # Expand to all environments
    goal_positions = np.tile(default_positions, (env_cfg.env.num_envs, 1, 1))
    goal_positions = torch.tensor(goal_positions, device=env.device, dtype=torch.float)

    # Get root state tensor
    root_state_tensor = env.gym.acquire_actor_root_state_tensor(env.sim)
    root_states = gymtorch.wrap_tensor(root_state_tensor)

    # Get initial observations
    obs = env.get_observations()

    # Load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    print("Initialized policy")

    # Export policy if needed
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    try:
        for i in range(10 * int(env.max_episode_length)):
            # Get policy actions
            actions = policy(obs.detach())

            # Step environment
            obs, _, rews, dones, infos = env.step(actions.detach())

            # Update visualization markers
            if markers:  # Only if markers were successfully created
                update_marker_positions(root_states, marker_actor_indices, goal_positions)

            # Since the viewer is disabled, we don't need to render
            # If you have other logging or visualization methods, you can include them here

            # Small sleep to maintain simulation pacing
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Visualization stopped by user")
    finally:
        if env.viewer:
            env.gym.destroy_viewer(env.viewer)

if __name__ == '__main__':
    EXPORT_POLICY = True
    args = get_args()
    play(args)
