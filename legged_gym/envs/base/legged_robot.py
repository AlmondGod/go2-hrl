from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

torch.set_default_tensor_type(torch.FloatTensor)

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
from legged_gym.utils.terrain import Terrain
from fetch.tamols.tamols import TAMOLSState, setup_variables
from fetch.tamols.constraints import add_initial_constraints, add_kinematic_constraints, add_giac_constraints, add_friction_cone_constraints
from fetch.tamols.costs import add_tracking_cost, add_foothold_on_ground_cost, add_nominal_kinematic_cost, add_base_pose_alignment_cost, add_edge_avoidance_cost
from fetch.tamols.map_processing import process_height_maps
from pydrake.solvers import SnoptSolver
from fetch.tamols.plotting_helpers import plot_optimal_solutions_interactive, save_optimal_solutions

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        self.immediate_heightfield = np.zeros((14, 14), dtype=np.int16)

        # target positions one starts at current leg locations
        self.target_positions_one = self.rigid_body_states[:, self.feet_indices, :3]  # Shape: (num_envs, num_feet, 3)
        self.target_positions_two = self.trajectory_optimizer()
        self.active_pair = 1  # Start with pair 1 (FR+BL)
       
    def trajectory_optimizer(self):
        """
        Uses TAMOLS optimization to find optimal foot positions given the current state and terrain.
        Returns: Tensor of shape (num_envs, num_feet, 3) containing target positions
        """
        # Initialize TAMOLS state
        tmls = TAMOLSState()
        
        # Set current state
        tmls.base_pose = np.array([
            self.root_states[0, 0].cpu().numpy(),  # x
            self.root_states[0, 1].cpu().numpy(),  # y
            self.root_states[0, 2].cpu().numpy(),  # z
            self.rpy[0, 0].cpu().numpy(),          # roll
            self.rpy[0, 1].cpu().numpy(),          # pitch
            self.rpy[0, 2].cpu().numpy()           # yaw
        ])
        
        # Set base velocity (linear and angular)
        tmls.base_vel = np.array([
            self.root_states[0, 7].cpu().numpy(),   # linear x
            self.root_states[0, 8].cpu().numpy(),   # linear y
            self.root_states[0, 9].cpu().numpy(),   # linear z
            self.root_states[0, 10].cpu().numpy(),  # angular x
            self.root_states[0, 11].cpu().numpy(),  # angular y
            self.root_states[0, 12].cpu().numpy()   # angular z
        ])

        # Set reference velocity and angular momentum
        tmls.ref_vel = np.array([0.1, 0, 0])  # Example: move forward at 0.1 m/s
        tmls.ref_angular_momentum = np.array([0, 0, 0])  # No angular momentum
        
        # Set gait pattern
        tmls.gait_pattern = {
            'phase_timing': [0, 0.4, 0.8, 1.2, 1.6, 2.0],
            'contact_states': [
                [1, 1, 1, 1],
                [1, 0, 1, 0],
                [1, 1, 1, 1],
                [0, 1, 0, 1],
                [1, 1, 1, 1],
            ],
            'at_des_position': [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 0, 1],
                [0, 1, 0, 1],
                [1, 1, 1, 1],
            ],
        }

        # Set current foot positions
        tmls.p_meas = self.rigid_body_states[0, self.feet_indices, :3].cpu().numpy()

        # Process heightfield
        elevation_map = self.immediate_heightfield.astype(np.float32)
        h_s1, h_s2, gradients = process_height_maps(elevation_map)
        
        # Set height maps and gradients
        tmls.h = elevation_map
        tmls.h_s1 = h_s1
        tmls.h_s2 = h_s2
        tmls.h_grad_x, tmls.h_grad_y = gradients['h']
        tmls.h_s1_grad_x, tmls.h_s1_grad_y = gradients['h_s1']
        tmls.h_s2_grad_x, tmls.h_s2_grad_y = gradients['h_s2']

        # Setup optimization variables
        setup_variables(tmls)

        # Temp constraint
        for leg_idx, pos in enumerate([[0.4, 0.1, 0], [0.4, -0.1, 0], [0, 0.1, 0], [0, -0.1, 0]]):
            for dim in range(2): # just x, y pos (z handled by foot on ground cost
                c = tmls.prog.AddLinearConstraint(tmls.p[leg_idx, dim] == pos[dim])
                tmls.test_constraints.append(c)

        # Add constraints
        add_initial_constraints(tmls)
        add_kinematic_constraints(tmls)
        # add_giac_constraints(tmls)
        # add_friction_cone_constraints(tmls)

        # Add costs
        add_foothold_on_ground_cost(tmls)
        # add_nominal_kinematic_cost(tmls)
        # add_base_pose_alignment_cost(tmls)

        # Solve optimization
        solver = SnoptSolver()
        tmls.result = solver.Solve(tmls.prog)

        # Check if optimization succeeded before accessing result
        if not tmls.result.is_success():
            print("Optimization failed, using fallback strategy")
            current_positions = self.rigid_body_states[:, self.feet_indices, :3]
            new_positions = current_positions.clone()
            new_positions[:, :, 0] += 0.1  # Move 10cm forward
            return new_positions

        # Only try to get solution if optimization succeeded
        tmls.optimal_footsteps = tmls.result.GetSolution(tmls.p)
        num_phases = len(tmls.gait_pattern['phase_timing']) - 1
        tmls.optimal_spline_coeffs = [tmls.result.GetSolution(tmls.spline_coeffs[i]) for i in range(num_phases)]

        plot_optimal_solutions_interactive(tmls)
        save_optimal_solutions(tmls)

        # Extract solution
        if tmls.result.is_success():
            optimal_footsteps = tmls.result.GetSolution(tmls.p)
            # Convert to tensor and repeat for all environments
            optimal_positions = torch.tensor(optimal_footsteps, device=self.device)
            optimal_positions = optimal_positions.unsqueeze(0).repeat(self.num_envs, 1, 1)
            return optimal_positions
        else:
            # If optimization fails, return current positions plus small forward step
            print("Optimization failed, using fallback strategy")
            current_positions = self.rigid_body_states[:, self.feet_indices, :3]
            new_positions = current_positions.clone()
            new_positions[:, :, 0] += 0.1  # Move 10cm forward
            return new_positions

    def step(self, actions):
        # Get current foot positions
        current_feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
        
        # Define diagonal pairs 
        pair1_indices = torch.tensor([0, 3], device=self.device)  # LF, RH
        pair2_indices = torch.tensor([1, 2], device=self.device)  # RF, LH
        
        # Check if active pair reached their targets # NOTE:is equals to strong of a constaint?
        if self.active_pair == 1:
            active_indices = pair1_indices
            check_positions = current_feet_pos[:, pair1_indices]
            check_targets = self.target_positions_two[:, pair1_indices]
        else:
            active_indices = pair2_indices
            check_positions = current_feet_pos[:, pair2_indices]
            check_targets = self.target_positions_two[:, pair2_indices]
        
        # Check if feet reached targets
        feet_reached = self._check_feet_at_target(check_positions, check_targets)
        
        # If feet reached targets, swap positions and calculate new targets
        if torch.any(feet_reached):
            # Update environments where feet reached targets
            env_ids = torch.where(feet_reached)[0]
            
            # Swap positions (pos2 becomes pos1)
            self.target_positions_one[env_ids] = self.target_positions_two[env_ids]
            
            # Calculate new target positions
            new_targets = self.trajectory_optimizer() # DO WE HAVE TO DO THIS JUST YET? (if only half way through the traj)
            self.target_positions_two[env_ids] = new_targets[env_ids]
            
            # Switch active pair
            self.active_pair = 3 - self.active_pair  # Toggles between 1 and 2
        
        # Continue with normal step logic
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10]).to(dtype=torch.float32)
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13]).to(dtype=torch.float32)
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec).to(dtype=torch.float32)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward() # COMPUTE REWARD
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        obs = torch.cat([
            self._ensure_float32(self.base_lin_vel * self.obs_scales.lin_vel),
            self._ensure_float32(self.base_ang_vel * self.obs_scales.ang_vel),
            self._ensure_float32(self.projected_gravity),
            self._ensure_float32(self.commands[:, :3] * self.commands_scale),
            self._ensure_float32(self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self._ensure_float32(self.dof_vel * self.obs_scales.dof_vel),
            self._ensure_float32(self.actions)
        ], dim=-1)
        
        return obs

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        # self._create_ground_plane()

        # TODO: write create_heightmap and create_triangle_meshes and run either (dont need to add mesh_type to cfg)
        mesh_type = self.cfg.terrain.mesh_type
        self.terrain = Terrain(self.cfg.terrain, self.num_envs)

        # TODO: use either create_heightfield or create_trimesh, whichever works better with the Trajectory Optimizer
        # self._create_heightfield()
        self._create_trimesh()
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.debug_viz and hasattr(self, 'viewer') and self.viewer is not None:
            self._draw_debug_vis()
            
            # If you have target positions, visualize them
            if self.target_positions is not None:
                self._draw_target_positions(self.target_positions)

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

        # Update local heightfield window
        self._update_local_heightfield()

        # print(f"immediate_heightfield shape: {self.immediate_heightfield.shape}")
        # print(f"immediate_heightfield: {self.immediate_heightfield}")
        # print(f"sum immediate_heightfield: {np.sum(self.immediate_heightfield)}")

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

   
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions

        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float32, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float32, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float32, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float32, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float32, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float32, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], dtype=torch.float32, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float32, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10]).to(dtype=torch.float32)
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13]).to(dtype=torch.float32)
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec).to(dtype=torch.float32)
      

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float32, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()

        # Convert heightfield to the correct format
        heightfield = self.terrain.height_field_raw.astype(np.int16)  # Ensure int16
        
        # Ensure the heightfield is in column-major order
        heightfield = heightfield.T.copy()  
        
        # Set up parameters
        hf_params.column_scale = self.cfg.terrain.horizontal_scale
        hf_params.row_scale = self.cfg.terrain.horizontal_scale
        hf_params.vertical_scale = self.cfg.terrain.vertical_scale
        
        # Set dimensions
        hf_params.nbRows = self.terrain.tot_rows
        hf_params.nbColumns = self.terrain.tot_cols
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.border
        hf_params.transform.p.y = -self.terrain.border

        # Set transform
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        
        # Set material properties
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        print(f"Adding heightfield with shape: {heightfield.shape}")
        print(f"Height range after scaling: {np.min(heightfield) * self.cfg.terrain.vertical_scale:.3f}m to {np.max(heightfield) * self.cfg.terrain.vertical_scale:.3f}m")
        
        # Add the heightfield
        self.gym.add_heightfield(self.sim, heightfield.flatten(), hf_params)
        self.height_samples = torch.tensor(heightfield).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        print("Adding trimesh to simulation...")
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)  
        print("Trimesh added")
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
      
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
     
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)


    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_forward_vel(self):
        # Get the forward velocity in the robot's local frame
        forward_vel = self.base_lin_vel[:, 0]  # x-axis velocity
        # Reward positive forward velocity
        return torch.clamp(forward_vel, min=0.0)

    def _reward_target_dists(self):
        """
        Rewards based on how close the current feet (foot_positions) are to target feet positions (self.target_positions)
        """
        foot_positions = self.rigid_body_states[:, self.feet_indices, :3]

        distances = torch.norm(foot_positions - self.target_positions_one, dim=-1) + torch.norm(foot_positions - self.target_positions_two, dim=-1)  # shape (num_envs, num_feet)
    
        total_distance = torch.sum(distances, dim=1)  # shape (num_envs)
        
        # distance to reward (closer = higher reward)
        # TODO: tune this (especially 0.1 scaling factor)
        reward = torch.exp(-total_distance / 0.1)  
        
        return reward
    
    # for base PPO encouraging x and y positive movement and same z (not falling into the void)
    def _reward_xy_progress(self):
        # Reward for making progress in x and y directions
        # Get the change in position since last step
        xy_vel = self.base_lin_vel[:, :2]  # Get x,y velocities
        return torch.sum(xy_vel, dim=1)  # Sum the x and y velocities

    def _reward_z_stability(self):
        # Reward for maintaining target z height
        # Get the current z position
        z_pos = self.root_states[:, 2]
        # Calculate distance from target height
        z_target = self.cfg.init_state.pos[2]  # Use initial z position as target
        z_diff = torch.abs(z_pos - z_target)
        # Convert distance to reward (closer = higher reward)
        return torch.exp(-z_diff * 10)  # Scale factor of 10 makes the reward more sensitive to height changes

    def _draw_debug_vis(self):
        """ Add debug visualizations for waypoints and footsteps """
        # Delete previous visualizations
        self.gym.clear_lines(self.viewer)
        
        # Draw cross markers for foot positions
        line_length = 0.1  # 10cm lines
        colors = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.]]  # One color per foot
        
        for env_idx in range(self.num_envs):
            # Get foot positions from the simulation
            foot_positions = []
            for foot_idx in self.feet_indices:
                body_state = self.gym.get_rigid_body_state(self.envs[env_idx], 
                                                        self.actor_handles[env_idx], 
                                                        foot_idx)
                foot_positions.append(body_state[:3])

            # Draw crosses for each foot
            for pos, color in zip(foot_positions, colors):
                # Vertical line
                self.gym.add_lines(self.viewer, 
                                self.envs[env_idx],
                                1,
                                [pos[0], pos[1], pos[2] + line_length],
                                [pos[0], pos[1], pos[2] - line_length],
                                color)
                
                # Horizontal line
                self.gym.add_lines(self.viewer, 
                                self.envs[env_idx],
                                1,
                                [pos[0] + line_length, pos[1], pos[2]],
                                [pos[0] - line_length, pos[1], pos[2]],
                                color)
                
                # Forward-back line
                self.gym.add_lines(self.viewer, 
                                self.envs[env_idx],
                                1,
                                [pos[0], pos[1] + line_length, pos[2]],
                                [pos[0], pos[1] - line_length, pos[2]],
                                color)

    def _draw_target_positions(self, target_positions, colors=None):
        """Draw target positions for footsteps
        
        Args:
            target_positions: Tensor of shape (num_envs, num_feet, 3) containing target positions
            colors: Optional list of colors for each foot
        """
        if colors is None:
            colors = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.]]
        
        line_length = 0.05  # 5cm lines
        
        for env_idx in range(self.num_envs):
            for foot_idx in range(len(self.feet_indices)):
                pos = target_positions[env_idx, foot_idx]
                color = colors[foot_idx]
                
                # Vertical line
                self.gym.add_lines(self.viewer, 
                                self.envs[env_idx],
                                1,
                                [pos[0], pos[1], pos[2] + line_length],
                                [pos[0], pos[1], pos[2] - line_length],
                                color)
                
                # Horizontal line
                self.gym.add_lines(self.viewer, 
                                self.envs[env_idx],
                                1,
                                [pos[0] + line_length, pos[1], pos[2]],
                                [pos[0] - line_length, pos[1], pos[2]],
                                color)
                
    def update_target_positions(self, positions):
        """Update the target positions for visualization
        
        Args:
            positions: Tensor of shape (num_envs, num_feet, 3) containing target positions
        """
        self.target_positions = positions

    def _update_local_heightfield(self):
        """Updates self.immediate_heightfield with height values in a 0.56m x 0.56m area around the robot, 
        sampled to a 14x14 grid"""
        # Physical dimensions desired
        window_size_meters = 1.11  # 3m x 3m window (TODO: Adjust however much you want)
        samples_per_side = 17      # 14 x 14 grid (TODO: adjust however much u want)
        
        # Calculate sampling interval in meters
        sample_spacing = window_size_meters / (samples_per_side - 1)  # Distance between samples
        
        # Get robot's current position in world coordinates
        base_pos = self.root_states[0, 0:3]  # Using first env's robot position
        
        # Initialize the result grid
        self.immediate_heightfield = np.zeros((samples_per_side, samples_per_side), dtype=np.int16)
        
        # Calculate world coordinates for each sample point
        for i in range(samples_per_side):
            for j in range(samples_per_side):
                # Calculate world position for this sample
                # Center the window on the robot and offset by sample position
                world_x = base_pos[0] + (i - samples_per_side/2) * sample_spacing
                world_y = base_pos[1] + (j - samples_per_side/2) * sample_spacing
                
                # Convert world coordinates to heightfield indices
                hf_row = int((world_x + self.terrain.cfg.border_size) / self.terrain.cfg.horizontal_scale)
                hf_col = int((world_y + self.terrain.cfg.border_size) / self.terrain.cfg.horizontal_scale)
                
                # Check bounds and sample height
                if (0 <= hf_row < self.terrain.height_field_raw.shape[0] and 
                    0 <= hf_col < self.terrain.height_field_raw.shape[1]):
                    self.immediate_heightfield[i, j] = self.terrain.height_field_raw[hf_row, hf_col]

    def _check_feet_at_target(self, feet_positions, target_positions, threshold=0.02):
        """
        Check if feet are within threshold distance of their targets
        Returns: Boolean tensor of shape (num_envs,)
        """
        distances = torch.norm(feet_positions - target_positions, dim=-1)  # Shape: (num_envs, num_feet)
        return torch.all(distances < threshold, dim=-1)  # Shape: (num_envs,)

    def _ensure_float32(self, tensor):
        if tensor.dtype != torch.float32:
            return tensor.to(dtype=torch.float32)
        return tensor