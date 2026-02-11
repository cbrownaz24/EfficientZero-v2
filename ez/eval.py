# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import os
import sys
sys.path.append(os.getcwd())

import time
import math
import torch
import ray
import copy
import cv2
import hydra
import multiprocessing
import numpy as np
import imageio
from PIL import Image, ImageDraw

from pathlib import Path
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from torch.cuda.amp import autocast as autocast
import torch.nn.functional as F
from ez import mcts
from ez import agents
from ez.envs import make_envs
from ez.utils.format import formalize_obs_lst, DiscreteSupport, prepare_obs_lst, symexp, profile
from ez.mcts.cy_mcts import Gumbel_MCTS
from ez.utils.distribution import SquashedNormal, TruncatedNormal
from ez.utils.latent_buffer import LatentStateActionBuffer

@hydra.main(config_path="./config", config_name='config', version_base='1.1')
def main(config):
    if config.exp_config is not None:
        exp_config = OmegaConf.load(config.exp_config)
        config = OmegaConf.merge(config, exp_config)

    # update config
    agent = agents.names[config.agent_name](config)

    num_gpus = torch.cuda.device_count()
    num_cpus = multiprocessing.cpu_count()
    ray.init(num_gpus=num_gpus, num_cpus=num_cpus,
             object_store_memory= 150 * 1024 * 1024 * 1024 if config.env.image_based else 100 * 1024 * 1024 * 1024)

    # prepare model
    model = agent.build_model()
    if os.path.exists(config.eval.model_path):
        weights = torch.load(config.eval.model_path)
        model.load_state_dict(weights)
        print('resume model from: ', config.eval.model_path)
    if int(torch.__version__[0]) == 2:
        model = torch.compile(model)

    n_episodes = 1
    save_path = Path(config.eval.save_path)

    eval(agent, model, n_episodes, save_path, config,
         max_steps=27000,
         use_pb=True, verbose=config.eval.verbose)

@torch.no_grad()
def eval(agent, model, n_episodes, save_path, config, max_steps=None, use_pb=False, verbose=0):
    model.cuda()
    model.eval()

    # prepare logs
    if save_path is not None:
        video_path = save_path / 'recordings'
        video_path.mkdir(parents=True, exist_ok=True)
    else:
        video_path = None

    dones = np.array([False for _ in range(n_episodes)])
    if use_pb:
        pb = tqdm(np.arange(max_steps), leave=True)
    ep_ori_rewards = np.zeros(n_episodes)

    # make env
    if max_steps is not None:
        config.env.max_episode_steps = max_steps
    envs = make_envs(config.env.env, config.env.game, n_episodes, config.env.base_seed, save_path=video_path,
                     episodic_life=False, **config.env)

    # initialization
    stack_obs_windows, game_trajs = agent.init_envs(envs, max_steps)

    # set infinity trajectory size
    [traj.set_inf_len() for traj in game_trajs]

    # --- Initialize latent buffers for SpectralDynamicsNetwork ---
    seq_len = config.model.spectral_dynamics.sequence_length
    if config.env.image_based:
        if config.model.down_sample:
            state_shape = (config.model.num_channels,
                           math.ceil(config.env.obs_shape[1] / 16),
                           math.ceil(config.env.obs_shape[2] / 16))
        else:
            state_shape = (config.model.num_channels,
                           config.env.obs_shape[1],
                           config.env.obs_shape[2])
    else:
        # State-based: hidden state is a flat vector
        state_shape = (config.model.hidden_shape,)
    action_dim = 1 if config.env.env == 'Atari' else config.env.action_space_size
    latent_buffers = [
        LatentStateActionBuffer(seq_len, state_shape, action_dim, device='cuda')
        for _ in range(n_episodes)
    ]

    # begin to evaluate
    step = 0
    frames = [[] for _ in range(n_episodes)]
    rewards = [[] for _ in range(n_episodes)]
    while not dones.all():
        # debug
        if verbose:
            import ipdb
            ipdb.set_trace()

        # stack obs
        current_stacked_obs = formalize_obs_lst(stack_obs_windows, image_based=config.env.image_based)
        # obtain the statistics at current steps
        with torch.no_grad():
            with autocast():
                states, values, policies = model.initial_inference(current_stacked_obs)

        values = values.detach().cpu().numpy().flatten()

        # Push root latent states into latent buffers (state only, action not yet known)
        for i in range(n_episodes):
            if not dones[i]:
                latent_buffers[i].push_state_only(states[i])


        # tree search for policies
        tree = mcts.names[config.mcts.language](
            # num_actions=config.env.action_space_size if config.env.env == 'Atari' else config.mcts.num_top_actions,
            num_actions=config.env.action_space_size if config.env.env == 'Atari' else config.mcts.num_sampled_actions,
            discount=config.rl.discount,
            env=config.env.env,
            **config.mcts,  # pass mcts related params
            **config.model,  # pass the value and reward support params
        )
        if config.env.env == 'Atari':
            if config.mcts.use_gumbel:
                r_values, r_policies, best_actions, _ = tree.search(model, n_episodes, states, values, policies,
                                                                    use_gumble_noise=False, verbose=verbose,
                                                                    latent_buffers=latent_buffers)
            else:
                r_values, r_policies, best_actions, _ = tree.search_ori_mcts(model, n_episodes, states, values, policies,
                                                                                use_noise=False,
                                                                                latent_buffers=latent_buffers)
        else:
            r_values, r_policies, best_actions, _, _, _ = tree.search_continuous(
                    model, n_episodes, states, values, policies,
                    use_gumble_noise=False, verbose=verbose, add_noise=False
                )

        # step action in environments
        for i in range(n_episodes):
            if dones[i]:
                continue

            action = best_actions[i]
            obs, reward, done, info = envs[i].step(action)
            frames[i].append(obs if config.env.image_based else envs[i].render(mode='rgb_array'))
            # rewards[i].append(reward)
            rewards[i].append(info['raw_reward'])
            dones[i] = done

            # Record the played action in the persistent latent buffer
            act_tensor = torch.tensor([action], device='cuda').float()
            if act_tensor.shape[0] != action_dim:
                act_tensor = act_tensor[:action_dim]
            latent_buffers[i].action_buffer[-1] = act_tensor

            # save data to trajectory buffer
            game_trajs[i].store_search_results(values[i], r_values[i], r_policies[i])
            game_trajs[i].append(action, obs, reward)
            if config.env.env == 'Atari':
                game_trajs[i].snapshot_lst.append(envs[i].ale.cloneState())
            else:
                game_trajs[i].snapshot_lst.append(envs[i].physics.get_state())

            del stack_obs_windows[i][0]
            stack_obs_windows[i].append(obs)

            # log
            ep_ori_rewards[i] += info['raw_reward']

        step += 1
        if use_pb:
            pb.set_description('{} In step {}, take action {}, scores: {}(max: {}, min: {}) currently.'
                               ''.format(config.env.game, step, best_actions,
                                         ep_ori_rewards.mean(), ep_ori_rewards.max(), ep_ori_rewards.min()))
            pb.update(1)

    [env.close() for env in envs]
    for i in range(n_episodes):
        writer = imageio.get_writer(video_path / f'epi_{i}_{max_steps}.mp4')
        rewards[i][0] = sum(rewards[i])

        j = 0
        for frame, reward in zip(frames[i], rewards[i]):
            frame = Image.fromarray(frame)
            draw = ImageDraw.Draw(frame)
            if config.env.game == 'hopper_hop':
                draw.text((5, 5), f'mu={game_trajs[i].action_lst[j][0]:.2f},{game_trajs[i].action_lst[j][1]:.2f}')
                draw.text((5, 20), f'{game_trajs[i].action_lst[j][2]:.2f},{game_trajs[i].action_lst[j][3]:.2f}')
                draw.text((5, 35), f'r={reward:.2f}')

            frame = np.array(frame)
            writer.append_data(frame)
            j += 1
        writer.close()

    return ep_ori_rewards



if __name__=='__main__':
    main()
