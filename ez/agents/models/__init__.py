# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn as nn
from ez.utils.format import normalize_state
from ez.utils.format import formalize_obs_lst, DiscreteSupport, allocate_gpu, prepare_obs_lst, symexp


class EfficientZero(nn.Module):
    def __init__(self,
                 representation_model,
                 dynamics_model,
                 reward_prediction_model,
                 value_policy_model,
                 projection_model,
                 projection_head_model,
                 config,
                 **kwargs,
                 ):
        """The basic models in EfficientZero
        Parameters
        ----------
        representation_model: nn.Module
            represent the observations
        dynamics_model: SpectralDynamicsNetwork
            predicts next latent state from (state_seq, action_seq) of shape (B, L, ...)
        reward_prediction_model: nn.Module
            predict the reward given the next state
        value_policy_model: nn.Module
            predict value and policy from state
        projection_model / projection_head_model: nn.Module
            for consistency loss projection
        kwargs: dict
            state_norm: bool  -- normalize latent states
            value_prefix: bool -- predict value prefix instead of reward
        """
        super().__init__()

        self.representation_model = representation_model
        self.dynamics_model = dynamics_model          # SpectralDynamicsNetwork
        self.reward_prediction_model = reward_prediction_model
        self.value_policy_model = value_policy_model
        self.projection_model = projection_model
        self.projection_head_model = projection_head_model
        self.config = config
        self.state_norm = kwargs.get('state_norm')
        self.value_prefix = kwargs.get('value_prefix')
        self.v_num = config.train.v_num

    def do_representation(self, obs):
        state = self.representation_model(obs)
        if self.state_norm:
            state = normalize_state(state)
        return state

    def do_dynamics(self, state_seq, action_seq):
        """
        SpectralDynamicsNetwork forward pass.

        Args:
            state_seq:  (B, L, C, H, W) or (B, L, D) -- history of latent states
            action_seq: (B, L, action_dim) -- history of raw actions
        Returns:
            next_state: (B, C, H, W) or (B, D)
        """
        next_state = self.dynamics_model(state_seq, action_seq)
        if self.state_norm:
            next_state = normalize_state(next_state)
        return next_state

    def do_reward_prediction(self, next_state, reward_hidden=None):
        if self.value_prefix:
            value_prefix, reward_hidden = self.reward_prediction_model(next_state, reward_hidden)
            return value_prefix, reward_hidden
        else:
            reward = self.reward_prediction_model(next_state)
            return reward, None

    def do_value_policy_prediction(self, state):
        value, policy = self.value_policy_model(state)
        return value, policy

    def do_projection(self, state, with_grad=True):
        proj = self.projection_model(state)
        if with_grad:
            proj = self.projection_head_model(proj)
            return proj
        else:
            return proj.detach()

    def initial_inference(self, obs, training=False):
        state = self.do_representation(obs)
        values, policy = self.do_value_policy_prediction(state)

        if training:
            return state, values, policy

        if self.v_num > 2:
            values = values[np.random.choice(self.v_num, 2, replace=False)]
        if self.config.model.value_support.type == 'symlog':
            output_values = symexp(values).min(0)[0]
        else:
            output_values = DiscreteSupport.vector_to_scalar(values, **self.config.model.value_support).min(0)[0]

        if self.config.env.env in ['DMC', 'Gym']:
            output_values = output_values.clip(0, 1e5)

        return state, output_values, policy

    def recurrent_inference(self, state_seq, action_seq, reward_hidden, training=False):
        """
        One-step recurrent inference using SpectralDynamicsNetwork.

        Args:
            state_seq:  (B, L, C, H, W)  -- sliding window of latent states
            action_seq: (B, L, action_dim) -- sliding window of raw actions
            reward_hidden: LSTM hidden state for value-prefix prediction
            training: bool -- if True, return raw (un-decoded) values for loss

        Returns (training=True):
            next_state, value_prefix, values, policy, reward_hidden
        Returns (training=False):
            next_state, value_prefix (scalar), output_values (scalar), policy, reward_hidden
        """
        next_state = self.do_dynamics(state_seq, action_seq)

        value_prefix, reward_hidden = self.do_reward_prediction(next_state, reward_hidden)
        values, policy = self.do_value_policy_prediction(next_state)

        if training:
            return next_state, value_prefix, values, policy, reward_hidden

        if self.v_num > 2:
            values = values[np.random.choice(self.v_num, 2, replace=False)]
        if self.config.model.value_support.type == 'symlog':
            output_values = symexp(values).min(0)[0]
        else:
            output_values = DiscreteSupport.vector_to_scalar(values, **self.config.model.value_support).min(0)[0]

        if self.config.env.env in ['DMC', 'Gym']:
            output_values = output_values.clip(0, 1e5)

        if self.config.model.reward_support.type == 'symlog':
            value_prefix = symexp(value_prefix)
        else:
            value_prefix = DiscreteSupport.vector_to_scalar(value_prefix, **self.config.model.reward_support)
        return next_state, value_prefix, output_values, policy, reward_hidden

    def get_weights(self, part='none'):
        if part == 'reward':
            weights = self.reward_prediction_model.state_dict()
        else:
            weights = self.state_dict()
        return {k: v.cpu() for k, v in weights.items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)