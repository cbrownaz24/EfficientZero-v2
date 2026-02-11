# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import math
import torch.nn as nn
import numpy as np
from .osf_predictor import OSFPredictor
from .layer import ResidualBlock, conv3x3, mlp


# Predict next hidden state given a *sequence* of latent states and raw actions
# via Observation Spectral Filtering (OSF).
class SpectralDynamicsNetwork(nn.Module):
    def __init__(self, num_channels, action_space_size, state_shape, sequence_length=5, is_continuous=False,
                    action_embedding=False, action_embedding_dim=32, num_filters=24):
        super().__init__()
        self.num_channels = num_channels
        self.action_space_size = action_space_size
        self.state_shape = state_shape
        self.sequence_length = sequence_length
        self.is_continuous = is_continuous
        self.action_embedding = action_embedding
        self.action_embedding_dim = action_embedding_dim
        
        H, W = state_shape[-2], state_shape[-1]
        
        in_ch = action_space_size if is_continuous else 1
        self.conv1x1 = nn.Conv2d(in_ch, action_embedding_dim, 1)
        self.ln = nn.LayerNorm([sequence_length, action_embedding_dim, H, W])
        
        D_in = action_embedding_dim * H * W
        D_out = num_channels * H * W
        
        m = min(sequence_length, 3)
        h = min(num_filters, sequence_length)
        
        self.osf_model = OSFPredictor(T=sequence_length, h=h, m=m, D_in=D_in, D_out=D_out)
    
    def forward(self, states, actions):
        """
        Args:
            states:  (B, L, C, H, W)  -- history of latent states
            actions: (B, L, 1) or (B, L, action_space_size) -- history of raw actions
        Returns:
            next_state: (B, C, H, W)
        """
        B = states.shape[0]
        L = self.sequence_length
        H, W = states.shape[-2], states.shape[-1]
        
        actions = actions.reshape(*actions.shape, 1, 1).repeat(1, 1, 1, H, W) # (B, L, 1, H, W) or (B, L, action_space_size, H, W)
        actions = actions.view(-1, *(actions.shape[2:])) # (B*L, 1, H, W) or (B*L, action_space_size, H, W)
        action_embeddings = self.conv1x1(actions) # (B*L, D, H, W) 
        action_embeddings = action_embeddings.view(B, L, *(action_embeddings.shape[1:])) # (B, L, D, H, W)
        action_embeddings = self.ln(action_embeddings) # (B, L, D, H, W)
        action_embeddings = torch.functional.relu(action_embeddings)
        action_embeddings = action_embeddings.view(B, L, -1)  # (B, L, D*H*W)
        
        states = states.view(B, L, -1) # (B, L, C*H*W)

        next_state = self.osf_model(u_seq=action_embeddings, y_seq=states)  # (B, C*H*W)
        next_state = next_state.view(B, self.num_channels, H, W) # (B, C, H, W)
        
        return next_state


# 1D (flat vector) variant of SpectralDynamicsNetwork for state-based environments.
# Hidden states are flat vectors of shape (B, D) rather than spatial (B, C, H, W).
class SpectralDynamicsNetwork1D(nn.Module):
    def __init__(self, hidden_shape, action_space_size, sequence_length=5,
                 action_embedding_dim=32, num_filters=24):
        """
        Args:
            hidden_shape: int, dimensionality of the latent state vector
            action_space_size: int, dimensionality of the action space
            sequence_length: int, T for OSF
            action_embedding_dim: int, embedding dim for actions
            num_filters: int, number of spectral filters
        """
        super().__init__()
        self.hidden_shape = hidden_shape
        self.action_space_size = action_space_size
        self.sequence_length = sequence_length

        # Action embedding MLP: action_space_size -> action_embedding_dim
        self.act_embed = nn.Sequential(
            nn.Linear(action_space_size, action_embedding_dim),
            nn.LayerNorm(action_embedding_dim),
            nn.ReLU(),
        )

        D_in = action_embedding_dim
        D_out = hidden_shape

        m = min(sequence_length, 3)
        h = min(num_filters, sequence_length)

        self.osf_model = OSFPredictor(T=sequence_length, h=h, m=m, D_in=D_in, D_out=D_out)

    def forward(self, states, actions):
        """
        Args:
            states:  (B, L, D)  -- history of latent state vectors
            actions: (B, L, action_dim) -- history of raw actions
        Returns:
            next_state: (B, D)
        """
        B, L, _ = states.shape

        # Embed actions: (B, L, action_dim) -> (B, L, action_embedding_dim)
        action_embeddings = self.act_embed(actions.view(B * L, -1))
        action_embeddings = action_embeddings.view(B, L, -1)

        # OSF: predict next state
        next_state = self.osf_model(u_seq=action_embeddings, y_seq=states)  # (B, D)

        return next_state


''' ------------------------------------------------------------------------------------------------ '''
''' ------------------------------------------------------------------------------------------------ '''
''' ------------------------------------------------------------------------------------------------ '''
''' ------------------------------------------------------------------------------------------------ '''
''' ---------------------------------------- DO NOT TOUCH ------------------------------------------ '''
''' ------------------------------------------------------------------------------------------------ '''
''' ------------------------------------------------------------------------------------------------ '''
''' ------------------------------------------------------------------------------------------------ '''
# Down_sample observations before representation network (See paper appendix Network Architecture)
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.resblocks1 = nn.ModuleList(
            [ResidualBlock(out_channels // 2, out_channels // 2) for _ in range(1)]
        )
        self.conv2 = nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.downsample_block = ResidualBlock(out_channels // 2, out_channels, downsample=self.conv2, stride=2)
        self.resblocks2 = nn.ModuleList(
            [ResidualBlock(out_channels, out_channels) for _ in range(1)]
        )
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = nn.ModuleList(
            [ResidualBlock(out_channels, out_channels) for _ in range(1)]
        )
        self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.downsample_block(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x

# Encode the observations into hidden states
class RepresentationNetwork(nn.Module):
    def __init__(self, observation_shape, num_blocks, num_channels, downsample):
        """
        Representation network
        :param observation_shape: tuple or list, shape of observations: [C, W, H]
        :param num_blocks: int, number of res blocks
        :param num_channels: int, channels of hidden states
        :param downsample: bool, True -> do downsampling for observations. (For board games, do not need)
        """
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            self.downsample_net = DownSample(
                observation_shape[0],
                num_channels,
            )
        else:
            self.conv = conv3x3(
                observation_shape[0],
                num_channels,
            )
            self.bn = nn.BatchNorm2d(num_channels)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        return x

class ValuePolicyNetwork(nn.Module):
    def __init__(self, num_blocks, num_channels, reduced_channels, flatten_size, fc_layers, value_output_size,
                 policy_output_size, init_zero, is_continuous=False, policy_distribution='beta', **kwargs):
        super().__init__()
        self.v_num = kwargs.get('v_num')
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )
        self.conv1x1_values = nn.ModuleList([nn.Conv2d(num_channels, reduced_channels, 1) for _ in range(self.v_num)])
        self.conv1x1_policy = nn.Conv2d(num_channels, reduced_channels, 1)
        self.bn_values = nn.ModuleList([nn.BatchNorm2d(reduced_channels) for _ in range(self.v_num)])
        self.bn_policy = nn.BatchNorm2d(reduced_channels)
        self.block_output_size_value = flatten_size
        self.block_output_size_policy = flatten_size
        self.fc_values = nn.ModuleList([mlp(self.block_output_size_value, fc_layers, value_output_size,
                            init_zero=False if is_continuous else init_zero) for _ in range(self.v_num)])
        self.fc_policy = mlp(self.block_output_size_policy, fc_layers if not is_continuous else [64],
                             policy_output_size, init_zero=init_zero)

        self.is_continuous = is_continuous
        self.init_std = 1.0
        self.min_std = 0.1

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)

        values = []
        for i in range(self.v_num):
            value = self.conv1x1_values[i](x)
            value = self.bn_values[i](value)
            value = nn.functional.relu(value)
            value = value.reshape(-1, self.block_output_size_value)
            value = self.fc_values[i](value)
            values.append(value)

        policy = self.conv1x1_policy(x)
        policy = self.bn_policy(policy)
        policy = nn.functional.relu(policy)
        policy = policy.reshape(-1, self.block_output_size_policy)
        policy = self.fc_policy(policy)

        if self.is_continuous:
            action_space_size = policy.shape[-1] // 2
            policy[:, :action_space_size] = 5 * torch.tanh(policy[:, :action_space_size] / 5)  # soft clamp mu
            policy[:, action_space_size:] = (torch.nn.functional.softplus(policy[:, action_space_size:] + self.init_std) + self.min_std)#.clip(0, 5)  # same as Dreamer-v3

        return torch.stack(values), policy

class SupportNetwork(nn.Module):
    def __init__(self, num_blocks, num_channels, reduced_channels, flatten_size, fc_layers, output_support_size, init_zero):
        super().__init__()
        self.flatten_size = flatten_size

        self.conv1x1 = nn.Conv2d(num_channels, reduced_channels, 1)
        self.bn = nn.BatchNorm2d(reduced_channels)
        self.fc = mlp(flatten_size, fc_layers, output_support_size, init_zero=init_zero)

    def forward(self, x):

        x = self.conv1x1(x)
        x = self.bn(x)
        x = nn.functional.relu(x)
        x = x.reshape(-1, self.flatten_size)
        x = self.fc(x)
        return x


class SupportLSTMNetwork(nn.Module):
    def __init__(self, num_blocks, num_channels, reduced_channels, flatten_size, fc_layers, output_support_size, lstm_hidden_size, init_zero):
        super().__init__()
        self.flatten_size = flatten_size

        self.conv1x1_reward = nn.Conv2d(num_channels, reduced_channels, 1)
        self.bn_reward = nn.BatchNorm2d(reduced_channels)
        self.lstm = nn.LSTM(input_size=flatten_size, hidden_size=lstm_hidden_size)
        self.bn_reward_sum = nn.BatchNorm1d(lstm_hidden_size)
        self.fc = mlp(lstm_hidden_size, fc_layers, output_support_size, init_zero=init_zero)

    def forward(self, x, hidden):

        x = self.conv1x1_reward(x)
        x = self.bn_reward(x)
        x = nn.functional.relu(x)
        x = x.reshape(-1, self.flatten_size).unsqueeze(0)
        x, hidden = self.lstm(x, hidden)
        x = x.squeeze(0)
        x = self.bn_reward_sum(x)
        x = nn.functional.relu(x)
        x = self.fc(x)
        return x, hidden


class ProjectionNetwork(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super().__init__()

        self.input_dim = input_dim
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),

            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),

            nn.Linear(hid_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.layer(x)


class ProjectionHeadNetwork(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        return self.layer(x)
