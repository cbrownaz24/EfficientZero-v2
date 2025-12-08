# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

"""
Action History Buffer Management

Maintains historical buffers of actions during trajectory collection and training
for use with MiniSTU-based dynamics models.
"""

import torch
import numpy as np


class ActionHistoryBuffer:
    def __init__(self, batch_size, sequence_length, action_dim, is_continuous=False, device='cuda'):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.action_dim = action_dim
        self.is_continuous = is_continuous
        self.device = device
        
        # Initialize buffer: (batch_size, sequence_length, action_dim)
        if is_continuous:
            self.buffer = torch.zeros(
                batch_size, sequence_length, action_dim,
                dtype=torch.float32, device=device
            )
        else:
            # For discrete actions, use dimension 1
            self.buffer = torch.zeros(
                batch_size, sequence_length, 1,
                dtype=torch.long, device=device
            )
        
        self.filled = 0  # How many timesteps have been recorded
    
    def reset(self):
        """Reset the buffer"""
        self.buffer.zero_()
        self.filled = 0
    
    def push(self, actions):
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        
        # Shift buffer left and add new actions at the end
        self.buffer = torch.cat([self.buffer[:, 1:, :], actions.unsqueeze(1)], dim=1)
        self.filled = min(self.filled + 1, self.sequence_length)
    
    def get_history(self, pad_with_zeros=True):
        if pad_with_zeros or self.filled >= self.sequence_length:
            return self.buffer.clone()
        else:
            # Return only filled portion (shouldn't normally happen)
            return self.buffer.clone()
    
    def to(self, device):
        self.device = device
        self.buffer = self.buffer.to(device)
        return self
    
    def clone(self):
        new_buffer = ActionHistoryBuffer(
            self.batch_size, self.sequence_length, self.action_dim,
            self.is_continuous, self.device
        )
        new_buffer.buffer = self.buffer.clone()
        new_buffer.filled = self.filled
        return new_buffer


class ActionHistoryManager:
    def __init__(self, sequence_length, action_dim, is_continuous=False, device='cuda'):
        self.sequence_length = sequence_length
        self.action_dim = action_dim
        self.is_continuous = is_continuous
        self.device = device
        self.buffers = {}
    
    def create_buffer(self, name, batch_size):
        self.buffers[name] = ActionHistoryBuffer(
            batch_size, self.sequence_length, self.action_dim,
            self.is_continuous, self.device
        )
    
    def get_buffer(self, name):
        return self.buffers.get(name)
    
    def push_action(self, name, actions):
        if name not in self.buffers:
            raise KeyError(f"Buffer '{name}' not found")
        self.buffers[name].push(actions)
    
    def get_history(self, name):
        if name not in self.buffers:
            raise KeyError(f"Buffer '{name}' not found")
        return self.buffers[name].get_history()
    
    def reset_buffer(self, name):
        if name not in self.buffers:
            raise KeyError(f"Buffer '{name}' not found")
        self.buffers[name].reset()
    
    def to(self, device):
        self.device = device
        for buffer in self.buffers.values():
            buffer.to(device)
        return self
