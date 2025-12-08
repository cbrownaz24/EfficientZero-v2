import torch
import torch.nn as nn
import numpy as np
from .layer import ResidualBlock, conv3x3, mlp

try:
    from mini_stu import MiniSTU
except ImportError:
    raise ImportError(
        "mini_stu library is required but not installed. "
        "Please install it with: pip install mini-stu"
    )


class MiniSTUDynamicsNetwork(nn.Module):
    def __init__(self, num_blocks, num_channels, action_space_size, state_shape,
                 sequence_length=5, is_continuous=False, action_embedding=False,
                 action_embedding_dim=32, use_mlp=True, mlp_hidden_dim=None,
                 num_filters=24, mlp_num_layers=2, mlp_dropout=0.1, mlp_activation='gelu'):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.action_space_size = action_space_size
        self.state_shape = state_shape
        self.sequence_length = sequence_length
        self.is_continuous = is_continuous
        self.action_embedding = action_embedding
        self.action_embedding_dim = action_embedding_dim
        
        # Determine action input dimension
        action_input_dim = action_space_size if is_continuous else 1
        
        # MiniSTU processes flattened state representation
        # state_dim is the flattened size of the hidden state
        state_dim = num_channels * state_shape[1] * state_shape[2]
        
        # CRITICAL: In MiniSTU library, num_filters must equal seq_len
        # This is a constraint of the spectral temporal unit implementation
        effective_num_filters = sequence_length
        
        # Create MiniSTU instance with proper configuration
        # This uses the imported MiniSTU library
        self.mini_stu = MiniSTU(
            seq_len=sequence_length,
            num_filters=effective_num_filters,  # Must equal sequence_length
            input_dim=action_input_dim,
            output_dim=state_dim,
            use_mlp=use_mlp,
            mlp_hidden_dim=(state_dim * 2) if mlp_hidden_dim is None else mlp_hidden_dim,
            mlp_num_layers=mlp_num_layers,
            mlp_dropout=mlp_dropout,
            mlp_activation=mlp_activation
        )
        
        # State refinement network: takes base prediction and refines it
        # using residual blocks
        self.refine_conv = conv3x3(num_channels, num_channels)
        self.refine_bn = nn.BatchNorm2d(num_channels)
        self.refine_blocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(max(1, num_blocks // 2))]
        )
    
    def forward(self, state, action, action_history=None):
        batch_size = state.shape[0]
        state_h, state_w = state.shape[2], state.shape[3]
        
        # If action_history is not provided, use fallback to current action
        if action_history is None:
            # Fallback: repeat current action to fill sequence length
            if not self.is_continuous:
                # Discrete: action is (batch, 1) -> expand to (batch, seq_len, 1)
                action_history = action.unsqueeze(1).repeat(1, self.sequence_length, 1).float()
            else:
                # Continuous: action is (batch, action_dim) -> expand to (batch, seq_len, action_dim)
                action_history = action.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Ensure action_history has correct shape (batch, sequence_length, action_dim)
        if action_history.dim() == 2:
            # (batch, seq_len) -> (batch, seq_len, 1)
            action_history = action_history.unsqueeze(-1)
        
        # Convert to float if needed
        action_history = action_history.float()
        
        # Process action sequence through MiniSTU
        # Input: (batch, sequence_length, action_dim)
        # Output: (batch, sequence_length, state_dim)
        state_pred_seq = self.mini_stu(action_history)
        
        # Take the last prediction (for next state)
        # Shape: (batch, state_dim)
        state_pred_flat = state_pred_seq[:, -1, :]
        
        # Reshape to spatial dimensions
        # Shape: (batch, num_channels, state_h, state_w)
        state_pred = state_pred_flat.view(batch_size, self.num_channels, state_h, state_w)
        
        # Refine prediction using residual blocks
        x = state_pred
        x = self.refine_conv(x)
        x = self.refine_bn(x)
        x = nn.functional.relu(x)
        
        for block in self.refine_blocks:
            x = block(x)
        
        # Add residual connection from input state prediction
        next_state = x + state_pred
        next_state = nn.functional.relu(next_state)
        
        return next_state


class DynamicsNetworkWrapper(nn.Module):
    def __init__(self, use_mini_stu, original_dynamics, mini_stu_dynamics=None):
        super().__init__()
        self.use_mini_stu = use_mini_stu
        self.original_dynamics = original_dynamics
        self.mini_stu_dynamics = mini_stu_dynamics
        
        if use_mini_stu and mini_stu_dynamics is None:
            raise ValueError("mini_stu_dynamics must be provided when use_mini_stu=True")
    
    def forward(self, state, action, action_history=None):
        if self.use_mini_stu:
            return self.mini_stu_dynamics(state, action, action_history)
        else:
            return self.original_dynamics(state, action)
