import torch
import torch.nn as nn
import numpy as np
from .layer import ResidualBlock, conv3x3, mlp
from ez.agents.models.base_model import *

try:
    from mini_stu import MiniSTU
except ImportError:
    raise ImportError(
        "mini_stu library is required but not installed. "
        "Please install it with: pip install mini-stu"
    )


class MiniSTUWorldModel(nn.Module):
    def __init__(self, num_blocks, num_channels, action_space_size, state_shape,
                 observation_shape, representation_net, sequence_length=5, is_continuous=False,
                 action_embedding=False, action_embedding_dim=32, use_mlp=True, 
                 mlp_hidden_dim=None, num_filters=24, mlp_num_layers=2, 
                 mlp_dropout=0.1, mlp_activation='gelu'):
        """
        Args:
            num_blocks: Number of residual blocks for state refinement
            num_channels: Dimensionality of latent states (state_dim for MiniSTU)
            action_space_size: Size of action space (discrete: num_actions, continuous: action_dim)
            state_shape: Shape of encoded states (without batch): (num_channels, H, W)
            observation_shape: Shape of raw observations: (C, H, W)
            representation_net: RepresentationNetwork instance for on-the-fly encoding
            sequence_length: Length of sequences processed by MiniSTU
            is_continuous: Whether action space is continuous
            action_embedding: Whether to use learned action embeddings
            action_embedding_dim: Dimension of action embeddings if used
            use_mlp: Whether MiniSTU uses MLP refinement
            mlp_hidden_dim: Hidden dimension for MiniSTU's MLP
            num_filters: MiniSTU num_filters (must equal sequence_length)
            mlp_num_layers: Number of MLP layers in MiniSTU
            mlp_dropout: Dropout rate for MiniSTU's MLP
            mlp_activation: Activation function for MiniSTU's MLP
        """
        super().__init__()
        self.num_blocks = num_blocks
        self.num_channels = num_channels  # This is state_dim
        self.action_space_size = action_space_size
        self.state_shape = state_shape
        self.observation_shape = observation_shape
        self.sequence_length = sequence_length
        self.is_continuous = is_continuous
        self.action_embedding = action_embedding
        self.action_embedding_dim = action_embedding_dim
        
        self.representation_net = representation_net
        
        if action_embedding:
            if is_continuous:
                self.action_embed = nn.Linear(action_space_size, action_embedding_dim)
            else:
                self.action_embed = nn.Embedding(action_space_size, action_embedding_dim)
            action_input_dim = action_embedding_dim
        else:
            action_input_dim = action_space_size if is_continuous else 1
        
        state_dim = num_channels * state_shape[1] * state_shape[2]
        
        # num_filters must equal seq_len
        effective_num_filters = sequence_length
        
        # Input: concatenated (flattened_encoded_state + action) at each timestep
        # Output: predicted flattened state at each timestep
        self.mini_stu = MiniSTU(
            seq_len=sequence_length,
            num_filters=effective_num_filters,  # Must equal sequence_length
            input_dim=state_dim + action_input_dim,  # Concatenated state-action pairs (flattened)
            output_dim=state_dim,  # Output is predicted flattened state
            use_mlp=use_mlp,
            mlp_hidden_dim=(state_dim * 2) if mlp_hidden_dim is None else mlp_hidden_dim,
            mlp_num_layers=mlp_num_layers,
            mlp_dropout=mlp_dropout,
            mlp_activation=mlp_activation
        )
        
        # Optional: State refinement network for post-processing predictions
        self.refine_conv = conv3x3(num_channels, num_channels)
        self.refine_bn = nn.BatchNorm2d(num_channels)
        self.refine_blocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(max(1, num_blocks // 2))]
        )
    
    def forward(self, obs_or_states_sequence, action_sequence, current_state=None):
        """
        Predict the next state given observation/state and action sequences.
        
        Args:
            obs_or_states_sequence: Either:
                - Raw observations (batch, seq_len, C, H, W) [TRAINING MODE]
                - Encoded states (batch, seq_len, num_channels, H', W') [PLANNING MODE]
            action_sequence: Action sequence (batch, seq_len, action_dim or 1)
            current_state: Optional current state for reference (unused but kept for compatibility)
        
        Returns:
            next_state: Predicted next state (batch, num_channels, H, W)
        
        The function auto-detects the input mode:
        - If 5D with C channels matching input_shape[0] → TRAINING MODE (raw observations)
        - If 5D with num_channels matching model's state channels → PLANNING MODE (encoded states)
        """
        batch_size = obs_or_states_sequence.shape[0]
        seq_len = obs_or_states_sequence.shape[1]
        
        # Ensure sequences have correct shape (5D: batch, seq_len, channels, H, W)
        if obs_or_states_sequence.dim() != 5:
            raise ValueError(
                f"obs_or_states_sequence must be 5D (batch, seq_len, C, H, W), "
                f"got {obs_or_states_sequence.shape}"
            )
        if action_sequence.dim() < 2:
            raise ValueError(
                f"action_sequence must be at least 2D (batch, seq_len, ...), "
                f"got {action_sequence.shape}"
            )
        if seq_len != self.sequence_length:
            raise ValueError(
                f"Sequence length mismatch: expected {self.sequence_length}, "
                f"got obs_seq={seq_len}, action_seq={action_sequence.shape[1]}"
            )
        
        # Auto-detect mode based on channel dimension
        input_channels = obs_or_states_sequence.shape[2]
        is_training_mode = (input_channels == self.observation_shape[0])
        
        if is_training_mode:
            # TRAINING MODE: Encode raw observations on-the-fly
            # Input: (batch, seq_len, C, H, W) - raw observations
            encoded_states = self._encode_observation_sequence(obs_or_states_sequence)
        else:
            # PLANNING MODE: Use already-encoded latent states
            # Input: (batch, seq_len, num_channels, H', W') - encoded states
            encoded_states = obs_or_states_sequence
        
        # ----------------------------------------------------------------------------
        # NOTE: Want to add feature where we have pre-trained representation network 
        # which is frozen during world model training
        # ----------------------------------------------------------------------------

        # Flatten state dimensions
        enc_channels, enc_h, enc_w = encoded_states.shape[2:]
        encoded_states_flat = encoded_states.reshape(batch_size, seq_len, -1)
        
        # Prepare and embed action sequence
        action_embedded = self._embed_action_sequence(action_sequence)
        
        # Concatenate encoded states with actions
        # (batch, seq_len, state_dim + action_dim)
        state_action_pairs = torch.cat([encoded_states_flat, action_embedded], dim=-1)
        
        # Input: (batch, seq_len, state_dim + action_dim)
        # Output: (batch, seq_len, state_dim)
        state_pred_seq = self.mini_stu(state_action_pairs)
        
        # Extract next state prediction (take the last in sequence)
        # (batch, state_dim) where state_dim = num_channels * H * W
        state_pred_flat = state_pred_seq[:, -1, :]
        
        # Reshape to spatial dimensions
        # (batch, num_channels, H', W')
        state_pred = state_pred_flat.view(batch_size, enc_channels, enc_h, enc_w)
        
        # OPTIONAL: refinement with residual blocks
        if self.num_blocks > 0:
            x = state_pred
            x = self.refine_conv(x)
            x = self.refine_bn(x)
            x = nn.functional.relu(x)
            
            for block in self.refine_blocks:
                x = block(x)
            
            # Residual connection
            next_state = x + state_pred
            next_state = nn.functional.relu(next_state)
        else:
            next_state = state_pred
        
        return next_state
    
    def _encode_observation_sequence(self, obs_sequence):
        """
        Encode observation sequence on-the-fly through RepresentationNetwork.
        
        Args:
            obs_sequence: (batch, seq_len, C, H, W) - raw observations
        
        Returns:
            encoded_states: (batch, seq_len, num_channels, H', W') - encoded states with gradient flow
        """
        batch_size = obs_sequence.shape[0]
        seq_len = obs_sequence.shape[1]
        
        # Reshape to (batch * seq_len, C, H, W) for batch encoding
        obs_flat = obs_sequence.reshape(batch_size * seq_len, *self.observation_shape)
        
        # Encode through RepresentationNetwork (gradients enabled)
        with torch.enable_grad():
            encoded_states = self.representation_net(obs_flat)  # (batch*seq_len, num_channels, H', W')
        
        # Get encoded state dimensions
        enc_channels, enc_h, enc_w = encoded_states.shape[1:]
        
        # Reshape back to (batch, seq_len, num_channels, H', W')
        encoded_states = encoded_states.reshape(batch_size, seq_len, enc_channels, enc_h, enc_w)
        
        return encoded_states
    
    def _embed_action_sequence(self, action_sequence):
        # Ensure action_sequence has shape (batch, seq_len, action_dim or 1)
        if action_sequence.dim() == 2:
            # (batch, seq_len) -> (batch, seq_len, 1)
            action_sequence = action_sequence.unsqueeze(-1)
        
        action_sequence = action_sequence.float()
        
        if self.action_embedding:
            if not self.is_continuous:
                action_seq_flat = action_sequence.squeeze(-1).long()  # (batch, seq_len)
                action_embedded = self.action_embed(action_seq_flat)  # (batch, seq_len, embed_dim)
            else:
                action_embedded = self.action_embed(action_sequence)  # (batch, seq_len, embed_dim)
        else:
            action_embedded = action_sequence.float()
        
        return action_embedded

class DynamicsNetworkWrapper(nn.Module):
    def __init__(self, use_mini_stu, original_dynamics, mini_stu_dynamics=None, world_model=None):
        super().__init__()
        self.use_mini_stu = use_mini_stu
        self.original_dynamics = original_dynamics
        self.mini_stu_dynamics = mini_stu_dynamics  # Original action-only version (deprecated)
        self.world_model = world_model  # New World Model version
        
        if use_mini_stu and world_model is None:
            raise ValueError("world_model must be provided when use_mini_stu=True")
    
    def forward(self, *args, **kwargs):
        if self.use_mini_stu and 'use_world_model' in kwargs and kwargs['use_world_model']:
            obs_sequence = args[0]
            action_sequence = args[1]
            current_state = args[2] if len(args) > 2 else kwargs.get('current_state', None)
            return self.world_model(obs_sequence, action_sequence, current_state)
        elif self.use_mini_stu and len(args) >= 2 and args[0].dim() == 5:
            obs_sequence = args[0]
            action_sequence = args[1]
            current_state = args[2] if len(args) > 2 else None
            return self.world_model(obs_sequence, action_sequence, current_state)
        else:
            return self.original_dynamics(*args, **kwargs)
