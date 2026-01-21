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


class MiniSTUWorldModel(nn.Module):
    """
    World Model using MiniSTU to predict state sequences from observation and action sequences.
    
    Supports two modes:
    
    **TRAINING MODE** (optimize representation network):
    - Input: raw observation sequences (batch, seq_len, C, H, W) + action sequences (batch, seq_len, action_dim)
    - Process: Encode observations on-the-fly through RepresentationNetwork, concatenate with actions, MiniSTU processes
    - Output: predicted next states (batch, num_channels, H', W')
    - Gradients flow through RepresentationNetwork for end-to-end learning
    
    **PLANNING MODE** (MCTS search with frozen representation):
    - Input: already-encoded latent states (batch, seq_len, num_channels, H', W') + action sequences
    - Process: Flatten states, concatenate with actions, MiniSTU processes  
    - Output: predicted next states (batch, num_channels, H', W')
    - Representation network frozen (no gradients needed)
    
    Key insight: During training, we always recompute s_1:T from ground truth observations
    to ensure the representation network optimization is based on accurate state information.
    """
    
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
        
        # Store reference to representation network (will be shared with full model)
        self.representation_net = representation_net
        
        # Action embedding layer if needed
        if action_embedding:
            if is_continuous:
                # For continuous actions, embed the raw action values
                self.action_embed = nn.Linear(action_space_size, action_embedding_dim)
            else:
                # For discrete actions, use embedding table
                self.action_embed = nn.Embedding(action_space_size, action_embedding_dim)
            action_input_dim = action_embedding_dim
        else:
            # Use raw action as input to MiniSTU
            action_input_dim = action_space_size if is_continuous else 1
        
        # State dimension for MiniSTU: concatenation of flattened encoded state + action
        # This must match the actual flattened state dimensions passed to MiniSTU
        state_dim = num_channels * state_shape[1] * state_shape[2]
        
        # CRITICAL: In MiniSTU library, num_filters must equal seq_len
        # This is a constraint of the spectral temporal unit implementation
        effective_num_filters = sequence_length
        
        # Create MiniSTU instance for sequence prediction
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
        
        # Ensure sequence lengths match
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
        
        # Get encoded state dimensions
        enc_channels, enc_h, enc_w = encoded_states.shape[2:]
        
        # Flatten spatial dimensions for MiniSTU processing
        # (batch, seq_len, num_channels * H * W)
        encoded_states_flat = encoded_states.reshape(batch_size, seq_len, -1)
        
        # Prepare and embed action sequence
        action_embedded = self._embed_action_sequence(action_sequence)
        
        # Concatenate encoded states with actions
        # (batch, seq_len, state_dim + action_dim)
        state_action_pairs = torch.cat([encoded_states_flat, action_embedded], dim=-1)
        
        # Process through MiniSTU
        # Input: (batch, seq_len, state_dim + action_dim)
        # Output: (batch, seq_len, state_dim)
        state_pred_seq = self.mini_stu(state_action_pairs)
        
        # Extract next state prediction (take the last in sequence)
        # (batch, state_dim) where state_dim = num_channels * H * W
        state_pred_flat = state_pred_seq[:, -1, :]
        
        # Reshape to spatial dimensions
        # (batch, num_channels, H', W')
        state_pred = state_pred_flat.view(batch_size, enc_channels, enc_h, enc_w)
        
        # Optional refinement with residual blocks
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
        """
        Prepare and embed action sequence.
        
        Args:
            action_sequence: (batch, seq_len, action_dim or 1) - actions
        
        Returns:
            action_embedded: (batch, seq_len, action_embedding_dim or action_dim)
        """
        # Ensure action_sequence has shape (batch, seq_len, action_dim or 1)
        if action_sequence.dim() == 2:
            # (batch, seq_len) -> (batch, seq_len, 1)
            action_sequence = action_sequence.unsqueeze(-1)
        
        action_sequence = action_sequence.float()
        
        # Embed actions if needed
        if self.action_embedding:
            if not self.is_continuous:
                # Discrete actions: squeeze and embed
                action_seq_flat = action_sequence.squeeze(-1).long()  # (batch, seq_len)
                action_embedded = self.action_embed(action_seq_flat)  # (batch, seq_len, embed_dim)
            else:
                # Continuous actions: directly embed
                action_embedded = self.action_embed(action_sequence)  # (batch, seq_len, embed_dim)
        else:
            # Use raw actions
            action_embedded = action_sequence.float()
        
        return action_embedded


class MiniSTUDynamicsNetwork(nn.Module):
    """
    DEPRECATED: Original broken MiniSTU implementation that only processes actions.
    Kept for backward compatibility. Use MiniSTUWorldModel instead.
    """
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
    """
    Wrapper that switches between classical DynamicsNetwork and World Model.
    
    Supports two interfaces:
    1. Classical: forward(state, action) - single step
    2. World Model: forward(obs_sequence, action_sequence, current_state=None, use_world_model=False)
    
    The wrapper automatically detects which interface is being used based on:
    - Input dimensionality (5D = world model, 4D = classical)
    - Explicit use_world_model flag
    """
    def __init__(self, use_mini_stu, original_dynamics, mini_stu_dynamics=None, world_model=None):
        super().__init__()
        self.use_mini_stu = use_mini_stu
        self.original_dynamics = original_dynamics
        self.mini_stu_dynamics = mini_stu_dynamics  # Original action-only version (deprecated)
        self.world_model = world_model  # New World Model version
        
        if use_mini_stu and world_model is None:
            raise ValueError("world_model must be provided when use_mini_stu=True")
    
    def forward(self, *args, **kwargs):
        """
        Flexible forward that supports both classical and world model interfaces.
        
        Classical (single step):
            forward(state, action)
        
        World Model (sequence):
            forward(obs_sequence, action_sequence, current_state=None, use_world_model=True)
        """
        if self.use_mini_stu and 'use_world_model' in kwargs and kwargs['use_world_model']:
            # World Model interface explicitly requested
            obs_sequence = args[0]
            action_sequence = args[1]
            current_state = args[2] if len(args) > 2 else kwargs.get('current_state', None)
            return self.world_model(obs_sequence, action_sequence, current_state)
        elif self.use_mini_stu and len(args) >= 2 and args[0].dim() == 5:
            # Auto-detect: 5D input tensor = world model interface
            obs_sequence = args[0]
            action_sequence = args[1]
            current_state = args[2] if len(args) > 2 else None
            return self.world_model(obs_sequence, action_sequence, current_state)
        else:
            # Classical interface (single step dynamics)
            # Always use original_dynamics for single-step, regardless of use_mini_stu
            # (World Model is for sequence processing only)
            return self.original_dynamics(*args, **kwargs)
