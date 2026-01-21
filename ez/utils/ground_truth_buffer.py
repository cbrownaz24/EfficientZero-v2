"""
Ground truth observation buffer implementation for World Model training.

Provides two complementary buffers:
1. GroundTruthObservationBuffer: Stores raw observations, encodes on-demand
2. ImaginaryRolloutBuffer: Stores predicted states during MCTS planning
"""

import torch
import torch.nn as nn


class GroundTruthObservationBuffer:
    """
    Maintains a sliding window of ground truth observations from the environment.
    
    Purpose:
    - During TRAINING: Recompute ground truth state encodings for consistency supervision
    - During PLANNING: Not used directly (imaginary buffer takes over)
    
    Key insight: We store RAW observations, not encoded states.
    This allows us to recompute encodings after the representation network is updated.
    """
    
    def __init__(self, sequence_length=5, obs_shape=(4, 84, 84), device='cuda'):
        """
        Initialize ground truth observation buffer.
        
        Args:
            sequence_length: T - number of observations to maintain
            obs_shape: Shape of each observation (C, H, W)
            device: Device for tensors ('cuda' or 'cpu')
        """
        self.sequence_length = sequence_length
        self.obs_shape = obs_shape
        self.device = device
        
        # Will store raw observations from environment
        self.obs_buffer = None
        self.is_filled = False
        self.current_size = 0
    
    def push_observation(self, obs):
        """
        Add a new observation to the buffer.
        
        Args:
            obs: Single observation tensor, shape obs_shape = (4, 84, 84)
                 or channels-last format [H, W, C] which will be transposed to [C, H, W]
        
        Effect:
            - If buffer not full: append to buffer
            - If buffer full: slide window left, add new at end
        """
        # Move to device if needed
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs)
        obs = obs.to(self.device)
        
        # Handle channels-last format: transpose [H, W, C] -> [C, H, W] if needed
        if obs.dim() == 3:
            # Check if observation is in channels-last format
            # Expected shape from config: obs_shape = [C, H, W]
            expected_shape = torch.Size(self.obs_shape)
            current_shape = obs.shape
            
            # If shapes don't match and it looks like [H, W, C] format, transpose
            if current_shape != expected_shape:
                # Assume channels-last if last dimension is much smaller (1, 3, or 4)
                # and matches expected channels
                if (current_shape[-1] == expected_shape[0] and 
                    current_shape[0] == expected_shape[1] and 
                    current_shape[1] == expected_shape[2]):
                    # Transpose from [H, W, C] to [C, H, W]
                    obs = obs.permute(2, 0, 1)
        
        if self.obs_buffer is None:
            # Initialize buffer on first observation
            self.obs_buffer = torch.zeros(
                self.sequence_length, *self.obs_shape,
                dtype=torch.float32,
                device=self.device
            )
        
        if self.current_size < self.sequence_length:
            # Buffer not yet full - just append
            self.obs_buffer[self.current_size] = obs
            self.current_size += 1
            
            if self.current_size == self.sequence_length:
                self.is_filled = True
        else:
            # Buffer is full - slide window forward
            self.obs_buffer[:-1] = self.obs_buffer[1:].clone()
            self.obs_buffer[-1] = obs
    
    def reset(self):
        """Reset buffer to empty state."""
        self.obs_buffer = None
        self.is_filled = False
        self.current_size = 0
    
    def get_raw_observations(self):
        """
        Get the current raw observation window.
        
        Returns:
            obs_window: (sequence_length, *obs_shape) = (5, 4, 84, 84)
        
        Raises:
            RuntimeError if buffer not yet filled
        """
        if not self.is_filled:
            raise RuntimeError(
                f"Buffer not yet filled! current_size={self.current_size}, "
                f"needed={self.sequence_length}"
            )
        return self.obs_buffer.clone()
    
    def get_encoded_states(self, representation_net, normalize=True):
        """
        Encode current observation window through representation network.
        
        This is the KEY method: It reencodes observations using the CURRENT
        representation network. After training updates the rep-net, calling
        this method will use the NEW parameters!
        
        Args:
            representation_net: RepresentationNetwork instance
            normalize: Whether to normalize observations (divide by 255)
        
        Returns:
            states: (sequence_length, num_channels, H', W') = (5, 128, 6, 6)
        
        Gradients: Enabled by default (so consistency loss can backprop)
        """
        if not self.is_filled:
            raise RuntimeError("Buffer not yet filled!")
        
        # Get raw observations
        obs_window = self.obs_buffer.clone()  # (5, 4, 84, 84)
        
        # Normalize if needed
        if normalize:
            obs_window = obs_window / 255.0
        
        # Encode all observations
        # Shape: (5, 4, 84, 84) → (5, 128, 6, 6)
        with torch.enable_grad():  # Ensure gradients are tracked
            states = representation_net(obs_window)
        
        return states
    
    def get_single_encoded_state(self, representation_net, index=-1, normalize=True):
        """
        Encode a single observation from the buffer.
        
        Args:
            representation_net: RepresentationNetwork instance
            index: Which observation to encode (-1 = latest)
            normalize: Whether to normalize
        
        Returns:
            state: (num_channels, H', W') = (128, 6, 6)
        """
        if not self.is_filled:
            raise RuntimeError("Buffer not yet filled!")
        
        obs = self.obs_buffer[index].clone()
        
        if normalize:
            obs = obs / 255.0
        
        with torch.enable_grad():
            state = representation_net(obs.unsqueeze(0))  # Add batch dim
            state = state.squeeze(0)  # Remove batch dim
        
        return state


class ImaginaryRolloutBuffer:
    """
    Temporary buffer for storing predicted states during MCTS planning.
    
    Different from GroundTruthObservationBuffer:
    - Stores ENCODED STATES, not raw observations
    - States are PREDICTIONS from world model
    - Representation network is FROZEN (no gradients)
    
    Usage:
    1. Initialize with ground truth states at start of planning
    2. Push predicted states during MCTS simulations
    3. Reset for each new simulation
    """
    
    def __init__(self, sequence_length=5, state_shape=(128, 6, 6), device='cuda'):
        """
        Initialize imaginary rollout buffer.
        
        Args:
            sequence_length: T - number of states to maintain
            state_shape: Shape of each state (num_channels, H', W')
            device: Device for tensors
        """
        self.sequence_length = sequence_length
        self.state_shape = state_shape
        self.device = device
        
        # Stores encoded STATES (latent predictions)
        self.state_buffer = None
        self.is_filled = False
        self.current_size = 0
    
    def initialize_from_ground_truth(self, gt_states):
        """
        Initialize imaginary buffer with ground truth encoded states.
        
        Called at: Start of MCTS planning
        
        Args:
            gt_states: Ground truth states from representation network
                      Shape: (sequence_length, *state_shape) = (5, 128, 6, 6)
        
        Effect:
            Imaginary rollout will start from these known-good states
        """
        if gt_states.shape[0] != self.sequence_length:
            raise ValueError(
                f"Ground truth states must have sequence_length={self.sequence_length}, "
                f"got {gt_states.shape[0]}"
            )
        
        self.state_buffer = gt_states.clone().detach()
        self.current_size = self.sequence_length
        self.is_filled = True
    
    def push_predicted_state(self, state):
        """
        Add predicted state from world model during imagined rollout.
        
        Called during: MCTS simulations in planning loop
        
        Args:
            state: Predicted state tensor, shape state_shape = (128, 6, 6)
        
        Effect: Slides window forward, adds new prediction
        """
        if self.state_buffer is None:
            # Initialize on first push
            self.state_buffer = torch.zeros(
                self.sequence_length, *self.state_shape,
                dtype=torch.float32,
                device=self.device
            )
        
        # Detach to prevent gradient accumulation during planning
        state = state.clone().detach()
        
        if self.current_size < self.sequence_length:
            self.state_buffer[self.current_size] = state
            self.current_size += 1
            
            if self.current_size == self.sequence_length:
                self.is_filled = True
        else:
            # Slide window forward
            self.state_buffer[:-1] = self.state_buffer[1:].clone()
            self.state_buffer[-1] = state
    
    def get_state_sequence(self):
        """
        Get current imagined state sequence for next world model step.
        
        Returns:
            states: (sequence_length, *state_shape) = (5, 128, 6, 6)
        
        Raises:
            RuntimeError if buffer not yet filled
        """
        if not self.is_filled:
            raise RuntimeError(
                f"Buffer not yet filled! current_size={self.current_size}, "
                f"needed={self.sequence_length}"
            )
        return self.state_buffer.clone().detach()
    
    def get_latest_state(self):
        """
        Get the most recent state from imaginary rollout.
        
        Used for: Getting current imagined state for value estimation
        
        Returns:
            state: (*state_shape) = (128, 6, 6)
        """
        if not self.is_filled or self.current_size == 0:
            raise RuntimeError("Buffer is empty!")
        return self.state_buffer[-1].clone().detach()
    
    def reset_for_new_simulation(self, gt_states):
        """
        Reset buffer for next MCTS simulation.
        
        Called at: Start of each MCTS simulation
        
        Args:
            gt_states: Latest ground truth encoded states
        
        Effect:
            Imagined rollout starts fresh from current real state
        """
        self.initialize_from_ground_truth(gt_states)


class DualBufferManager:
    """
    Convenience wrapper managing both buffers together.
    
    Ensures consistency and provides high-level interface.
    """
    
    def __init__(self, sequence_length=5, obs_shape=(4, 84, 84), 
                 state_shape=(128, 6, 6), device='cuda'):
        """Initialize both buffers."""
        self.gt_buffer = GroundTruthObservationBuffer(
            sequence_length=sequence_length,
            obs_shape=obs_shape,
            device=device
        )
        
        self.imaginary_buffer = ImaginaryRolloutBuffer(
            sequence_length=sequence_length,
            state_shape=state_shape,
            device=device
        )
        
        self.device = device
        self.sequence_length = sequence_length
    
    def observe_frame(self, obs):
        """Update ground truth buffer with new observation."""
        self.gt_buffer.push_observation(obs)
    
    def get_gt_encoded_states(self, representation_net):
        """Get ground truth encoded states from observations."""
        return self.gt_buffer.get_encoded_states(representation_net)
    
    def start_planning(self, gt_states):
        """Initialize imaginary buffer for MCTS."""
        self.imaginary_buffer.initialize_from_ground_truth(gt_states)
    
    def imagine_step(self, predicted_state):
        """Add predicted state during MCTS simulation."""
        self.imaginary_buffer.push_predicted_state(predicted_state)
    
    def get_imagined_sequence(self):
        """Get current imagined state sequence."""
        return self.imaginary_buffer.get_state_sequence()
    
    def reset_imagination(self, gt_states):
        """Reset imaginary buffer for next MCTS simulation."""
        self.imaginary_buffer.reset_for_new_simulation(gt_states)
    
    def reset(self):
        """Reset all buffers."""
        self.gt_buffer.reset()
        # Imaginary buffer will be reset on next start_planning
