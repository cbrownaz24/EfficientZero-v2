"""
Observation sequence buffer for World Model training.

This module provides utilities to manage and retrieve observation sequences
from batch data for use with the MiniSTU World Model.
"""

import torch
import numpy as np


class ObservationSequenceExtractor:
    """
    Extracts observation sequences from batch data for World Model training.
    
    The World Model requires:
    - Observation sequences: (batch, seq_len, C, H, W)
    - Action sequences: (batch, seq_len, action_dim)
    
    This class handles extracting these from the standard EZ batch format
    where observations are concatenated as: [s_t, s_t+1, s_t+2, ..., s_t+unroll]
    """
    
    @staticmethod
    def extract_obs_action_sequences(obs_batch_raw, action_batch, n_stack, image_channel, 
                                     sequence_length, unroll_steps):
        """
        Extract observation and action sequences for World Model training.
        
        Args:
            obs_batch_raw: Raw observation batch (batch, total_frames*C, H, W)
                          Contains stacked observations from t-n_stack to t+unroll
            action_batch: Action batch (batch, unroll_steps+1, action_dim or 1)
            n_stack: Number of frames stacked for initial observation
            image_channel: Number of channels per frame (C)
            sequence_length: Desired length of sequences for MiniSTU
            unroll_steps: Number of unroll steps in training
        
        Returns:
            obs_sequences: List of observation sequences (batch, seq_len, C, H, W)
            action_sequences: List of action sequences (batch, seq_len, action_dim)
            
            Note: Returns multiple sequences if multiple can be extracted from unroll
        """
        batch_size = obs_batch_raw.shape[0]
        height, width = obs_batch_raw.shape[2:]
        
        # Determine how many observation frames are available
        # obs_batch_raw has: n_stack initial frames + unroll_steps future frames
        total_frames = obs_batch_raw.shape[1] // image_channel
        
        # Extract sequences of length sequence_length
        obs_sequences = []
        action_sequences = []
        
        # We can extract multiple overlapping sequences from the unroll
        max_start_idx = max(0, total_frames - sequence_length + 1)
        
        for start_idx in range(max_start_idx):
            # Extract observation sequence
            obs_seq_frames = []
            for frame_idx in range(start_idx, start_idx + sequence_length):
                frame_start = frame_idx * image_channel
                frame_end = frame_start + image_channel
                # (batch, C, H, W)
                frame = obs_batch_raw[:, frame_start:frame_end, :, :]
                obs_seq_frames.append(frame)
            
            # Stack into (batch, seq_len, C, H, W)
            obs_seq = torch.stack(obs_seq_frames, dim=1)
            obs_sequences.append(obs_seq)
            
            # Extract corresponding action sequence
            # Actions are indexed relative to the observation frames
            # Frame i corresponds to action taken to reach frame i+1
            action_seq_indices = []
            for frame_idx in range(start_idx, start_idx + sequence_length):
                # Map frame index to action index
                # Frame 0-n_stack correspond to initial observation, action 0 is first step
                action_idx = max(0, frame_idx - n_stack)
                if action_idx < action_batch.shape[1]:
                    action_seq_indices.append(action_idx)
            
            # If we couldn't get enough actions, pad with last action
            while len(action_seq_indices) < sequence_length:
                action_seq_indices.append(min(action_batch.shape[1] - 1, 
                                             action_batch.shape[1] - 1))
            
            # Extract actions
            action_seq = action_batch[:, action_seq_indices, :]  # (batch, seq_len, action_dim or 1)
            action_sequences.append(action_seq)
        
        return obs_sequences, action_sequences
    
    @staticmethod
    def extract_single_sequence(obs_batch_raw, action_batch, n_stack, image_channel,
                               sequence_length, unroll_steps):
        """
        Extract a single observation-action sequence pair for World Model training.
        
        This is a simpler interface that extracts one sequence starting from the
        initial observation frame.
        
        Args:
            obs_batch_raw: Raw observation batch (batch, total_frames*C, H, W)
            action_batch: Action batch (batch, unroll_steps+1, action_dim or 1)
            n_stack: Number of frames stacked for initial observation
            image_channel: Number of channels per frame (C)
            sequence_length: Desired length of sequences for MiniSTU
            unroll_steps: Number of unroll steps in training
        
        Returns:
            obs_seq: Observation sequence (batch, seq_len, C, H, W)
            action_seq: Action sequence (batch, seq_len, action_dim or 1)
        """
        batch_size = obs_batch_raw.shape[0]
        
        # Extract observation frames starting from frame n_stack (current state)
        obs_seq_frames = []
        for i in range(sequence_length):
            frame_idx = n_stack + i
            frame_start = frame_idx * image_channel
            frame_end = frame_start + image_channel
            
            if frame_start < obs_batch_raw.shape[1]:
                # (batch, C, H, W)
                frame = obs_batch_raw[:, frame_start:frame_end, :, :]
                obs_seq_frames.append(frame)
            else:
                # Pad with last available frame if needed
                frame = obs_batch_raw[:, -image_channel:, :, :]
                obs_seq_frames.append(frame)
        
        # Stack into (batch, seq_len, C, H, W)
        obs_seq = torch.stack(obs_seq_frames, dim=1)
        
        # Extract action sequence
        # Actions correspond to transitions between frames
        action_seq_indices = []
        for i in range(sequence_length):
            action_idx = min(i, action_batch.shape[1] - 1)
            action_seq_indices.append(action_idx)
        
        action_seq = action_batch[:, action_seq_indices, :]  # (batch, seq_len, action_dim or 1)
        
        return obs_seq, action_seq


def extract_world_model_batch(batch_data, config):
    """
    Convert standard EZ batch format to World Model format.
    
    Args:
        batch_data: Standard EZ batch tuple containing:
            (obs_batch_raw, action_batch, mask_batch, weights_lst, ...)
        config: Configuration object with model.mini_stu settings
    
    Returns:
        world_model_batch: Tuple of (obs_sequences, action_sequences)
    """
    obs_batch_raw = batch_data[0]
    action_batch = batch_data[1]
    
    n_stack = config.env.n_stack
    image_channel = config.env.obs_shape[0]  # Number of channels
    sequence_length = config.model.mini_stu.sequence_length
    unroll_steps = config.rl.unroll_steps
    
    # Extract sequences
    obs_seq, action_seq = ObservationSequenceExtractor.extract_single_sequence(
        obs_batch_raw, action_batch, n_stack, image_channel, 
        sequence_length, unroll_steps
    )
    
    return obs_seq, action_seq
