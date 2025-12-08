import gym
import numpy as np

from ez.utils.format import arr_to_str


# Compatibility helper for gym 0.22.0+ which changed reset/step signatures
def _normalize_reset_result(result):
    """Convert reset result to always be just obs (gym < 0.22.0 format)"""
    if isinstance(result, tuple) and len(result) == 2:
        return result[0]  # gym 0.22.0+ returns (obs, info)
    return result  # older gym returns just obs


def _normalize_step_result(result):
    """Convert step result to 4-tuple (gym < 0.22.0 format) from 5-tuple or 4-tuple"""
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
        return obs, reward, done, info
    return result  # already 4-tuple


class BaseWrapper(gym.Wrapper):
    def __init__(self, env, obs_to_string, clip_reward):
        """Cosine Consistency loss function: similarity loss
        Parameters
        ----------
        obs_to_string: bool. Convert the observation to jpeg string if True, in order to save memory usage.
        """
        super().__init__(env)
        self.obs_to_string = obs_to_string
        self.clip_reward = clip_reward

    def format_obs(self, obs):
        if self.obs_to_string:
            # convert obs to jpeg string for lower memory usage
            obs = obs.astype(np.uint8)
            obs = arr_to_str(obs)
        return obs

    def step(self, action):
        step_result = self.env.step(action)
        step_result = _normalize_step_result(step_result)
        obs, reward, done, info = step_result
        # format observation
        obs = self.format_obs(obs)

        info['raw_reward'] = reward
        if self.clip_reward:
            reward = np.sign(reward)

        return obs, reward, done, info

    def reset(self, **kwargs):
        reset_result = self.env.reset(**kwargs)
        # Extract obs, handling both gym API versions
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs = reset_result[0]
        else:
            obs = reset_result
        # format observation
        obs = self.format_obs(obs)

        return obs

    def close(self):
        return self.env.close()
