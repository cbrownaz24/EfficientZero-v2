import cv2
import gym
import numpy as np


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


class CompatWrapper:
    """Base wrapper class compatible with both gym <0.22.0 and >=0.22.0"""
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reward_range = getattr(env, 'reward_range', (-float('inf'), float('inf')))
        self.metadata = getattr(env, 'metadata', {})
    
    @property
    def unwrapped(self):
        return self.env.unwrapped
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment"""
        return getattr(self.env, name)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self, mode='human', **kwargs):
        return self.env.render(mode=mode, **kwargs)
    
    def close(self):
        return self.env.close()
    
    def seed(self, seed=None):
        return self.env.seed(seed)


class TimeLimit(CompatWrapper):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        step_result = self.env.step(ac)
        step_result = _normalize_step_result(step_result)
        observation, reward, done, info = step_result
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        try:
            # Try gym 0.22.0+ style first (returns (obs, info))
            reset_result = self.env.reset(**kwargs)
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs, info = reset_result
                return obs
            else:
                # Old gym style (returns just obs)
                return reset_result
        except (TypeError, ValueError):
            # Fallback
            return self.env.reset(**kwargs)


class NoopResetEnv(CompatWrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            try:
                noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
            except:
                noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            step_result = self.env.step(self.noop_action)
            step_result = _normalize_step_result(step_result)
            obs, done = step_result[0], step_result[2]
            if done:
                reset_result = self.env.reset(**kwargs)
                # Extract obs, handling both gym API versions
                if isinstance(reset_result, tuple) and len(reset_result) == 2:
                    obs = reset_result[0]
                else:
                    obs = reset_result
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(CompatWrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super().__init__(env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        step_result = self.env.step(action)
        step_result = _normalize_step_result(step_result)
        obs, reward, done, info = step_result
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            reset_result = self.env.reset(**kwargs)
            # Extract obs, handling both gym API versions
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs = reset_result[0]
            else:
                obs = reset_result
        else:
            # no-op step to advance from terminal/lost life state
            step_result = self.env.step(0)
            step_result = _normalize_step_result(step_result)
            obs = step_result[0]
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(CompatWrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip
        self.max_frame = np.zeros(env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        info = {}
        for i in range(self._skip):
            step_result = self.env.step(action)
            step_result = _normalize_step_result(step_result)
            obs, reward, done, info = step_result
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        self.max_frame = self._obs_buffer.max(axis=0)

        return self.max_frame, total_reward, done, info

    def reset(self, **kwargs):
        reset_result = self.env.reset(**kwargs)
        # Extract obs, handling both gym API versions
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            return reset_result[0]
        else:
            return reset_result

    def render(self, mode='human', **kwargs):
        img = self.max_frame
        img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA).astype(np.uint8)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen


class WarpFrame(CompatWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs

    def step(self, action):
        step_result = self.env.step(action)
        step_result = _normalize_step_result(step_result)
        obs, reward, done, info = step_result
        obs = self.observation(obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        reset_result = self.env.reset(**kwargs)
        # Extract obs, handling both gym API versions
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs = reset_result[0]
        else:
            obs = reset_result
        obs = self.observation(obs)
        return obs

class DMC_Obs_Wrapper(CompatWrapper):

    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        obs = np.moveaxis(obs, 0, -1)
        return obs

    def step(self, action):
        step_result = self.env.step(action)
        step_result = _normalize_step_result(step_result)
        obs, reward, done, info = step_result
        obs = self.observation(obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        reset_result = self.env.reset(**kwargs)
        # Extract obs, handling both gym API versions
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs = reset_result[0]
        else:
            obs = reset_result
        obs = self.observation(obs)
        return obs
