import os

import random
import numpy as np
import matplotlib.pyplot as plt
import collections
from gym.core import Env
import time
import cv2


class MODE:
    IMG = 'img'
    IMG_PROP = 'img_prop'
    PROP = 'prop'

def make_dir(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError:
        pass
    return dir_path

def set_seed_everywhere(seed, env=None):
    np.random.seed(seed)
    random.seed(seed)

    if env is not None:
        env.seed(seed)

def smoothed_curve(returns, ep_lens, x_tick=5000, window_len=5000):
    """
    Args:
        returns: 1-D numpy array with episodic returs
        ep_lens: 1-D numpy array with episodic returs
        x_tick (int): Bin size
        window_len (int): Length of averaging window
    Returns:
        A numpy array
    """
    rets = []
    x = []
    cum_episode_lengths = np.cumsum(ep_lens)

    if cum_episode_lengths[-1] >= x_tick:
        y = cum_episode_lengths[-1] + 1
        steps_show = np.arange(x_tick, y, x_tick)

        for i in range(len(steps_show)):
            rets_in_window = returns[(cum_episode_lengths > max(0, x_tick * (i + 1) - window_len)) *
                                     (cum_episode_lengths < x_tick * (i + 1))]
            if rets_in_window.any():
                rets.append(np.mean(rets_in_window))
                x.append((i+1) * x_tick)

    return np.array(rets), np.array(x)

def show_learning_curve(fname, rets, ep_lens, xtick, xlimit=None, ylimit=None, save_fig=True):
        plot_rets, plot_x = smoothed_curve(
                np.array(rets), np.array(ep_lens), x_tick=xtick, window_len=xtick)
        
        if len(plot_rets):
            plt.clf()
            if xlimit is not None:
                plt.xlim(xlimit)
        
            if ylimit is not None:
                plt.ylim(ylimit)
                
            plt.plot(plot_x, plot_rets)
            plt.pause(0.001)
            if save_fig:
                plt.savefig(fname)

## SRC: https://github.com/kindredresearch/SenseAct/blob/master/senseact/utils.py

class EnvSpec():
    def __init__(self, env_spec, observation_space, action_space):
        self._observation_space = observation_space
        self._action_space = action_space
        self._unwrapped_spec = env_spec

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

Step = collections.namedtuple("Step", ["observation", "reward", "done", "info"])


class WrappedEnv(Env):
    def __init__(
            self,
            env,
            episode_max_steps=-1,
            is_min_time=False,
            reward_scale=1.0,
            reward_penalty=0,
            steps_penalty=0,
            start_step=0,
            start_episode=0):
        
        if is_min_time:
            assert episode_max_steps > 0
        
        self._wrapped_env = env
        self._episode_max_steps = episode_max_steps
        self._is_min_time = is_min_time
        self._reward_scale = reward_scale
        self._reward_penalty = reward_penalty
        self._steps_penalty = steps_penalty
        self._spec = EnvSpec(env.spec, self.observation_space, self.action_space)

        self._total_steps  = start_step
        self._episode = start_episode

    def _reset_stats(self):
        self._reward_sum = 0
        self._episode_steps = 0
        if self._is_min_time:
            self._sub_episode = 0
            self._sub_episode_steps=0
        self._start_time = time.time()

    def _monitor(self, reward, done, info):
        self._reward_sum += reward
        self._episode_steps += 1
        self._total_steps += 1 

        new_info = {}

        if 'battery_charge' in info:
            new_info['battery_charge'] = info['battery_charge']

        if not self._is_min_time and self._episode_max_steps > 0 \
            and self._episode_steps == self._episode_max_steps:
            new_info['TimeLimit.truncated'] = True

        if 'TimeLimit.truncated' in info:
            new_info['TimeLimit.truncated'] = True

        if done or (not self._is_min_time and 'TimeLimit.truncated' in new_info):
            new_info['episode'] = self._episode
            new_info['step'] = self._total_steps
            new_info['episode_steps'] = self._episode_steps
            new_info['duration'] = time.time() - self._start_time
            new_info['return'] = self._reward_sum
            
            if done and 'target_size' in info:
                new_info['target_size'] = info['target_size']

            self._episode += 1
            return done, new_info
        
        if self._is_min_time:
            self._sub_episode_steps += 1
            if self._sub_episode_steps == self._episode_max_steps:
                self._reward_sum += self._reward_penalty
                self._total_steps += self._steps_penalty
                self._episode_steps += self._steps_penalty
                
                new_info['TimeLimit.truncated'] = True
                new_info['episode'] = self._episode
                new_info['sub_episode'] = self._sub_episode
                new_info['sub_episode_steps'] = self._sub_episode_steps 
                self._sub_episode_steps = 0
                self._sub_episode += 1
                return done, new_info
        
        return done, new_info

    def reset(self, reset_stats=True):
        ret = self._wrapped_env.reset()
        if reset_stats:
            self._reset_stats()
        return ret

    def step(self, action):
        # rescale the action
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        done, info = self._monitor(reward, done, info)

        return Step(next_obs, reward * self._reward_scale, done, info)

    def __str__(self):
        return "RealTimeEnv: %s" % self._wrapped_env


    @property
    def total_steps(self):
        return self._total_steps

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def start(self):
        return self._wrapped_env.start()

    def close(self):
        super(WrappedEnv, self).close()
        return self._wrapped_env.close()

    @property
    def action_space(self):
        return self._wrapped_env.action_space

    @property
    def observation_space(self):
        return self._wrapped_env.observation_space

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        self._wrapped_env.terminate()

    def get_param_values(self):
        return self._wrapped_env.get_param_values()

    def set_param_values(self, params):
        self._wrapped_env.set_param_values(params)

    def __getattr__(self, attr):
        orig_attr = self.wrapped_env.__getattribute__(attr)
        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if result == self.wrapped_env:
                    return self
                return result

            return hooked
        else:
            return orig_attr
