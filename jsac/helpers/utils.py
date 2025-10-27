import os
import time
import random
import collections
import numpy as np
import pandas as pd
import seaborn as sns
from gymnasium import spaces
from gymnasium.core import Env
import matplotlib.pyplot as plt


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

def save_learning_curve(fname, 
                        returns, 
                        ep_lens, 
                        xtick):
    sns.set_theme(rc={'figure.figsize':(10, 7)})
    sns.set_style("whitegrid")

    df = pd.DataFrame(columns=["Step", "Return"])

    steps = 0
    rets = []
    end_step = xtick
    data = []
    for (i, epi_s) in enumerate(ep_lens):
        if steps + epi_s > end_step:
            if len(rets) > 0:
                data.append([end_step, sum(rets)/len(rets)]) 
                rets = []
            end_step += xtick
        
        steps += epi_s 
        ret = returns[i]  
        rets.append(ret) 
        
    if len(rets) > 0:  
        data.append([end_step, sum(rets)/len(rets)])
    df = pd.DataFrame(data, columns=["Step", "Return"])
    sns.lineplot(x="Step", y='Return', data=df, 
                 color=sns.color_palette('bright')[0], 
                 linewidth=1.5, errorbar=None)

    plt.title('Learning Curve')
    plt.savefig(fname)
    plt.close()
    
def save_eval_learning_curve(fname, 
                             returns, 
                             steps):
    sns.set_theme(rc={'figure.figsize':(10, 7)})
    sns.set_style("whitegrid")

    df = pd.DataFrame(columns=["Step", "Return"])

    curr_step = steps[0]
    rets = [] 
    data = []
    for (i, step) in enumerate(steps):
        if step != curr_step:
            if len(rets) > 0:
                data.append([curr_step, sum(rets)/len(rets)]) 
                rets = []
            curr_step = step
         
        ret = returns[i]  
        rets.append(ret) 
        
    if len(rets) > 0:  
        data.append([curr_step, sum(rets)/len(rets)])
        
    df = pd.DataFrame(data, columns=["Step", "Return"])
    sns.lineplot(x="Step", y='Return', data=df, 
                 color=sns.color_palette('bright')[0], 
                 linewidth=1.5, errorbar=None)

    plt.title('Eval Learning Curve')
    plt.savefig(fname)
    plt.close()

def get_episode_and_steps_from_log(path):
    path = os.path.join(path, 'train.log')
    assert os.path.isfile(path), 'Log file (train.log) does not exist. Aborting...'
    episode, step = 0, 0
    with open(path, 'r') as fl:
        while True:
            line = fl.readline()
            if not line:
                break
            dc = eval(line)
            episode = int(dc['episode'])
            step = int(dc['step'])
    return episode + 1, step 


## SRC: https://github.com/kindredresearch/SenseAct/blob/master/senseact/utils.py

class EnvSpec():
    def __init__(self, 
                 env_spec, 
                 observation_space, 
                 action_space):
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
            new_info['truncated'] = True

        if 'TimeLimit.truncated' in info or 'truncated' in info:
            new_info['truncated'] = True

        if done or (not self._is_min_time and 'truncated' in new_info):
            new_info['episode'] = self._episode
            new_info['step'] = self._total_steps
            new_info['episode_steps'] = self._episode_steps
            new_info['duration'] = time.time() - self._start_time
            new_info['return'] = self._reward_sum
            self._episode += 1
            return done, new_info
        
        if self._is_min_time:
            self._sub_episode_steps += 1
            if self._sub_episode_steps == self._episode_max_steps:
                self._reward_sum += self._reward_penalty
                self._total_steps += self._steps_penalty
                self._episode_steps += self._steps_penalty
                
                new_info['truncated'] = True
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
        if isinstance(self._wrapped_env.action_space, spaces.Box):
            lb = self._wrapped_env.action_space.low
            ub = self._wrapped_env.action_space.high
            scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
            scaled_action = np.clip(scaled_action, lb, ub)
        else:
            scaled_action = action

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_state, reward, terminated, truncated, info = wrapped_step
        done = terminated | truncated
        done, info = self._monitor(reward, done, info)

        return Step(next_state, reward * self._reward_scale, done, info)

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
        if hasattr(super(), 'close'):
            super().close()
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
        try:
            orig_attr = self._wrapped_env.__getattribute__(attr)
            if callable(orig_attr):
                def hooked(*args, **kwargs):
                    result = orig_attr(*args, **kwargs)
                    # prevent wrapped_class from becoming unwrapped
                    if result == self._wrapped_env:
                        return self
                    return result
                return hooked
            else:
                return orig_attr
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")



            
            
            
