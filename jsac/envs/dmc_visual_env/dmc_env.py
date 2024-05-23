import gymnasium as gym
from gymnasium.spaces import Box
from dm_control import suite
from collections import deque
import numpy as np 
from jsac.helpers.utils import MODE


# https://github.com/gauthamvasan/rl_suite/blob/main/rl_suite/envs/dm_control_wrapper.py
ENV_MAP = {
    "acrobot":                  {'domain': "acrobot", "task": "swingup"},
    "acrobot_sparse":           {'domain': "acrobot", "task": "swingup_sparse"},
    "ball_in_cup":              {"domain": "ball_in_cup", "task": "catch"},
    "cartpole_balance":         {"domain": "cartpole", "task": "balance"},
    "cartpole_balance_sparse":  {"domain": "cartpole", "task": "balance_sparse"},
    "cartpole_swingup":         {"domain": "cartpole", "task": "swingup"},
    "cartpole_swingup_sparse":  {"domain": "cartpole", "task": "swingup_sparse"},
    "cheetah":                  {"domain": "cheetah", "task": "run"},
    "finger_spin":              {"domain": "finger", "task": "spin"},
    "finger_turn_easy":         {"domain": "finger", "task": "turn_easy"},
    "finger_turn_hard":         {"domain": "finger", "task": "turn_hard"},
    "fish_upright":             {"domain": "fish", "task": "upright"},
    "fish_swim":                {"domain": "fish", "task": "swim"},
    "hopper_stand":             {"domain": "hopper", "task": "stand"},
    "hopper_hop":               {"domain": "hopper", "task": "hop"},
    "humanoid_stand":           {"domain": "humanoid", "task": "stand"},
    "humanoid_walk":            {"domain": "humanoid", "task": "walk"},
    "humanoid_run":             {"domain": "humanoid", "task": "run"},
    "manipulator_bring_ball":   {"domain": "manipulator", "task": "bring_ball"},
    "pendulum_swingup":         {"domain": "pendulum", "task": "swingup"},
    "point_mass_easy":          {"domain": "point_mass", "task": "easy"},
    "reacher_easy":             {"domain": "reacher", "task": "easy"},
    "reacher_hard":             {"domain": "reacher", "task": "hard"},
    "swimmer6":                 {"domain": "swimmer", "task": "swimmer6"},
    "swimmer15":                {"domain": "swimmer", "task": "swimmer15"},
    "walker_stand":             {"domain": "walker", "task": "stand"},
    "walker_walk":              {"domain": "walker", "task": "walk"},
    "walker_run":               {"domain": "walker", "task": "run"},
}
    

class DMCVisualEnv(gym.Wrapper):
    def __init__(self, 
                 env_name, 
                 mode, 
                 seed=0, 
                 image_history=3, 
                 image_width=160, 
                 image_height=120,
                 cameras=2, 
                 img_type='hwc'):
        
        """
        Last channel(s) has the latest images.
        If image size is (120, 160, 18) (image_history=3, cameras=2)
        Last images can be found using [:, :, -6:]
        """
        
        assert env_name in ENV_MAP, f'"{env_name}" is not recognized.'
        domain_name = ENV_MAP[env_name]['domain']
        task_name = ENV_MAP[env_name]['task']
        self._env = suite.load(domain_name=domain_name, 
                               task_name=task_name,
                               task_kwargs={'random': seed})
        
        
        self._mode = mode
        self._img_type = img_type
        self._cameras = cameras
        self._image_height = image_height
        self._image_width = image_width

        if self._mode == MODE.IMG or self._mode == MODE.IMG_PROP:
            channel_size = cameras * image_history * 3
            if self._img_type == 'chw':
                self._channel_axis = 0
                self._image_shape = (channel_size, image_height, image_width)
            else:
                self._channel_axis = -1
                self._image_shape = (image_height, image_width, channel_size)
            
            self._image_buffer = deque([], maxlen=image_history)
        else:
            self._image_shape = (0, 0, 0)
            
        # Observation space
        self._obs_dim = 0
        for key, val in self._env.observation_spec().items():
            if val.shape:
                self._obs_dim += val.shape[0]
            else:
                self._obs_dim += 1
        
        # Action space
        self._action_dim = self._env.action_spec().shape[0]

        self._latest_image = None
        self._reset = False

    @property
    def image_space(self):
        return Box(low=0, high=255, shape=self._image_shape)

    @property
    def proprioception_space(self):
        return self.observation_space
    
    @property
    def observation_space(self):
        return Box(shape=(self._obs_dim,), high=10, low=-10)
    
    @property
    def action_space(self):
        return Box(shape=(self._action_dim,), high=1, low=-1)


    def step(self, a):
        assert self._reset 
        
        time_step = self._env.step(a)
        ob = self._make_obs(time_step.observation)
        reward = time_step.reward
        done = time_step.last()
        info = {}  

        if self._mode == MODE.IMG or self._mode == MODE.IMG_PROP: 
            images = []
            for cam in range(self._cameras):
                images.append(self._get_new_img(cam)) 
            new_image = np.concatenate(images, 
                                       axis=self._channel_axis)
            self._image_buffer.append(new_image)
            self._latest_image = np.concatenate(self._image_buffer, 
                                                axis=self._channel_axis)

        if done:
            self._reset = False

        if self._mode == MODE.IMG:
            return self._latest_image, reward, done, info
        if self._mode == MODE.PROP:
            return ob, reward, done, info
        return (self._latest_image, ob), reward, done, info 

    def reset(self):
        time_step = self._env.reset() 
        ob = self._make_obs(time_step.observation)
        
        if self._mode == MODE.IMG or self._mode == MODE.IMG_PROP:
            images = []
            for cam in range(self._cameras):
                images.append(self._get_new_img(cam)) 
            new_image = np.concatenate(images, 
                                       axis=self._channel_axis)
            for _ in range(self._image_buffer.maxlen):
                self._image_buffer.append(new_image)
            self._latest_image = np.concatenate(self._image_buffer, 
                                                axis=self._channel_axis)
        
        self._reset = True
        
        if self._mode == MODE.IMG:
            return self._latest_image
        if self._mode == MODE.PROP:
            return ob
        return (self._latest_image, ob)

    def _get_new_img(self, cam):
        img = self._env.physics.render(height=self._image_height, 
                                       width=self._image_width, 
                                       camera_id=cam)
        if self._img_type == 'chw':
            img = np.transpose(img, [2, 0, 1])

        return img
    
    def _make_obs(self, x):
        obs = []
        for _, val in x.items():
            obs.append(val.ravel())
        return np.concatenate(obs)

    def close(self):
        super().close()
        del self


