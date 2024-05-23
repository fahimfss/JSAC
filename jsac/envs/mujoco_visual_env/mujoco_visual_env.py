import cv2
import gymnasium as gym
from collections import deque
from gymnasium.spaces import Box
import numpy as np
from jsac.helpers.utils import MODE


class MujocoVisualEnv(gym.Wrapper):
    def __init__(self, env_name, mode, seed=0, image_stack=1, 
                 image_width=120, image_height=120, img_type='hwc'):
        super().__init__(gym.make(env_name, render_mode="rgb_array"))  
        self._mode = mode
        self._img_type = img_type

        if self._mode == MODE.IMG or self._mode == MODE.IMG_PROP:
            if self._img_type == 'chw':
                self._channel_axis = 0
                self._image_shape = (image_stack * 3, image_height, image_width)
            else:
                self._channel_axis = -1
                self._image_shape = (image_height, image_width, image_stack * 3)
            
            self._image_buffer = deque([], maxlen=image_stack)
        else:
            self._image_shape = (0, 0, 0)

        self.env.reset(seed=seed) 
        
        # remember to reset 
        self._latest_image = None
        self._reset = False

    @property
    def image_space(self):
        return Box(low=0, high=255, shape=self._image_shape)

    @property
    def proprioception_space(self):
        return self.env.observation_space


    def step(self, a):
        assert self._reset
        ob, reward, terminated, truncated, info = self.env.step(a)
        done = terminated or truncated 
        ob = self._get_ob(ob)
         

        if self._mode == MODE.IMG or self._mode == MODE.IMG_PROP:
            new_img = self._get_new_img()
            self._image_buffer.append(new_img)
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
        ob = self.env.reset() 
        ob = self._get_ob(ob[0])

        if self._mode == MODE.IMG or self._mode == MODE.IMG_PROP:
            new_img = self._get_new_img()
            for _ in range(self._image_buffer.maxlen):
                self._image_buffer.append(new_img)

            self._latest_image = np.concatenate(self._image_buffer, 
                                                axis=self._channel_axis)
        
        self._reset = True
        
        if self._mode == MODE.IMG:
            return self._latest_image
        if self._mode == MODE.PROP:
            return ob
        return (self._latest_image, ob)

    def _get_new_img(self):
        img = self.env.render()
        if self._img_type == 'chw':
            c, h, w = self._image_shape
        else:
            h, w, c = self._image_shape

        img = cv2.resize(img, (h, w), interpolation=cv2.INTER_AREA)

        if self._img_type == 'chw':
            img = np.transpose(img, [2, 0, 1])

        return img
    
    def seed(self, seed):
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
    
    def _get_ob(self, ob):
        return ob

    def close(self):
        super().close()
        del self

# def test_env():
#     env = MujocoVisualEnv('InvertedDoublePendulum-v2', True)
#     img, prop = env.reset(save_img=True)
#     print(f'0\tprop:{prop}')
#     for i in range(15):
#         action = env.action_space.sample()
#         img, prop, reward, done, info = env.step(action)
#         print(f'{i+1}\tprop:{prop}\treward:{reward}')

#         if done:
#             img, prop = env.reset(save_img=True)
#             print(f'0\tprop:{prop}')


# test_env()