from jsac.algo.replay_buffer import Batch, AsyncSMRadReplayBuffer

import jax
from jax import vmap, random
from jsac.algo.initializers import init_actor, init_critic
from jsac.algo.initializers import init_temperature, init_inference_actor
from jsac.algo.update_functions import critic_update, actor_update
from jsac.algo.update_functions import target_update, temp_update
from jsac.helpers.utils import MODE
import copy
import functools
import numpy as np
import time
import os
import shutil
import jax.numpy as jnp
from flax.core.frozen_dict import freeze
import multiprocessing as mp
from threading import Thread
from flax.training import orbax_utils
import orbax


class BaseAgent:
    def __init__(self, args):
        self._rng = jax.random.PRNGKey(args.seed)

        self._mode = args.mode
        self._apply_rad = args.apply_rad
        self._image_shape = args.image_shape

        if self._apply_rad:
            h, w, c = self._image_shape
            self._rad_height = int(h * args.rad_offset)
            self._rad_width = int(w * args.rad_offset)

            if self._rad_height == 0:
                self._rad_height = 1
            if self._rad_width == 0:
                self._rad_width = 1

            self._rad_image_shape = ((h - (2 * self._rad_height)),
                                     (w - (2 * self._rad_width)), c)

        else:
            self._rad_height = 0
            self._rad_width = 0

        self._proprioception_shape = args.proprioception_shape
        self._action_shape = args.action_shape
        self._action_dim = args.action_shape[-1]
        self._target_entropy = -self._action_dim / 2
        self._replay_buffer_capacity = args.replay_buffer_capacity
        self._batch_size = args.batch_size
        self._init_steps = args.init_steps
        self._critic_tau = args.critic_tau
        self._actor_update_freq = args.actor_update_freq
        self._critic_target_update_freq = args.critic_target_update_freq
        self._discount = args.discount
        self._use_critic_encoder = args.use_critic_encoder
        self._critic_lr = args.critic_lr
        self._net_params = args.net_params
        self._spatial_softmax = args.spatial_softmax
        self._actor_lr = args.actor_lr
        self._temp_lr = args.temp_lr
        self._init_temperature = args.init_temperature
        self._sync_mode = args.sync_mode 
        self._cur_env_step = 0

        self._replay_buffer = None

        self._load_model = args.load_model
        self._model_dir = args.model_dir
        self._buffer_save_path = args.buffer_save_path
        self._buffer_load_path = args.buffer_load_path
        self._total_env_steps = args.env_steps

        self._update_step = 0

    def _unpack(self, state):
        if self._mode == MODE.IMG:
            image = state
            proprioception = None
        elif self._mode == MODE.PROP:
            image = None
            proprioception = state
        elif self._mode == MODE.IMG_PROP:
            image, proprioception = state

        return image, proprioception

    def _init_models(self, init_image_shape, init_proprioception_shape):
        self._rng, self._critic = init_critic(
            self._rng, self._critic_lr, init_image_shape,
            init_proprioception_shape, self._action_dim, self._net_params,
            self._spatial_softmax, self._mode)

        self._rng, self._critic_target = init_critic(
            self._rng, self._critic_lr, init_image_shape,
            init_proprioception_shape, self._action_dim, self._net_params,
            self._spatial_softmax, self._mode)

        critic_target_params = copy.deepcopy(self._critic.params)
        self._critic_target = self._critic_target.replace(
            params=critic_target_params)

        self._rng, self._actor = init_actor(
            self._rng, self._critic, self._actor_lr, init_image_shape,
            init_proprioception_shape, self._action_dim, self._net_params,
            self._spatial_softmax, self._use_critic_encoder, self._mode)

        self._rng, self._temp = init_temperature(
            self._rng, self._temp_lr, self._init_temperature)

        if self._load_model > 0:
            self._load_model_fnc()

            
    def update(self):
        self._update_step += 1

        t1 = time.time()
        batch = self._replay_buffer.sample()

        self._rng, actor, critic, critic_target, temp, info = update_jit(
            self._rng,
            self._actor,
            self._critic,
            self._critic_target,
            self._temp,
            batch,
            self._discount,
            self._critic_tau,
            self._target_entropy,
            self._update_step % self._actor_update_freq == 0,
            self._update_step % self._critic_target_update_freq == 0,
            self._use_critic_encoder,
            self._apply_rad,
            self._rad_height,
            self._rad_width)

        jax.block_until_ready(actor.params)
        self._actor = actor
        self._critic = critic
        self._critic_target = critic_target
        self._temp = temp

        t2 = time.time()

        info['update_time'] = (t2 - t1) * 1000
        info['num_updates'] = self._update_step

        return [info]

    def _load_model_fnc(self):
        model_dir = os.path.join(self._model_dir, str(self._load_model)) 
        assert os.path.exists(model_dir)
        ckpt = {
            'critic': self._critic,
            'critic_target': self._critic_target,
            'actor': self._actor,
            'temp': self._temp,
            'step': self._update_step
        }
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        state_restored = orbax_checkpointer.restore(model_dir, item=ckpt)
        self._critic = state_restored['critic']
        self._critic_target = state_restored['critic_target']
        self._actor = state_restored['actor']
        self._temp = state_restored['temp']
        self._update_step = state_restored['step']

        print(f'Restored models from step: {self._load_model}')

    def _save_model_fnc(self, step):
        print(f'Saving model, steps: {step}.')
        model_dir = os.path.join(self._model_dir, str(step)) 
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

        ckpt = {
            'critic': self._critic,
            'critic_target': self._critic_target,
            'actor': self._actor,
            'temp': self._temp,
            'step': self._update_step
        }

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(model_dir, ckpt, save_args=save_args)


class SACRADAgent(BaseAgent):
    def __init__(self, args):
        """
        An implementation of the version of Soft-Actor-Critic 
        described in https://arxiv.org/abs/1812.05905
        """
        super().__init__(args)

        self._obs_queue = mp.Queue()

        self._replay_buffer = AsyncSMRadReplayBuffer(
            self._image_shape, self._proprioception_shape, self._action_shape,
            self._replay_buffer_capacity, self._batch_size, self._obs_queue,
            self._init_steps, self._buffer_load_path)

        image_shape = self._image_shape

        if self._apply_rad:
            image_shape = self._rad_image_shape

        self._init_models(image_shape, self._proprioception_shape)

    def add(self, state, action, reward, next_state, done):
        image, proprioception = self._unpack(state)
        next_image, next_proprioception = self._unpack(next_state)

        self._obs_queue.put((image, proprioception, action, reward,
                             next_image, next_proprioception, done))

    def sample_actions(self, state, deterministic=False):
        if deterministic:
            actions = sample_actions_deterministic(
                self._actor.apply_fn, self._actor.params, state, self._mode,
                self._apply_rad, self._rad_height, self._rad_width)
        else:
            self._rng, actions = sample_actions(
                self._rng, self._actor.apply_fn, self._actor.params, state,
                self._mode, self._apply_rad, self._rad_height, self._rad_width)
        
        return np.asarray(actions)
    
    def checkpoint(self, step):
        self._save_model_fnc(step)

    def close(self, without_save=False):
        if not without_save and self._buffer_save_path:
            self._replay_buffer.save(self._buffer_save_path)
        self._replay_buffer.close()


class AsyncSACRADAgent(BaseAgent):

    def __init__(self, args):
        """
        An implementation of the version of Soft-Actor-Critic 
        described in https://arxiv.org/abs/1812.05905
        """
        super().__init__(args)

        self._obs_queue = mp.Queue()
        self._actor_queue = mp.Queue()
        self._update_queue = mp.Queue()
        self._instructions_queue = mp.Queue()

        self._actor_lock = mp.Lock()
        self._closeing_lock = mp.Lock()

        self._pause_update = True

        self._update_process = mp.Process(target=self._init_async)
        self._update_process.start()

        image_shape = self._image_shape
        if self._apply_rad:
            image_shape = self._rad_image_shape

        self._rng, self._actor_model = init_inference_actor(
            self._rng, image_shape, self._proprioception_shape,
            self._action_dim, self._net_params, self._spatial_softmax, 
            self._mode)

        self._actor_params = self._actor_queue.get()

        self._actor_update_thread = Thread(target=self._get_actor_param)
        self._actor_update_thread.start()

    def _init_async(self):
        self._closeing_lock.acquire()

        self._replay_buffer = AsyncSMRadReplayBuffer(
            self._image_shape, self._proprioception_shape, self._action_shape,
            self._replay_buffer_capacity, self._batch_size, self._obs_queue,
            self._init_steps, self._buffer_load_path)

        image_shape = self._image_shape

        if self._apply_rad:
            image_shape = self._rad_image_shape

        self._init_models(image_shape, self._proprioception_shape)
        
        self._actor_queue.put(self._actor.params)

        self._async_tasks()

    def add(self, state, action, reward, next_state, done):
        image, proprioception = self._unpack(state)
        next_image, next_proprioception = self._unpack(next_state)

        self._obs_queue.put((image, proprioception, action, reward,
                             next_image, next_proprioception, done))
    
    def update_masks(self, target_size):
        self._obs_queue.put('update_masks')
        self._obs_queue.put(target_size)

    def sample_actions(self, state, deterministic=False):
        with self._actor_lock:
            if deterministic:
                actions = sample_actions_deterministic(
                    self._actor_model.apply, self._actor_params, state,
                    self._mode, self._apply_rad, self._rad_height,
                    self._rad_width)
            else:
                self._rng, actions = sample_actions(
                    self._rng, self._actor_model.apply, self._actor_params,
                    state, self._mode, self._apply_rad, self._rad_height,
                    self._rad_width)

        return np.asarray(actions)

    def update(self):
        if not self._update_queue.empty():
            info = []
            while not self._update_queue.empty():
                info.append(self._update_queue.get())
            return info
        else:
            return None

    def _get_actor_param(self):
        while True:
            data = self._actor_queue.get()
            if isinstance(data, str):
                if data == 'close':
                    return
            with self._actor_lock:
                self._actor_params = data

    def _async_tasks(self):
        while True:
            if self._pause_update or not self._instructions_queue.empty():
                ins = self._instructions_queue.get()
                if ins == 'resume':
                    print('Updates resumed. Completed ' + 
                          f'{self._update_step} updates.')
                    self._pause_update = False
                elif ins == 'pause':
                    print('Updates paused. Completed ' + 
                          f'{self._update_step} updates.')
                    self._pause_update = True
                    continue
                elif ins == 'checkpoint':
                    step = int(self._instructions_queue.get())
                    self._save_model_fnc(step)
                    continue
                elif ins == 'close':
                    print('Closing asynchronous updates. ' 
                          f'Completed {self._update_step} updates.')
                    if self._buffer_save_path:
                        self._replay_buffer.save(self._buffer_save_path)
                    self._replay_buffer.close()
                    self._closeing_lock.release()
                    return
                
                elif ins == 'close_no_save':
                    print('Closing asynchronous updates. ' 
                          f'Completed {self._update_step} updates.')
                    self._replay_buffer.close()
                    self._closeing_lock.release()
                    return

            info = super().update()

            if self._update_step >= 10:
                self._update_queue.put(info[0])
                if self._update_step % self._actor_update_freq == 0:
                    self._actor_queue.put(self._actor.params)        

    def pause_update(self):
        if self._pause_update:
            return
        self._pause_update = True
        self._instructions_queue.put('pause')

    def resume_update(self):
        if not self._pause_update:
            return
        self._pause_update = False
        self._instructions_queue.put('resume')

    def checkpoint(self, step):
        self._instructions_queue.put('checkpoint')
        self._instructions_queue.put(step)

    def close(self, without_save=False):
        if without_save:
            self._instructions_queue.put('close_no_save')
        else:
            self._instructions_queue.put('close')
        with self._closeing_lock:
            self._actor_queue.put('close')
            time.sleep(1)
            self._update_process.terminate()
            self._update_process.join()


def process_state(state, mode, apply_rad, rad_height, rad_width):
    image_ob = None
    propri_ob = None
    
    if mode == 'img_prop':
        image_ob, propri_ob = state
    elif mode == 'prop':
        propri_ob = state
    elif mode == 'img':
        image_ob = state
    
    if image_ob is not None:
        if apply_rad:
            h, w, c = image_ob.shape
            rad_image_shape = ((h - (2 * rad_height)), (w - (2 * rad_width)), c)
            image_ob = jax.lax.dynamic_slice(
                image_ob, (rad_height, rad_width, 0), rad_image_shape)
        image_ob = jnp.expand_dims(image_ob, 0)
    
    if propri_ob is not None:
        propri_ob = jnp.expand_dims(propri_ob, 0)

    return image_ob, propri_ob


@functools.partial(jax.jit, static_argnames=('apply_fn', 'mode', 'apply_rad',
                                             'rad_height', 'rad_width'))
def sample_actions(rng, apply_fn, params, state, mode, apply_rad, rad_height,
                   rad_width):
    rng, key = random.split(rng)
    image_ob, propri_ob = process_state(state, mode, apply_rad, rad_height,
                                        rad_width)
    _, actions, _, _ = apply_fn({"params": params}, image_ob, propri_ob,
                                False, key)
    return rng, jnp.squeeze(actions, 0)


@functools.partial(jax.jit, static_argnames=('apply_fn', 'mode', 'apply_rad',
                                             'rad_height', 'rad_width'))
def sample_actions_deterministic(apply_fn, params, state, mode, apply_rad,
                                 rad_height, rad_width):
    image_ob, propri_ob = process_state(state, mode, apply_rad, rad_height,
                                        rad_width)
    actions = apply_fn({"params": params}, image_ob, propri_ob, True)
    return jnp.squeeze(actions, 0)


@functools.partial(jax.jit, static_argnames=('image_shape'))
def augment(image, start_h, start_w, image_shape):
    return jax.lax.dynamic_slice(image, (start_h, start_w, 0), image_shape)


@functools.partial(jax.jit, static_argnames=('update_actor',
                                             'update_target',
                                             'use_critic_encoder',
                                             'apply_rad',
                                             'rad_height',
                                             'rad_width'))
def update_jit(rng, actor, critic, critic_target, temp, batch, discount, tau,
               target_entropy, update_actor, update_target, use_critic_encoder,
               apply_rad, rad_height, rad_width):

    if apply_rad:
        rng, key1, key2 = random.split(rng, 3)
        batch_size = batch.images.shape[0]

        rad_height_maxval = (2 * rad_height) + 1
        rad_width_maxval = (2 * rad_width) + 1

        rand_height = random.randint(
            key1, minval=0, maxval=rad_height_maxval, shape=(batch_size,))
        rand_width = random.randint(
            key2, minval=0, maxval=rad_width_maxval, shape=(batch_size,))

        h, w, c = batch.images.shape[1:]
        rad_image_shape = ((h - (2 * rad_height)), (w - (2 * rad_width)), c)

        get_augments = vmap(augment, in_axes=(0, 0, 0, None))

        rad_images = get_augments(batch.images, rand_height, rand_width,
                                  rad_image_shape)
        rad_next_images = get_augments(batch.next_images, rand_height,
                                       rand_width, rad_image_shape)

        batch = Batch(
            images=rad_images,
            proprioceptions=batch.proprioceptions,
            actions=batch.actions,
            rewards=batch.rewards,
            masks=batch.masks,
            next_images=rad_next_images,
            next_proprioceptions=batch.next_proprioceptions)

    rng, critic_new, critic_info = critic_update(
        rng, actor, critic, critic_target, temp, batch, discount)

    if update_target:
        new_critic_target = target_update(critic_new, critic_target, tau)
    else:
        new_critic_target = critic_target

    if update_actor:
        rng, new_actor, actor_info = actor_update(rng, actor, critic_new, temp,
                                                  batch, use_critic_encoder)

        new_temp, alpha_info = temp_update(temp, actor_info['entropy'],
                                           target_entropy)
    else:
        new_actor = actor
        new_temp = temp
        actor_info = {}
        alpha_info = {}

    return rng, new_actor, critic_new, new_critic_target, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }
