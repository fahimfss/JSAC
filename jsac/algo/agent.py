import os
import jax
import time
import copy
import orbax
import shutil
import functools
import numpy as np
from jax import random
import jax.numpy as jnp
import multiprocessing as mp
from threading import Thread
from flax.training import orbax_utils

from jsac.helpers.utils import MODE
from jsac.algo.update_functions import update_jit
from jsac.algo.initializers import init_actor, init_critic
from jsac.algo.replay_buffer import ReplayBuffer, AsyncSMReplayBuffer
from jsac.algo.initializers import init_temperature, init_inference_actor 


class BaseAgent:
    def __init__(self, args):
        for key in args:
            if key == 'seed':
                self._rng = jax.random.PRNGKey(args[key])
            elif key == 'action_shape':
                self._action_shape = args[key]
                self._action_dim = self._action_shape[-1]
                self._target_entropy = -self._action_dim / 2
            else: 
                setattr(self, f'_{key}', args[key])
             
        if self._mode == MODE.PROP:
            self._image_shape = None
            self._image_history = 0
            self._spatial_softmax = False
        else:
            self._obs_queue = mp.Queue()
        
        self._dtype = jnp.float32
        
        self._replay_buffer = None
        self._update_step = 0

    def _init_buffers(self):
        if self._mode == MODE.PROP:
            self._replay_buffer = ReplayBuffer(
                self._image_shape, 
                self._proprioception_shape, 
                self._action_shape,
                self._replay_buffer_capacity, 
                self._batch_size * self._num_critic_updates,
                load_path=self._buffer_load_path)
        else:    
            self._replay_buffer = AsyncSMReplayBuffer(
                self._image_shape, 
                self._proprioception_shape, 
                self._action_shape,
                self._replay_buffer_capacity, 
                self._batch_size * self._num_critic_updates,
                self._obs_queue,
                load_path=self._buffer_load_path)


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
            self._rng, 
            self._critic_lr, 
            init_image_shape,
            init_proprioception_shape, 
            self._net_params,
            self._action_dim, 
            self._spatial_softmax,
            self._mode,
            self._dtype,
            self._num_critic_networks,
            self._global_norm)
        self._critic_target_params = copy.deepcopy(self._critic.params)

        self._rng, self._actor = init_actor(
            self._rng, 
            self._critic, 
            self._actor_lr, 
            init_image_shape,
            init_proprioception_shape, 
            self._net_params,
            self._action_dim, 
            self._spatial_softmax, 
            self._mode,
            self._dtype,
            self._global_norm)

        self._rng, self._temp = init_temperature(
            self._rng, self._temp_lr, self._init_temperature, self._dtype)

        if self._load_model > 0:
            self._load_model_fnc()

    def add(self, state, action, reward, next_state, mask, first_step):
        image, proprioception = self._unpack(state)
        next_image, next_proprioception = self._unpack(next_state)
        
        if self._mode == MODE.PROP:
            self._replay_buffer.add(image, 
                                    proprioception, 
                                    action, 
                                    reward, 
                                    next_image, 
                                    next_proprioception, 
                                    mask,
                                    first_step)
        else:
            self._obs_queue.put((image, 
                                proprioception, 
                                action, 
                                reward,
                                next_image, 
                                next_proprioception, 
                                mask, 
                                first_step))


      
    def update(self):
        self._update_step += 1

        t1 = time.time()
         
        batch = self._replay_buffer.sample() 
                
        self._rng, actor, critic, critic_target_params, temp, info = update_jit(
            self._rng,
            self._actor,
            self._critic,
            self._critic_target_params,
            self._temp,
            batch,
            self._discount,
            self._critic_tau,
            self._target_entropy,
            self._update_step % self._actor_update_freq == 0,
            self._update_step % self._critic_target_update_freq == 0,
            self._num_critic_updates,
            self._ntk
        )

        jax.block_until_ready(actor.params)
        self._actor = actor
        self._critic = critic
        self._critic_target_params = critic_target_params
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
            'critic_target': self._critic_target_params,
            'actor': self._actor,
            'temp': self._temp,
            'step': self._update_step
        }
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        state_restored = orbax_checkpointer.restore(model_dir, item=ckpt)
        self._critic = state_restored['critic']
        self._critic_target_params = state_restored['critic_target']
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
            'critic_target': self._critic_target_params,
            'actor': self._actor,
            'temp': self._temp,
            'step': self._update_step
        }

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(model_dir, ckpt, save_args=save_args)
        
    def get_actor_params(self): 
        if self._sync_mode:
            return self._actor.params
        else:
            return self._actor_params
    
    def set_actor_params(self, params): 
        if self._sync_mode:
            self._actor = self._actor.replace(params=params) 
        else:
            self._actor_params = params
            

class SACRADAgent(BaseAgent):
    def __init__(self, args):
        """
        An implementation of the version of Soft-Actor-Critic 
        described in https://arxiv.org/abs/1812.05905
        """
        super().__init__(args)
        self._init_buffers()
        self._init_models(self._image_shape, self._proprioception_shape)

    def sample_actions(self, state, deterministic=False):
        self._rng, actions = sample_actions(
            self._rng, 
            self._actor.apply_fn, 
            self._actor.params,
            state, 
            self._mode,
            deterministic)

        return np.asarray(actions).clip(-1, 1)
    
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

        self._rng, self._actor_model = init_inference_actor(
            self._rng, 
            self._image_shape, 
            self._proprioception_shape,
            self._net_params, 
            self._action_dim, 
            self._spatial_softmax,
            self._mode,
            self._dtype)

        self._actor_params = self._actor_queue.get()

        self._actor_update_thread = Thread(target=self._get_actor_param)
        self._actor_update_thread.start()

    def _init_async(self):
        self._closeing_lock.acquire()
        self._init_buffers()
        self._init_models(self._image_shape, self._proprioception_shape)
        self._actor_queue.put(self._actor.params)

        if self._save_model:
            self.checkpoint(0)

        self._async_tasks()

    def sample_actions(self, state, deterministic=False):
        with self._actor_lock:
            self._rng, actions = sample_actions(
                self._rng, 
                self._actor_model.apply, 
                self._actor_params,
                state, 
                self._mode,
                deterministic)

        return np.asarray(actions).clip(-1, 1)

    def update(self):
        if not self._update_queue.empty():
            infos = []
            while not self._update_queue.empty():
                info = self._update_queue.get()
                infos.append(info)
            return infos
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

            self._update_queue.put(info[0])
            if self._update_step % self._actor_sync_freq == 0:
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


def process_state(state, mode):
    image_ob = None
    propri_ob = None
    
    if mode == 'img_prop':
        image_ob, propri_ob = state
    elif mode == 'prop':
        propri_ob = state
    elif mode == 'img':
        image_ob = state
    
    if image_ob is not None:
        image_ob = jnp.expand_dims(image_ob, 0)
    
    if propri_ob is not None:
        propri_ob = jnp.expand_dims(propri_ob, 0)

    return image_ob, propri_ob


@functools.partial(jax.jit, static_argnames=('apply_fn', 
                                             'mode', 
                                             'deterministic'))
def sample_actions(rng, 
                   apply_fn, 
                   params, 
                   state,  
                   mode, 
                   deterministic=False):
    rng, key = random.split(rng)
    image_ob, propri_ob = process_state(state, mode)
    if deterministic: 
        actions, _ = apply_fn({"params": params}, 
                              key, 
                              image_ob, 
                              propri_ob, 
                              0.0)
    else:
        actions, _ = apply_fn({"params": params}, 
                              key, 
                              image_ob, 
                              propri_ob)
    
    return rng, jnp.squeeze(actions, 0)

