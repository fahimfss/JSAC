from jax import random
from flax import linen as nn
import optax
from jsac.algo.models import ActorModel, CriticModel, Temperature
from flax.training.train_state import TrainState
from jsac.helpers.utils import MODE
import numpy as np
from flax import traverse_util
from typing import Any
import jax


class ResnetTrainState(TrainState):
    batch_stats: Any
    
    
def get_init_data(init_image_shape, 
                  init_proprioception_shape, 
                  mode):
    init_image = None
    init_proprioception = None 

    if mode == MODE.IMG or mode == MODE.IMG_PROP:
        init_image = np.random.randint(
            0, 256, size=(1, *init_image_shape), dtype=np.uint8)
    if mode == MODE.PROP or mode == MODE.IMG_PROP:
        init_proprioception = np.random.uniform(
            size=(1, *init_proprioception_shape)).astype(np.float32)

    return init_image, init_proprioception
    

def init_critic(rng,
                learning_rate, 
                init_image_shape, 
                init_proprioception_shape, 
                action_dim, 
                net_params, 
                rad_offset, 
                image_model,
                dtype,
                clip_global_norm,
                mode=MODE.IMG_PROP):

    model = CriticModel(net_params, 
                        action_dim, 
                        rad_offset,  
                        image_model,
                        mode,
                        dtype)
    
    rng, *keys = random.split(rng, 4)
    init_actions = random.uniform(keys[0], (1, action_dim), dtype=dtype)

    init_image, init_proprioception = get_init_data(
        init_image_shape, 
        init_proprioception_shape, 
        mode)
    
    variables = model.init(keys[1], 
                        keys[2:],
                        init_image, 
                        init_proprioception, 
                        init_actions) 

    tx = optax.chain(optax.clip_by_global_norm(clip_global_norm), 
                     optax.adam(learning_rate=learning_rate))

    if image_model == 'conv':
        ts = TrainState.create(apply_fn=model.apply,
                               params=variables['params'], 
                               tx=tx)
    else:
        ts = ResnetTrainState.create(apply_fn=model.apply, 
                                     params=variables['params'], 
                                     tx=tx, 
                                     batch_stats=variables['batch_stats'])
    return rng, ts


def init_inference_actor(rng, 
                         init_image_shape, 
                         init_proprioception_shape, 
                         action_dim, 
                         net_params, 
                         rad_offset,
                         image_model,
                         dtype,
                         mode=MODE.IMG_PROP):
    
    model = ActorModel(net_params,
                       action_dim,  
                       rad_offset,
                       image_model, 
                       mode,
                       dtype)
    
    init_image, init_proprioception = get_init_data(
        init_image_shape, 
        init_proprioception_shape, 
        mode)

    rng, *keys = random.split(rng, 5)
    model.init(keys[0], 
               keys[1:],
               init_image, 
               init_proprioception)

    return rng, model

def init_actor(rng, 
               critic,
               learning_rate, 
               init_image_shape, 
               init_proprioception_shape, 
               action_dim, 
               net_params, 
               rad_offset,
               image_model,
               dtype,  
               mode=MODE.IMG_PROP):
    
    model = ActorModel(net_params,
                       action_dim,  
                       rad_offset, 
                       image_model,
                       mode,
                       dtype)

    rng, *keys = random.split(rng, 5)
    
    init_image, init_proprioception = get_init_data(
        init_image_shape, 
        init_proprioception_shape, 
        mode)
    
    params = model.init(keys[0], 
                        keys[1:],
                        init_image,
                        init_proprioception)['params']
    
    # We do not train the actor encoder. Instead we copy
    # the encoder values from the critic.
    if mode==MODE.IMG_PROP or mode==MODE.IMG:
        params['encoder'] = critic.params['encoder']
        partition_optimizers = {'trainable': optax.adam(learning_rate=learning_rate), 
                                'frozen': optax.set_to_zero()}
        param_partitions = traverse_util.path_aware_map(
            lambda path, v: 'frozen' if 'encoder' in path else 'trainable', params)
        tx = optax.multi_transform(partition_optimizers, param_partitions)
    else:
        tx = optax.adam(learning_rate=learning_rate, mu_dtype=dtype)
    
    return rng, TrainState.create(apply_fn=model.apply, 
                                  params=params, 
                                  tx=tx)


def init_temperature(rng, learning_rate, dtype, alpha=1.0):
    model = Temperature(alpha, dtype)
    rng, key = random.split(rng)
    params = model.init(key)['params']

    tx = optax.adam(learning_rate=learning_rate, mu_dtype=dtype)

    return rng, TrainState.create(apply_fn=model.apply, 
                                  params=params, 
                                  tx=tx)
