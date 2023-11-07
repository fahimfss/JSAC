from jax import random
import optax
from jsac.algo.models import ActorModel, CriticModel, Temperature
from flax.training.train_state import TrainState
from flax.core.frozen_dict import freeze
from flax import traverse_util
from jsac.helpers.utils import MODE
import numpy as np


def get_init_data(init_image_shape, init_proprioception_shape, mode):
    init_image = None
    init_proprioception = None 

    if mode == MODE.IMG or mode == MODE.IMG_PROP:
        init_image = np.random.randint(
            0, 256, size=(1, *init_image_shape), dtype=np.uint8)
    if mode == MODE.PROP or mode == MODE.IMG_PROP:
        init_proprioception = np.random.uniform(
            size=(1, *init_proprioception_shape)).astype(np.float32)

    return init_image, init_proprioception
    

def init_inference_actor(rng, init_image_shape, init_proprioception_shape, 
                         action_dim, net_params, spatial_softmax=True, 
                         mode=MODE.IMG_PROP):
    
    model = ActorModel(action_dim=action_dim, net_params=net_params,
                       spatial_softmax=spatial_softmax, mode=mode)
    
    init_image, init_proprioception = get_init_data(
        init_image_shape, init_proprioception_shape, mode)

    rng, key1, key2 = random.split(rng, 3)
    model.init(key1, init_image, init_proprioception, False, key2)['params']

    return rng, model


def init_actor(rng, critic, learning_rate, init_image_shape, 
               init_proprioception_shape, action_dim, net_params, 
               spatial_softmax=True, use_critic_encoder=True, 
               mode=MODE.IMG_PROP):

    model = ActorModel(action_dim=action_dim, net_params=net_params,
                       spatial_softmax=spatial_softmax, mode=mode)

    rng, key1, key2 = random.split(rng, 3)
    
    init_image, init_proprioception = get_init_data(
        init_image_shape, init_proprioception_shape, mode)
    
    params = model.init(key1, init_image, init_proprioception, False, 
                        key2)['params']
    
    if use_critic_encoder:
        
        partition_optimizers = {
            'trainable': optax.adam(learning_rate=learning_rate), 
            'frozen': optax.set_to_zero()}
        
        param_partitions = traverse_util.path_aware_map(
            lambda path, v: 'frozen' if 'encoder' in path else 'trainable', params)
        
        tx = optax.multi_transform(partition_optimizers, param_partitions)

        params['encoder'] = critic.params['encoder']

    else:
        tx = optax.adam(learning_rate=learning_rate)
    
    return rng, TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def init_critic(rng, learning_rate, init_image_shape, init_proprioception_shape, 
                action_dim, net_params, spatial_softmax=True, 
                mode=MODE.IMG_PROP):
    
    model = CriticModel(action_dim=action_dim, net_params=net_params,
                        spatial_softmax=spatial_softmax, mode=mode)
    
    rng, key = random.split(rng)
    init_actions = random.uniform(key, (1, action_dim))

    rng, key = random.split(rng)

    init_image, init_proprioception = get_init_data(
        init_image_shape, init_proprioception_shape, mode)
    
    params = model.init(key, init_image, init_proprioception, 
                        init_actions)['params']

    tx = optax.adam(learning_rate=learning_rate)

    return rng, TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def init_temperature(rng, learning_rate, alpha=1.0):
    model = Temperature(initial_temperature=alpha)
    rng, key = random.split(rng)
    params = model.init(key)['params']

    tx = optax.adam(learning_rate=learning_rate)

    return rng, TrainState.create(apply_fn = model.apply, params=params, tx=tx)

