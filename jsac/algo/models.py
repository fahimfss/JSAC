from typing import Optional, Sequence

import flax
import flax.linen as nn
import jax.numpy as jnp
from jax import random
from jsac.helpers.utils import MODE


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class SpatialSoftmax(nn.Module):
    height: float
    width: float
    channel: float
    temp: float = 1.0

    def setup(self):
      pos_x, pos_y = jnp.meshgrid(
         jnp.linspace(-1., 1., self.height),
         jnp.linspace(-1., 1., self.width)
      )
      self._pos_x = pos_x.reshape(self.height*self.width)
      self._pos_y = pos_y.reshape(self.height*self.width)

    #   self._temperature = self.param(
    #       'temperature', 
    #       nn.initializers.constant(self.temp), (1,)) 

    @nn.compact
    def __call__(self, feature):  
        feature = feature.transpose(0, 3, 1, 2)
        feature = feature.reshape(-1, self.height*self.width)

        # feature = feature/self._temperature
    
        softmax_attention = nn.activation.softmax(feature, axis = -1)

        expected_x = jnp.sum(self._pos_x*softmax_attention, axis = 1, 
                             keepdims=True)
        expected_y = jnp.sum(self._pos_y*softmax_attention, axis = 1,
                             keepdims=True)

        expected_xy = jnp.concatenate(axis = 1, arrays=(expected_x, expected_y))
        
        feature_keypoints = expected_xy.reshape(-1, self.channel * 2) 
        
        return feature_keypoints
    

class Encoder(nn.Module):
    net_params: dict 
    spatial_softmax: bool = True
    mode: str = MODE.IMG_PROP

    @nn.compact
    def __call__(self, images, proprioceptions):
        if self.mode == MODE.PROP:
            return proprioceptions
        
        conv_params = self.net_params['conv']

        x = images / 255.0
        height, width, channel = images.shape[1:]

        for i, (_, out_channel, kernel_size, stride) in enumerate(conv_params):
            layer_name = 'encoder_conv_' + str(i)
            x = nn.Conv(features=out_channel, 
                        kernel_size=(kernel_size, kernel_size),
                        strides=stride,
                        padding=0,
                        kernel_init=nn.initializers
                        .delta_orthogonal(column_axis=-1),
                        name=layer_name 
            )(x)

            if i < len(conv_params) - 1:
                x = nn.relu(x)
        
        if self.spatial_softmax:
            b, height, width, channel = x.shape
            x = SpatialSoftmax(height=height, width=width, channel=channel, 
                               name='encoder_spatialsoftmax')(x)
        else:
            x = x.reshape((x.shape[0], -1))
            x = nn.Dense(self.net_params['latent'], kernel_init=default_init(), 
                         name='encoder_dense')(x)
            x = nn.LayerNorm(name='encoder_layernorm')(x)

        if self.mode == MODE.IMG_PROP:
           x = jnp.concatenate(axis = -1, arrays=(x, proprioceptions)) 

        return x


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activate_final: int = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = nn.relu(x)
        return x


LOG_STD_MIN = -10.0
LOG_STD_MAX = 10.0


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * jnp.power(noise, 2) - \
                log_std).sum(-1, keepdims=True)
    return residual - 0.5 * jnp.log(2 * jnp.pi) * \
        noise.shape[-1]


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = nn.tanh(mu)
    if pi is not None:
        pi = nn.tanh(pi)
    if log_pi is not None:
        log_pi -= jnp.log(nn.relu(1 - jnp.power(pi, 2)) + \
                          1e-6).sum(-1, keepdims=True)
    return mu, pi, log_pi


class ActorModel(nn.Module):
    action_dim: int
    net_params: dict 
    spatial_softmax: bool = True
    mode: str = MODE.IMG_PROP

    @nn.compact
    def __call__(self, images, proprioceptions, deterministic=False, key=None):
        
        latents = Encoder(self.net_params, self.spatial_softmax,
                          name='encoder',
                          mode=self.mode)(images, proprioceptions)
        
        outputs = MLP(self.net_params['mlp'], activate_final=True)(latents)

        mu = nn.Dense(self.action_dim, kernel_init=default_init(0.0))(outputs)
        
        if deterministic:
            return nn.tanh(mu)

        log_std = nn.Dense(self.action_dim, kernel_init=default_init(0.0))(outputs)

        log_std = nn.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (
            LOG_STD_MAX - LOG_STD_MIN
        ) * (log_std + 1)

        std = jnp.exp(log_std)
        noise = random.normal(key, mu.shape)
        pi = mu + noise * std

        log_pi = gaussian_logprob(noise, log_std)

        mu, pi, log_pi = squash(mu, pi, log_pi)
        log_pi = jnp.squeeze(log_pi, -1)
        return mu, pi, log_pi, log_std
    
    def __hash__(self): 
        return id(self)


class QFunction(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, latents, actions):
        inputs = jnp.concatenate([latents, actions], -1)
        critic = MLP((*self.hidden_dims, 1))(inputs)
        return jnp.squeeze(critic, -1)


class CriticModel(nn.Module):
    action_dim: int
    net_params: dict  
    spatial_softmax: bool = True
    mode: str = MODE.IMG_PROP
    num_qs: int = 2

    @nn.compact
    def __call__(self, images, proprioceptions, actions):
        latents = Encoder(self.net_params, self.spatial_softmax, name='encoder',
                          mode=self.mode)(images, proprioceptions)
        
        VmapCritic = nn.vmap(QFunction, variable_axes={'params': 0},
                             split_rngs={'params': True}, in_axes=None,
                             out_axes=0, axis_size=self.num_qs)
        
        qs = VmapCritic(self.net_params['mlp'])(latents, actions)
        
        return qs
    

class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            'log_temp', 
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)