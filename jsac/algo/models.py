import jax 
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Sequence, Any
from tensorflow_probability.substrates import jax as tfp

from jsac.helpers.utils import MODE

tfd = tfp.distributions
tfb = tfp.bijectors


def default_init(scale=jnp.sqrt(2), dtype=jnp.float32):
    return nn.initializers.orthogonal(scale, dtype=dtype)


class SpatialSoftmax(nn.Module):
    height: float
    width: float
    channel: float
    dtype: Any = jnp.float32

    def setup(self):
      pos_x, pos_y = jnp.meshgrid(
         jnp.linspace(-1., 1., self.height, dtype=self.dtype),
         jnp.linspace(-1., 1., self.width, dtype=self.dtype)
      )
      self._pos_x = pos_x.reshape(self.height*self.width)
      self._pos_y = pos_y.reshape(self.height*self.width)

    @nn.compact
    def __call__(self, feature):  
        feature = feature.transpose(0, 3, 1, 2)
        feature = feature.reshape(-1, self.height*self.width)
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
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, images):          
        conv_params = self.net_params['conv']
        x = images.astype(self.dtype)
        x = (x / 255.0) - 0.5 

        for i, (_, out_channel, kernel_size, stride) in enumerate(conv_params):
            layer_name = 'encoder_conv_' + str(i)
            x = nn.Conv(features=out_channel, 
                        kernel_size=(kernel_size, kernel_size),
                        strides=stride,
                        padding=0,  
                        kernel_init=default_init(dtype=self.dtype), 
                        param_dtype=self.dtype,
                        name=layer_name)(x) 
            if i < len(conv_params) - 1:
                x = nn.leaky_relu(x)
                
        b, height, width, channel = x.shape
        if self.spatial_softmax:
            x = SpatialSoftmax(width, height, channel, self.dtype,
                               name='encoder_spatialsoftmax')(x)
        else: 
            x = jnp.reshape(x, (b, -1))   
        
        return x


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activate_final: int = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init(dtype=self.dtype))(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = nn.relu(x)
        return x


LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0
        
class ActorModel(nn.Module):
    net_params: dict 
    action_dim: int
    spatial_softmax: bool = True
    mode: str = MODE.IMG_PROP
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, 
                 key, 
                 images, 
                 proprioceptions, 
                 temperature=1.0):

        if self.mode == MODE.IMG or self.mode == MODE.IMG_PROP:
            x = Encoder(self.net_params, 
                        self.spatial_softmax, 
                        self.mode, 
                        self.dtype, 
                        name='encoder')(images)
            x = jax.lax.stop_gradient(x)
            x = nn.Dense(self.net_params['latent_dim'], 
                         kernel_init=default_init(dtype=self.dtype))(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
            if self.mode == MODE.IMG_PROP:
                proprioceptions = jnp.clip(proprioceptions, -10, 10)
                x = jnp.concatenate(axis = -1, arrays=(x, proprioceptions))
        else:
            x = proprioceptions 

        outputs = MLP(self.net_params['mlp'], activate_final=True, dtype=self.dtype)(x)
        init = default_init(1.0, self.dtype)
        mu = nn.Dense(self.action_dim, kernel_init=init, dtype=self.dtype)(outputs)
        log_std = nn.Dense(self.action_dim, kernel_init=init, dtype=self.dtype)(outputs)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)

        base_dist = tfd.MultivariateNormalDiag(loc=mu,
                                               scale_diag=jnp.exp(log_std)* 
                                               temperature)

        dist = tfd.TransformedDistribution(distribution=base_dist,
                                               bijector=tfb.Tanh())
        pi = dist.sample(seed=key)
        log_pi = dist.log_prob(pi)
        
        return pi, log_pi
    
    def __hash__(self): 
        return id(self)


class QFunction(nn.Module):
    hidden_dims: Sequence[int]
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, latents, actions):
        inputs = jnp.concatenate([latents, actions], -1)
        outputs = MLP(self.hidden_dims, activate_final=True, dtype=self.dtype)(inputs)
        critic = nn.Dense(1, kernel_init=default_init(1.0, self.dtype), dtype=self.dtype)(outputs)
        return jnp.squeeze(critic, -1), outputs


class CriticModel(nn.Module):
    net_params: dict  
    action_dim: int
    spatial_softmax: bool = True
    mode: str = MODE.IMG_PROP
    dtype: Any = jnp.float32
    num_critic_networks: int = 10

    @nn.compact
    def __call__(self, images, proprioceptions, actions):
        if self.mode == MODE.IMG or self.mode == MODE.IMG_PROP:
            x = Encoder(self.net_params, 
                            self.spatial_softmax,
                            self.mode,
                            self.dtype,
                            name='encoder')(images)
            x = nn.Dense(self.net_params['latent_dim'], 
                         kernel_init=default_init(dtype=self.dtype))(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
            if self.mode == MODE.IMG_PROP:
                proprioceptions = jnp.clip(proprioceptions, -10, 10)
                x = jnp.concatenate(axis = -1, arrays=(x, proprioceptions))
        else:
            x = proprioceptions 

        VmapCritic = nn.vmap(
            QFunction,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_critic_networks,
        )
        qs, latents = VmapCritic(
            self.net_params['mlp'],
            self.dtype,
        )(
            x,
            actions,
        )
        self.sow("intermediates", "latent", latents)
        
        return qs 
    

class Temperature(nn.Module):
    initial_temperature: float = 1.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            'log_temp', 
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature), dtype=self.dtype))
        return jnp.exp(log_temp)