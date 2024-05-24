from jsac.helpers.utils import MODE
from typing import Optional, Sequence, Any
import flax.linen as nn
import jax 
import jax.numpy as jnp
from jax import random, vmap 
import functools  
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from jsac.algo.resnet import ResNet1, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet200 


def default_init(scale: Optional[float] = jnp.sqrt(2), dtype: Any = jnp.float32):
    return nn.initializers.orthogonal(scale, dtype=dtype)


@functools.partial(jax.jit, static_argnames=('image_shape'))
def augment(image, start_h, start_w, image_shape):
    return jax.lax.dynamic_slice(image,
                                 (start_h, start_w, 0), 
                                 image_shape)


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
    rad_offset: float = 0.01
    image_model: str = 'conv'
    mode: str = MODE.IMG_PROP
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, 
                 keys,
                 images, 
                 proprioceptions, 
                 training=False,
                 apply_rad=False,
                 stop_gradient=False):          
        
        proprioceptions = jnp.array(proprioceptions, dtype=self.dtype)
        
        if self.mode == MODE.PROP:
            return proprioceptions
        
        batch_size, height, width, channel = images.shape

        rad_h = max(round(self.rad_offset * height), 1)
        rad_w = max(round(self.rad_offset * width), 1)
        rad_image_shape = ((height - (2 * rad_h)), 
                           (width - (2 * rad_w)), 
                           channel)   
        get_augments = vmap(augment, in_axes=(0, 0, 0, None))
            
        if not apply_rad:
            # Still need to crop the images
            crop_height = jnp.ones((batch_size,), dtype=jnp.int32) * rad_h
            crop_width = jnp.ones((batch_size,), dtype=jnp.int32) * rad_w
        else:
            crop_height = random.randint(keys[0], (batch_size,), 0, rad_h+1, jnp.int32)
            crop_width = random.randint(keys[1], (batch_size,), 0, rad_w+1, jnp.int32)

        images = get_augments(images,
                              crop_height,
                              crop_width,
                              rad_image_shape)

        x = images.astype(self.dtype)
        x = (x - 127.5) / 127.5
        
        if self.image_model == 'conv':
            conv_params = self.net_params['conv']
            for i, (_, out_channel, kernel_size, stride) in enumerate(conv_params):
                layer_name = 'encoder_conv_' + str(i)

                x = nn.Conv(features=out_channel, 
                            kernel_size=(kernel_size, kernel_size),
                            strides=stride,
                            padding=0,  
                            kernel_init=nn.initializers
                            .delta_orthogonal(dtype=self.dtype), 
                            name=layer_name 
                )(x)

                if i < len(conv_params) - 1:
                    x = nn.relu(x)

            b, height, width, channel = x.shape
            x = SpatialSoftmax(width, height, channel, name='encoder_spatialsoftmax', 
                                dtype=self.dtype)(x)
            
        elif self.image_model.startswith('resnet'):
            resnet_model_no = int(self.image_model[6:])
            if resnet_model_no == 1:
                rn_class = ResNet1
            elif resnet_model_no == 18:
                rn_class = ResNet18
            elif resnet_model_no == 34:
                rn_class = ResNet34
            elif resnet_model_no == 50:
                rn_class = ResNet50
            elif resnet_model_no == 101:
                rn_class = ResNet101
            elif resnet_model_no == 152:
                rn_class = ResNet152
            elif resnet_model_no == 200:
                rn_class = ResNet200
            
            resnet_model = rn_class(num_classes=self.net_params['latent'], 
                                    dtype=self.dtype,
                                    name='encoder_resnet')
            x = resnet_model(x, training)
            
        if stop_gradient:
            x = jax.lax.stop_gradient(x)

        if self.mode == MODE.IMG_PROP:
           x = jnp.concatenate(axis = -1, arrays=(x, proprioceptions)) 

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
LOG_STD_MAX = 10.0


class ActorModel(nn.Module):
    net_params: dict 
    action_dim: int
    rad_offset: float = 0.01
    image_model: str = 'conv'
    mode: str = MODE.IMG_PROP
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, 
                 keys, 
                 images, 
                 proprioceptions, 
                 training=False, 
                 apply_rad=False):

        latents = Encoder(self.net_params, 
                          self.rad_offset,
                          self.image_model,
                          self.mode,
                          self.dtype,
                          name='encoder')(keys[1:],
                                          images, 
                                          proprioceptions, 
                                          training, 
                                          apply_rad,
                                          True)
        
        outputs = MLP(self.net_params['mlp'], activate_final=True, dtype=self.dtype)(latents)
        init = nn.initializers.zeros_init()
        mu = nn.Dense(self.action_dim, kernel_init=init, dtype=self.dtype)(outputs)
        log_std = nn.Dense(self.action_dim, kernel_init=init, dtype=self.dtype)(outputs)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)

        ## From https://github.com/ikostrikov/jaxrl
        mu = nn.tanh(mu)
        base_dist = tfd.MultivariateNormalDiag(loc=mu,
                                               scale_diag=jnp.exp(log_std))

        dist = tfd.TransformedDistribution(distribution=base_dist,
                                               bijector=tfb.Tanh())
        pi = dist.sample(seed=keys[0])
        log_pi = dist.log_prob(pi)
        
        return mu, pi, log_pi, log_std
    
    def __hash__(self): 
        return id(self)


class QFunction(nn.Module):
    hidden_dims: Sequence[int]
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, latents, actions):
        inputs = jnp.concatenate([latents, actions], -1)
        critic = MLP((*self.hidden_dims, 1), dtype=self.dtype)(inputs)
        return jnp.squeeze(critic, -1)


class CriticModel(nn.Module):
    net_params: dict  
    action_dim: int
    rad_offset: float = 0.01
    image_model: str = 'conv'
    mode: str = MODE.IMG_PROP
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, 
                 keys,
                 images, 
                 proprioceptions, 
                 actions,
                 training=False,  
                 apply_rad=False):
        
        latents = Encoder(self.net_params, 
                          self.rad_offset,
                          self.image_model,
                          self.mode,
                          self.dtype,
                          name='encoder')(keys,
                                          images, 
                                          proprioceptions,
                                          training, 
                                          apply_rad)     
        
        VmapCritic = nn.vmap(
            QFunction,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=2,
        )
        qs = VmapCritic(self.net_params['mlp'])(latents, actions)
        
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