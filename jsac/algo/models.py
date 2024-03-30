from jsac.helpers.utils import MODE
from typing import Optional, Sequence, Any
import flax
import flax.linen as nn
import jax 
import jax.numpy as jnp
from jax import random, vmap 
import functools 
import einops  
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


def default_init(scale: Optional[float] = jnp.sqrt(2), dtype: Any = jnp.bfloat16):
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
    dtype: Any = jnp.bfloat16

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
    

### MIXER MODEL: https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_mixer.py

class MlpBlock(nn.Module):
    mlp_dim: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        y = nn.Dense(self.mlp_dim, param_dtype=self.dtype)(x)
        y = nn.gelu(y)
        return nn.Dense(x.shape[-1], param_dtype=self.dtype)(y)


class MixerBlock(nn.Module):
    """Mixer block layer."""

    tokens_mlp_dim: int
    channels_mlp_dim: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(param_dtype=self.dtype)(x)
        y = jnp.swapaxes(y, 1, 2)
        y = MlpBlock(self.tokens_mlp_dim, self.dtype, name="token_mixing")(y)
        y = jnp.swapaxes(y, 1, 2)
        x = x + y
        y = nn.LayerNorm(param_dtype=self.dtype)(x)
        return x + MlpBlock(self.channels_mlp_dim, self.dtype, name="channel_mixing")(y)


class MlpMixer(nn.Module):
    """Mixer architecture."""

    patch_size: Any
    num_classes: int
    num_blocks: int
    hidden_dim: int
    tokens_mlp_dim: int
    channels_mlp_dim: int
    dtype: Any = jnp.bfloat16
    model_name: Optional[str] = None

    @nn.compact
    def __call__(self, inputs):
        x = nn.Conv(
            self.hidden_dim,
            self.patch_size,
            strides=self.patch_size,
            name="stem",
            dtype=self.dtype,
        )(inputs)
        x = einops.rearrange(x, "n h w c -> n (h w) c")
        for _ in range(self.num_blocks):
            x = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim, self.dtype)(x)
        x = nn.LayerNorm(name="pre_head_layer_norm", dtype=self.dtype)(x)
        x = jnp.mean(x, axis=1)
        if self.num_classes:
            x = nn.Dense(
                self.num_classes,
                kernel_init=nn.initializers.zeros,
                name="head",
                dtype=self.dtype,
            )(x)
        return x


def mixer_model(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name
    """Factory function to easily create a Model variant like "L/16"."""

    if variant is not None:
        model_size, patch = variant.split("/")
        kw.setdefault("patch_size", (int(patch), int(patch)))
        config = {
            "T": {
                "hidden_dim": 384,
                "num_blocks": 6,
                "channels_mlp_dim": 1536,
                "tokens_mlp_dim": 192,
            },
            "S": {
                "hidden_dim": 512,
                "num_blocks": 8,
                "channels_mlp_dim": 2048,
                "tokens_mlp_dim": 256,
            },
            "B": {
                "hidden_dim": 768,
                "num_blocks": 12,
                "channels_mlp_dim": 3072,
                "tokens_mlp_dim": 384,
            },
            "L": {
                "hidden_dim": 1024,
                "num_blocks": 24,
                "channels_mlp_dim": 4096,
                "tokens_mlp_dim": 512,
            },
            "H": {
                "hidden_dim": 1280,
                "num_blocks": 32,
                "channels_mlp_dim": 5120,
                "tokens_mlp_dim": 640,
            },
            
        }[model_size]

        for k, v in config.items():
            kw.setdefault(k, v)

    return MlpMixer(num_classes=num_classes, **kw)


class Encoder(nn.Module):
    net_params: dict 
    rad_offset: float = 0.01
    mode: str = MODE.IMG_PROP
    vision_model: str = "B/16"
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, 
                 keys,
                 images, 
                 proprioceptions, 
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
        
        if self.vision_model == "spatial_softmax":
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
        else:
            x = mixer_model(num_classes=self.net_params['latent'], 
                            variant=self.vision_model,
                            dtype=self.dtype,
                            name='encoder_mixer')(x)
            
        if stop_gradient:
            x = jax.lax.stop_gradient(x)

        if self.mode == MODE.IMG_PROP:
           x = jnp.concatenate(axis = -1, arrays=(x, proprioceptions)) 

        return x


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activate_final: int = False
    dtype: Any = jnp.bfloat16

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
    mode: str = MODE.IMG_PROP
    vision_model: str = "B/16"
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, 
                 keys, 
                 images, 
                 proprioceptions, 
                 apply_rad=False):

        latents = Encoder(self.net_params, 
                          self.rad_offset,
                          self.mode,
                          self.vision_model,
                          self.dtype,
                          name='encoder')(keys[1:],
                                          images, 
                                          proprioceptions, 
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
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, latents, actions):
        inputs = jnp.concatenate([latents, actions], -1)
        critic = MLP((*self.hidden_dims, 1), dtype=self.dtype)(inputs)
        return jnp.squeeze(critic, -1)


class CriticModel(nn.Module):
    net_params: dict  
    action_dim: int
    rad_offset: float = 0.01
    mode: str = MODE.IMG_PROP
    vision_model: str = "B/16"
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, 
                 keys,
                 images, 
                 proprioceptions, 
                 actions,  
                 apply_rad=False):
        
        latents = Encoder(self.net_params, 
                          self.rad_offset,
                          self.mode,
                          self.vision_model,
                          self.dtype,
                          name='encoder')(keys,
                                          images, 
                                          proprioceptions, 
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
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            'log_temp', 
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature), dtype=self.dtype))
        return jnp.exp(log_temp)