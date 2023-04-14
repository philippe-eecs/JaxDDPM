from typing import Callable, Optional, Sequence, Type

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import lecun_normal
import jax

default_init = nn.initializers.xavier_uniform


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size,
                             kernel_init=default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x

class Base2FourierFeatures(nn.Module):
    start: int = 0
    stop: int = 8
    step: int = 1

    @nn.compact
    def __call__(self, inputs):
      freqs = range(self.start, self.stop, self.step)

      # Create Base 2 Fourier features
      w = 2.**(jnp.asarray(freqs, dtype=inputs.dtype)) * 2 * jnp.pi
      w = jnp.tile(w[None, :], (1, inputs.shape[-1]))

      # Compute features
      h = jnp.repeat(inputs, len(freqs), axis=-1)
      h = w * h
      h = jnp.concatenate([jnp.sin(h), jnp.cos(h)], axis=-1)
      return h 

class FourierFeatures(nn.Module):
    output_size: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param('kernel', nn.initializers.normal(0.2),
                           (self.output_size // 2, x.shape[-1]), jnp.float32)
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


class DiffusionTransformerMLP(nn.Module):
    time_encoder: Type[nn.Module]
    reverse_encoder: Type[nn.Module]
    time_preprocess: Type[nn.Module]

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 time: jnp.ndarray,
                 training: bool = False):
        t = self.time_preprocess(time)
        post_t = self.time_encoder(t, training=training)

        post_t = post_t[..., jnp.newaxis, :]
        post_t = jnp.repeat(post_t, axis=-2, repeats=x.shape[-1])
        x = x[..., jnp.newaxis]

        reverse_input = jnp.concatenate([x, post_t], axis = -1)

        return self.reverse_encoder(reverse_input, training=training)

class DiffusionMLP(nn.Module):
    time_encoder: Type[nn.Module]
    reverse_encoder: Type[nn.Module]
    preprocess_ff: Type[nn.Module]
    use_one_hot: bool = False
    use_ff_features: bool = False
    T: int = 1000


    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 time: jnp.ndarray,
                 training: bool = False):
        if self.use_one_hot:
            time = time[..., 0]
            post_t = jax.nn.one_hot(time, self.T)
        else:
            t = self.preprocess_ff(time)
            post_t = self.time_encoder(t, training=training)
        if len(x.shape) == 3:
            post_t = jnp.expand_dims(post_t, axis=0).repeat(x.shape[0], axis=0)

        if self.use_ff_features:
            x_ff = self.preprocess_ff(x)
            reverse_input = jnp.concatenate([x, x_ff, post_t], axis = -1)
        else:
            reverse_input = jnp.concatenate([x, post_t], axis = -1)

        return self.reverse_encoder(reverse_input, training=training)


# Ensures that it's not used:
"""
def sinusoidalposemb(x, dim):
    half_dim = dim // 2
    emb = np.log(10000) / (half_dim)
    emb = np.exp(np.arange(half_dim) * -emb)
    emb = x[:, None] * emb[None, :]
    emb = np.concatenate((np.sin(emb), np.cos(emb)), axis=-1)
    return emb
"""
