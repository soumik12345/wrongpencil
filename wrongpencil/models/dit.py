import math

import jax
from flax import nnx
from jax import lax, numpy as jnp


class TimeStepEmbedder(nnx.Module):
    """Embeds scalar timesteps into vector representations.

    Args:
        hidden_size (int): Embedding dimension.
        frequency_embedding_size (int): Size of the raw sinusoidal features.
        rngs (nnx.Rngs): Rng key.
    """

    def __init__(
        self, hidden_size: int, frequency_embedding_size: int = 256, *, rngs: nnx.Rngs
    ):
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        dense_1_kernel_init = nnx.initializers.normal(stddev=0.02)
        self.dense_1 = nnx.Linear(
            in_features=frequency_embedding_size,
            out_features=hidden_size,
            kernel_init=dense_1_kernel_init,
            rngs=rngs,
        )
        dense_2_kernel_init = nnx.initializers.normal(stddev=0.02)
        self.dense_2 = nnx.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            kernel_init=dense_2_kernel_init,
            rngs=rngs,
        )

    def create_sinusoidal_timestep_embedding(self, timestep, max_period: int = 10000):
        timestep = lax.convert_element_type(timestep, jnp.float32)
        half_dim = self.frequency_embedding_size // 2
        frequencies = jnp.exp(
            -math.log(max_period)
            * jnp.arange(start=0, stop=half_dim, dtype=jnp.float32)
            / half_dim
        )
        args = timestep[:, None] * frequencies[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding

    def __call__(self, timestep):
        x = self.create_sinusoidal_timestep_embedding(timestep)
        x = self.dense_1(x)
        x = nnx.silu(x)
        x = self.dense_2(x)
        return x


class LabelEmbedder(nnx.Module):
    def __init__(
        self,
        dropout_probas: float,
        num_classes: int,
        hidden_size: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.dropout_probas = dropout_probas
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.rngs = rngs
        embedding_init = nnx.initializers.normal(0.02)
        self.embedding_table = nnx.Embed(
            num_embeddings=num_classes + 1,
            embedding_init=embedding_init,
            features=hidden_size,
            rngs=self.rngs,
        )

    def token_drop(self, labels, force_drop_ids: int | None = None):
        drop_ids = (
            jax.random.bernoulli(self.rngs, self.dropout_probas, (labels.shape[0],))
            if force_drop_ids is None
            else force_drop_ids == 1
        )
        labels = jnp.where(drop_ids, self.num_classes, labels)
        return labels

    def __call__(self, labels, train: bool = False, force_drop_ids: int | None = None):
        if (train and self.dropout_probas > 0) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
