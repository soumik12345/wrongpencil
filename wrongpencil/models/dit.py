"""
References:
    - https://arxiv.org/abs/2212.09748
    - https://github.com/facebookresearch/DiT/blob/main/models.py
    - https://github.com/kvfrans/jax-diffusion-transformer/blob/main/diffusion_transformer.py

"""

import math
from typing import Any, Callable

import jax
from einops import rearrange
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

    def create_sinusoidal_timestep_embedding(
        self, timestep: jnp.ndarray, max_period: int = 10000
    ) -> jnp.ndarray:
        """
        Creates sinusoidal embeddings for timesteps.

        Args:
            timestep (jnp.ndarray): Timesteps to embed.
            max_period (int): Maximum period for sinusoidal embedding.

        Returns:
            jnp.ndarray: Sinusoidal embeddings.
        """
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

    def __call__(self, timestep: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass for the timestep embedder.

        Args:
            timestep (jnp.ndarray): Timesteps to embed.

        Returns:
            jnp.ndarray: Sinusoidal embeddings.
        """
        x = self.create_sinusoidal_timestep_embedding(timestep)
        x = self.dense_1(x)
        x = nnx.silu(x)
        x = self.dense_2(x)
        return x


class LabelEmbedder(nnx.Module):
    """
    Embeds class labels into the vector representation.

    Args:
        dropout_probas (float): Dropout probability for class labels.
        num_classes (int): Number of class labels.
        hidden_size (int): Hidden dimension.
        rngs (nnx.Rngs): Rng keys.
    """

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

    def token_drop(
        self, labels: jnp.ndarray, force_drop_ids: int | None = None
    ) -> jnp.ndarray:
        """
        Drops tokens based on the dropout probability.

        Args:
            labels (jnp.ndarray): Labels to drop.
            force_drop_ids (int | None): Force drop ids.

        Returns:
            jnp.ndarray: Labels after dropping.
        """
        drop_ids = (
            jax.random.bernoulli(
                self.rngs.dropout(), self.dropout_probas, (labels.shape[0],)
            )
            if force_drop_ids is None
            else force_drop_ids == 1
        )
        labels = jnp.where(drop_ids, self.num_classes, labels)
        return labels

    def __call__(
        self,
        labels: jnp.ndarray,
        train: bool = False,
        force_drop_ids: int | None = None,
    ) -> jnp.ndarray:
        """
        Forward pass for the label embedder.

        Args:
            labels (jnp.ndarray): Labels to embed.
            train (bool): Whether the model is in training mode.
            force_drop_ids (int | None): Force drop ids.

        Returns:
            jnp.ndarray: Embeddings.
        """
        if (train and self.dropout_probas > 0) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class MLPBlock(nnx.Module):
    """
    MLP block for the DIT model.

    Args:
        in_features (int): Input features.
        mlp_dimension (int): MLP dimension.
        output_dimesion (int | None): Output dimension.
        dropout_rate (float | None): Dropout rate.
        dtype (Any): Data type.
        kernel_init (Callable): Kernel initialization.
        bias_init (Callable): Bias initialization.
        rngs (nnx.Rngs): RNGs.
    """

    def __init__(
        self,
        in_features: int,
        mlp_dimension: int,
        output_dimesion: int | None = None,
        dropout_rate: float | None = None,
        dtype: Any = jnp.float32,
        kernel_init: Callable = nnx.initializers.xavier_normal(),
        bias_init: Callable = nnx.initializers.normal(stddev=1e-6),
        *,
        rngs: nnx.Rngs,
    ):
        self.dropout_rate = dropout_rate
        self.dense_1 = nnx.Linear(
            in_features=in_features,
            out_features=mlp_dimension,
            dtype=dtype,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs,
        )
        self.dense_2 = nnx.Linear(
            in_features=mlp_dimension,
            out_features=in_features if output_dimesion is None else output_dimesion,
            dtype=dtype,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs,
        )
        if dropout_rate is not None:
            self.dropout_1 = nnx.Dropout(rate=dropout_rate)
            self.dropout_2 = nnx.Dropout(rate=dropout_rate)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass for the MLP block.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor.
        """
        x = self.dense_1(x)
        x = nnx.gelu(x)
        if self.dropout_rate is not None:
            x = self.dropout_1(x)
        x = self.dense_2(x)
        if self.dropout_rate is not None:
            x = self.dropout_2(x)
        return x


class PatchEmbed(nnx.Module):
    """
    Patch embedding for the DIT model.

    Args:
        in_features (int): Input features.
        patch_size (int): Patch size.
        embedding_dimension (int): Embedding dimension.
        use_bias (bool): Whether to use bias.
        rngs (nnx.Rngs): RNGs.
    """

    def __init__(
        self,
        in_features: int,
        patch_size: int,
        embedding_dimension: int,
        use_bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_features = in_features
        self.patch_size = patch_size
        self.embedding_dimension = embedding_dimension
        self.use_bias = use_bias
        conv_kernel_init = nnx.initializers.xavier_uniform()
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=embedding_dimension,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            use_bias=use_bias,
            padding="VALID",
            kernel_init=conv_kernel_init,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass for the patch embedding module.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor.
        """
        _, height, width, _ = x.shape
        num_height_patches = height // self.patch_size
        num_width_patches = width // self.patch_size
        x = self.conv(x)
        x = rearrange(
            x, "b h w c -> b (h w) c", h=num_height_patches, w=num_width_patches
        )
        return x


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32) / (embed_dim / 2.0)
    omega = 1.0 / (10000**omega)  # (D/2,)
    out = jnp.einsum("m,d->md", pos.reshape(-1), omega)  # (M, D/2)
    return jnp.concatenate([jnp.sin(out), jnp.cos(out)], axis=1)  # (M, D)


def get_2d_sincos_pos_embed(embed_dim: int, num_patches: int) -> jnp.ndarray:
    """Returns a fixed 2D sinusoidal positional embedding of shape (num_patches, embed_dim).

    Args:
        embed_dim (int): Embedding dimension.
        num_patches (int): Number of patches.

    Returns:
        jnp.ndarray: 2D sinusoidal positional embedding of shape (num_patches, embed_dim).
    """
    grid_size = int(num_patches**0.5)
    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w, grid_h = jnp.meshgrid(grid_w, grid_h)  # each (grid_size, grid_size)
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)  # (N, D/2)
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)  # (N, D/2)
    return jnp.concatenate([emb_w, emb_h], axis=1)  # (N, D)


def modulate(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    return x * (1 + scale[:, None]) + shift[:, None]


class DITBlock(nnx.Module):
    """
    DIT block for the DIT model.

    Args:
        hidden_size (int): Hidden dimension.
        num_heads (int): Number of heads.
        mlp_ratio (float): MLP ratio.
        rngs (nnx.Rngs): RNGs.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = nnx.LayerNorm(
            num_features=hidden_size,
            use_bias=False,
            use_scale=False,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(
            num_features=hidden_size,
            use_bias=False,
            use_scale=False,
            rngs=rngs,
        )
        self.adaLN_modulation = nnx.Linear(
            in_features=hidden_size,
            out_features=6 * hidden_size,
            kernel_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            kernel_init=nnx.initializers.xavier_uniform(),
            rngs=rngs,
        )
        self.mlp = MLPBlock(
            in_features=hidden_size,
            mlp_dimension=int(hidden_size * mlp_ratio),
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass for the DIT block.

        Args:
            x (jnp.ndarray): Input tensor.
            c (jnp.ndarray): Conditioning tensor.

        Returns:
            jnp.ndarray: Output tensor.
        """
        c = nnx.silu(c)
        c = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            c, 6, axis=-1
        )

        x_norm = self.norm1(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)
        attn_x = self.attention(x_modulated)
        x = x + (gate_msa[:, None] * attn_x)

        x_norm2 = self.norm2(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_x = self.mlp(x_modulated2)
        x = x + (gate_mlp[:, None] * mlp_x)
        return x


class FinalLayer(nnx.Module):
    """
    Final layer for the DIT model.

    Args:
        patch_size (int): Patch size.
        out_channels (int): Output channels.
        hidden_size (int): Hidden dimension.
        rngs (nnx.Rngs): RNGs.
    """

    def __init__(
        self, patch_size: int, out_channels: int, hidden_size: int, *, rngs: nnx.Rngs
    ):
        dense_kernel_init = nnx.initializers.constant(0)
        self.dense = nnx.Linear(
            in_features=hidden_size,
            out_features=2 * hidden_size,
            kernel_init=dense_kernel_init,
            rngs=rngs,
        )
        self.adaLN_modulation_init = nnx.initializers.constant(0)
        self.adaLN_modulation = nnx.Linear(
            in_features=hidden_size,
            out_features=patch_size * patch_size * out_channels,
            kernel_init=self.adaLN_modulation_init,
            rngs=rngs,
        )
        self.layer_norm = nnx.LayerNorm(
            num_features=hidden_size, use_bias=False, use_scale=False
        )

    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass for the final layer.

        Args:
            x (jnp.ndarray): Input tensor.
            c (jnp.ndarray): Conditioning tensor.

        Returns:
            jnp.ndarray: Output tensor.
        """
        c = nnx.silu(c)
        c = self.dense(c)
        shift, scale = jnp.split(c, 2, axis=-1)

        x = self.layer_norm(x)
        x = modulate(x, shift, scale)
        return self.adaLN_modulation(x)


class DiffusionTransformer(nnx.Module):
    """Diffusion Transformer model as proposed by
    [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748).

    Args:
        input_size (int): Spatial resolution of the input image (height == width).
        patch_size (int): Size of each patch.
        in_channels (int): Number of input image channels.
        hidden_size (int): Transformer hidden dimension.
        depth (int): Number of DiT blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): MLP hidden-dim multiplier.
        class_dropout_prob (float): Dropout probability for class labels.
        num_classes (int): Number of class labels.
        learn_sigma (bool): If True, the model predicts both mean and variance.
        rngs (nnx.Rngs): Rng keys.
    """

    def __init__(
        self,
        input_size: int,
        patch_size: int,
        in_channels: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        class_dropout_prob: float,
        num_classes: int,
        learn_sigma: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.patch_size = patch_size
        self.learn_sigma = learn_sigma
        self.out_channels = in_channels * 2 if learn_sigma else in_channels

        num_patches = (input_size // patch_size) ** 2
        pos_embed_val = get_2d_sincos_pos_embed(hidden_size, num_patches)
        self.pos_embed = nnx.Variable(pos_embed_val)

        self.patch_embed = PatchEmbed(
            in_features=in_channels,
            patch_size=patch_size,
            embedding_dimension=hidden_size,
            rngs=rngs,
        )
        self.timestep_embedder = TimeStepEmbedder(hidden_size=hidden_size, rngs=rngs)
        self.label_embedder = LabelEmbedder(
            dropout_probas=class_dropout_prob,
            num_classes=num_classes,
            hidden_size=hidden_size,
            rngs=rngs,
        )
        self.dit_blocks = [
            DITBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                rngs=rngs,
            )
            for _ in range(depth)
        ]
        self.final_layer = FinalLayer(
            patch_size=patch_size,
            out_channels=self.out_channels,
            hidden_size=hidden_size,
            rngs=rngs,
        )

    def __call__(
        self, image, timestep, label, train: bool = False, force_drop_ids=None
    ):
        """
        Forward pass for the DIT model.

        Args:
            image (jnp.ndarray): Input image.
            timestep (jnp.ndarray): Timestep.
            label (jnp.ndarray): Label.
            train (bool): Whether the model is in training mode.
            force_drop_ids (int | None): Force drop ids.

        Returns:
            jnp.ndarray: Output tensor.
        """
        # image: (B, H, W, C), timestep: (B,), label: (B,)
        batch_size = image.shape[0]
        input_size = image.shape[1]
        num_patches_side = input_size // self.patch_size

        x = self.patch_embed(image)
        x = x + lax.stop_gradient(self.pos_embed.value)

        timestep_embedding = self.timestep_embedder(timestep)
        label_embedding = self.label_embedder(
            label, train=train, force_drop_ids=force_drop_ids
        )
        condition_embedding = timestep_embedding + label_embedding

        for block in self.dit_blocks:
            x = block(x, condition_embedding)

        x = self.final_layer(x, condition_embedding)

        x = jnp.reshape(
            x,
            (
                batch_size,
                num_patches_side,
                num_patches_side,
                self.patch_size,
                self.patch_size,
                self.out_channels,
            ),
        )
        x = jnp.einsum("bhwpqc->bhpwqc", x)
        x = rearrange(
            x,
            "B H P W Q C -> B (H P) (W Q) C",
            H=num_patches_side,
            W=num_patches_side,
        )
        return x
