from typing import Iterator

import grain.python as grain
import jax
import tensorflow_datasets as tfds

from wrongpencil.data.transforms import (
    CenterCropAndResize,
    ExtractImageAndLabel,
    NormalizeImage,
    RandomFlipLeftRight,
)


def _get_target_size(dataset_name: str) -> int:
    """Returns the target image size for the given dataset name."""
    if "imagenet256" in dataset_name:
        return 256
    elif "imagenet128" in dataset_name:
        return 128
    else:
        raise ValueError(f"Unknown imagenet variant: {dataset_name}")


def get_source_and_transforms(
    dataset_name: str, is_train: bool, debug_overfit: bool = False
) -> tuple[grain.DataSource, list[grain.Transformation]]:
    if "imagenet" in dataset_name:
        split = "train" if (is_train or debug_overfit) else "validation"
        source = tfds.data_source("imagenet2012", split=split)
        target_size = _get_target_size(dataset_name)
        transforms: list[grain.Transformation] = [
            CenterCropAndResize(target_size),
        ]
        if is_train:
            transforms.append(RandomFlipLeftRight())
        transforms.append(NormalizeImage())
        transforms.append(ExtractImageAndLabel())
    elif dataset_name == "celebahq256":
        split = "train"
        source = tfds.data_source("celebahq256", split=split)
        transforms = [
            NormalizeImage(),
            ExtractImageAndLabel(),
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return source, transforms


def get_dataset(
    dataset_name: str,
    batch_size: int,
    is_train: bool,
    debug_overfit: bool = False,
    seed: int = 42,
    worker_count: int = 4,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Creates a grain-based data pipeline for JAX training.

    Args:
        dataset_name: Name of the dataset. Supports 'imagenet256', 'imagenet128',
            and 'celebahq256'.
        batch_size: Per-host batch size (total batch size divided by the number
            of hosts).
        is_train: Whether this is a training dataset.
        debug_overfit: If True, use only 8 samples from the training split and
            repeat them (for debugging overfitting).
        seed: Random seed for shuffling and random augmentations.
        worker_count: Number of grain DataLoader workers. Set to 0 for
            single-process debugging.

    Returns:
        An iterator yielding (images, labels) tuples as numpy arrays, where
        images has shape (batch_size, H, W, C) as float32 in [-1, 1] and
        labels has shape (batch_size,).

    Raises:
        ValueError: If the dataset_name is not recognized.
    """
    # Determine split and load data source
    source, transforms = get_source_and_transforms(
        dataset_name, is_train, debug_overfit
    )

    num_records = len(source)

    if debug_overfit:
        # For debug overfitting: use only 8 records, repeat indefinitely
        num_records = 8
        sampler = grain.IndexSampler(
            num_records=num_records,
            num_epochs=None,  # repeat indefinitely
            shard_options=grain.ShardOptions(
                shard_index=jax.process_index(),
                shard_count=jax.process_count(),
                drop_remainder=True,
            ),
            shuffle=False,
            seed=seed,
        )
    else:
        sampler = grain.IndexSampler(
            num_records=num_records,
            num_epochs=None,  # repeat indefinitely
            shard_options=grain.ShardOptions(
                shard_index=jax.process_index(),
                shard_count=jax.process_count(),
                drop_remainder=True,
            ),
            shuffle=is_train,
            seed=seed,
        )

    transforms.append(grain.Batch(batch_size=batch_size, drop_remainder=True))

    dataloader = grain.DataLoader(
        data_source=source,
        operations=transforms,
        sampler=sampler,
        worker_count=worker_count,
    )

    return iter(dataloader)
