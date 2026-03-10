"""This module contains custom feature extractors for use in the models."""

from typing import Dict, List

import gymnasium as gym
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import TensorDict

# ==================
# Utils


def is_image_space(
    observation_space: gym.Space,
    check_channels: bool = False,
    normalized_image: bool = False,
) -> bool:
    """This is an extension of the sb3 is_image_space to support both regular images
    (HxWxC) and images with an additional dimension (NxHxWxC)."""
    from stable_baselines3.common.preprocessing import (
        is_image_space as sb3_is_image_space,
    )

    return len(observation_space.shape) == 4 or sb3_is_image_space(
        observation_space, normalized_image=normalized_image
    )


def maybe_transpose_space(observation_space: spaces.Box) -> spaces.Box:
    """This is an extension of the sb3 maybe_transpose_space to support both regular
    images (HxWxC) and images with an additional dimension (NxHxWxC). sb3 will call
    maybe_transpose_space on the 3D case, but not the 4D."""

    if len(observation_space.shape) == 4:
        num, height, width, channels = observation_space.shape
        new_shape = (num, channels, height, width)
        observation_space = spaces.Box(
            low=observation_space.low.reshape(new_shape),
            high=observation_space.high.reshape(new_shape),
            dtype=observation_space.dtype,
        )
    return observation_space


def maybe_transpose_obs(observation: torch.Tensor) -> torch.Tensor:
    """This is an extension of the sb3 maybe_transpose_obs to support both regular
    images (HxWxC) and images with an additional dimension (NxHxWxC). sb3 will call
    maybe_transpose_obs on the 3D case, but not the 4D.

    Note:
        In this case, there is a batch dimension, so the observation is 5D.
    """

    if len(observation.shape) == 5:
        observation = observation.permute(0, 1, 4, 2, 3)  # [B, T, N, H, W]

    return observation


# ==================
# Feature Extractors


class MjCambrianCombinedExtractor(BaseFeaturesExtractor):
    """Overwrite of the default feature extractor of Stable Baselines 3."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        normalized_image: bool,
        image_extractor: BaseFeaturesExtractor,
        share_image_extractor: bool = False,
    ) -> None:
        # We do not know features-dim here before going over all the items, so put
        # something there.
        super().__init__(observation_space, features_dim=1)

        self._image_extractor = None
        if share_image_extractor:
            # Verify all the image spaces have the same shape
            image_space = None
            for subspace in observation_space.values():
                if is_image_space(subspace, normalized_image=normalized_image):
                    subspace = maybe_transpose_space(subspace)
                    if image_space is None:
                        image_space = subspace
                    assert image_space.shape == subspace.shape, (
                        "All the image spaces must have the same shape if "
                        + "using shared image extractor"
                    )
            assert image_space is not None, "There must be at least one image space"
            self._image_extractor = image_extractor(image_space)

        extractors: Dict[str, BaseFeaturesExtractor] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                subspace = maybe_transpose_space(subspace)
                if share_image_extractor:
                    extractors[key] = self._image_extractor
                else:
                    extractors[key] = image_extractor(subspace)
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = FlattenExtractor(subspace)
            total_concat_size += extractors[key].features_dim

        self.extractors = torch.nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            obs = maybe_transpose_obs(observations[key])
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)


class PermutedFlattenExtractor(FlattenExtractor):
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        flattened = super().forward(observations)
        perm = torch.randperm(flattened.size(-1))
        return flattened[:, perm]


class MjCambrianImageFeaturesExtractor(BaseFeaturesExtractor):
    """This is a feature extractor for images. Will implement an image queue for
    temporal features. Should be inherited by other classes."""

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int,
        activation: torch.nn.Module,
    ):
        super().__init__(observation_space, features_dim)

        self._queue_size = 1
        if len(observation_space.shape) == 4:
            self._queue_size = observation_space.shape[0]
        height, width, n_channels = observation_space.shape[-3:]
        self._num_pixels = n_channels * height * width

        self.temporal_linear = torch.nn.Sequential(
            torch.nn.Linear(features_dim * self._queue_size, features_dim),
            activation(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.temporal_linear(observations)


class MjCambrianMLPExtractor(MjCambrianImageFeaturesExtractor):
    """MLP feature extractor for small images. Essentially NatureCNN but with MLPs."""

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int,
        activation: torch.nn.Module,
        architecture: List[int],
    ) -> None:
        super().__init__(observation_space, features_dim, activation)

        layers = []
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(self._num_pixels, architecture[0]))
        layers.append(activation())
        for i in range(1, len(architecture)):
            layers.append(torch.nn.Linear(architecture[i - 1], architecture[i]))
            layers.append(activation())
        layers.append(torch.nn.Linear(architecture[-1], features_dim))
        layers.append(activation())
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]

        observations = observations.reshape(-1, self._num_pixels)  # [B, C * H * W]
        encodings = self.mlp(observations)
        encodings = encodings.reshape(B, -1)

        return super().forward(encodings)


class MjCambrianNatureCNNExtractor(MjCambrianImageFeaturesExtractor):
    """Nature CNN feature extractor for images. This is the default feature extractor
    for stable baseline3 images. The main differences between this and the original
    is that this supports temporal features (i.e. image stacks) and dynamically
    calculates the kernel sizes and strides. In sb3, the fixed kernel sizes and strides
    restricted the image size to be > 36x36, which is a bd assumption here."""

    def __init__(
        self, observation_space: gym.Space, features_dim, activation: torch.nn.Module
    ):
        super().__init__(observation_space, features_dim, activation)

        n_channels = observation_space.shape[1]
        width, height = observation_space.shape[2], observation_space.shape[3]

        # Dynamically calculate kernel sizes and strides
        k_sizes, strides = self.calculate_dynamic_params(width, height)

        # Create CNN layers
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(n_channels, 32, kernel_size=k_sizes[0], stride=strides[0]),
            activation(),
            torch.nn.Conv2d(32, 64, kernel_size=k_sizes[1], stride=strides[1]),
            activation(),
            torch.nn.Conv2d(64, 64, kernel_size=k_sizes[2], stride=strides[2]),
            activation(),
            torch.nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample())
            n_flatten = self.cnn(sample)
            n_flatten = n_flatten.shape[0] * n_flatten.shape[1]

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, self._queue_size * features_dim), activation()
        )

    def calculate_dynamic_params(self, width, height):
        # Define max sizes and strides (from sb3, i.e. if width x height > 36x36, it's
        # the same).
        max_kernel_sizes = [8, 4, 2]
        max_strides = [4, 2, 1]

        # Adjust kernel sizes and strides based on input dimensions
        kernel_sizes = [min(k, height, width) for k in max_kernel_sizes]
        strides = [
            min(s, height // k, width // k) for s, k in zip(max_strides, kernel_sizes)
        ]

        return kernel_sizes, strides

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]
        observations = observations.reshape(-1, *observations.shape[2:])
        observations = self.cnn(observations)
        observations = observations.reshape(B, -1)
        return super().forward(self.linear(observations))
