"""Defines a multi-eye system that procedurally generates multiple eyes and manages
them."""

from typing import Any, Callable, Dict, Self, Tuple

import numpy as np
import torch
from gymnasium import spaces
from hydra_config import config_wrapper

from cambrian.eyes.eye import MjCambrianEye, MjCambrianEyeConfig
from cambrian.renderer.overlays import MjCambrianCursor, MjCambrianViewerOverlay
from cambrian.renderer.render_utils import generate_composite
from cambrian.utils import MjCambrianGeometry, generate_sequence_from_range
from cambrian.utils.cambrian_xml import MjCambrianXML
from cambrian.utils.types import ObsType


@config_wrapper
class MjCambrianMultiEyeConfig(MjCambrianEyeConfig):
    """Config for MjCambrianMultiEye.

    Inherits from MjCambrianEyeConfig and adds attributes for procedural eye placement.

    Attributes:
        instance (Callable[[Self, str], MjCambrianEye]): The class instance to use
            when creating the eye. Takes the config and the name of the eye as
            arguments.

        single_eye (MjCambrianEyeConfig): The config for a single eye. This is used as
            the base configuration for all eyes in the multi-eye system.

        lat_range (Optional[Tuple[float, float]]): The x range of the eye. This is
            used to determine the placement of the eye on the agent. Specified in
            degrees. This is the latitudinal/vertical range of the evenly placed eye
            about the agent's bounding sphere.
        lon_range (Optional[Tuple[float, float]]): The y range of the eye. This is
            used to determine the placement of the eye on the agent. Specified in
            degrees. This is the longitudinal/horizontal range of the evenly placed eye
            about the agent's bounding sphere.
        num_eyes (Optional[Tuple[int, int]]): The num of eyes to generate.
            If this is specified, then the eyes will be generated on a spherical
            grid. The first element is the number of eyes to generate latitudinally and
            the second element is the number of eyes to generate longitudinally. The
            eyes will be named sequentially starting from `eye_0`. Each eye will default
            to use the first eye config in the `eyes` attribute. `eyes` must have a
            length of 1 if this is specified. Each eye is named `eye_{lat}_{lon}` where
            `lat` is the latitude index and `lon` is the longitude index.

        flatten_observation (Optional[bool]): Whether to flatten the observation space
            of the multi-eye system. If True, the observation space will be a Box space
            with the shape `(num_eyes * eye_observation_space,)`. If False, the
            observation space will be a Dict space with the keys as the eye names and
            the values as the eye observation spaces. Defaults to False.
    """

    instance: Callable[[Self, str], "MjCambrianMultiEye"]

    single_eye: MjCambrianEyeConfig

    # private attribute used as a workaround to allow overriding eye attributes from
    # command line without adding single_eye in the argument.
    _single_eye: MjCambrianEyeConfig | Any

    lat_range: Tuple[float, float]
    lon_range: Tuple[float, float]
    num_eyes: Tuple[int, int]

    flatten_observations: bool


class MjCambrianMultiEye(MjCambrianEye):
    """Defines a multi-eye system that procedurally generates multiple eyes and manages
    them.

    Inherits from MjCambrianEye but manages multiple eyes.

    Args:
        config (MjCambrianMultiEyeConfig): Configuration for the multi-eye system.
        name (str): Base name for the eyes.
    """

    def __init__(
        self, config: MjCambrianMultiEyeConfig, name: str, disable_render: bool = True
    ):
        self._config: MjCambrianMultiEyeConfig = config
        self._name = name

        # Generate eyes procedurally
        self._eyes: Dict[str, MjCambrianEye] = {}
        self._place_eyes()

        super().__init__(config, name, disable_render=disable_render)

    def _place_eyes(self):
        """Place the eyes procedurally based on config."""
        nlat, nlon = self._config.num_eyes
        lat_bins = generate_sequence_from_range(self._config.lat_range, nlat)
        lon_bins = generate_sequence_from_range(self._config.lon_range, nlon)
        for lat_idx, lat in enumerate(lat_bins):
            for lon_idx, lon in enumerate(lon_bins):
                eye_name = f"{self._name}_{lat_idx}_{lon_idx}"
                eye_config = self._config._single_eye.copy()
                # Update the eye's coord to the current lat, lon
                eye_config.coord = [lat, lon]
                # Create the eye instance
                eye = eye_config.instance(eye_config, eye_name)
                self._eyes[eye_name] = eye

    def generate_xml(
        self, parent_xml: MjCambrianXML, geom: MjCambrianGeometry, parent_body_name: str
    ) -> MjCambrianXML:
        """Generate the XML for all eyes."""
        xml = super().generate_xml(parent_xml, geom, parent_body_name)
        for eye in self._eyes.values():
            eye_xml = eye.generate_xml(parent_xml, geom, parent_body_name)
            xml += eye_xml
        return xml

    def reset(self, *args) -> ObsType:
        """Reset all eyes."""
        obs = {}
        for name, eye in self._eyes.items():
            obs[name] = eye.reset(*args)

        super().reset(*args)

        return self._update_obs(obs)

    def step(self, obs: ObsType | None = None) -> ObsType:
        """Step all eyes and collect observations."""
        if obs is None:
            obs = {}
            for name, eye in self._eyes.items():
                obs[name] = eye.step()
        return self._update_obs(obs)

    def _update_obs(self, obs: ObsType) -> ObsType:
        """Update the observation space."""
        if self._config.flatten_observations:
            obs = torch.cat(list(obs.values()), dim=0)
        return obs

    def render(self) -> list[MjCambrianViewerOverlay]:
        """This is a debug method which renders the eye's as a composite image.

        Will appear as a compound eye. For example, if we have a 3x3 grid of eyes:
            TL T TR
            ML M MR
            BL B BR

        Each eye has a red border around it.
        """
        if self._config.num_eyes == 0:
            return []

        overlays = []

        # Sort the eyes based on their lat/lon
        images: Dict[float, Dict[float, torch.Tensor]] = {}
        for eye in self.eyes.values():
            lat, lon = eye.config.coord
            if lat not in images:
                images[lat] = {}
            assert lon not in images[lat], f"Duplicate eye at {lat}, {lon}."

            # Add the image to the dictionary
            images[lat][lon] = eye.prev_obs

        # Add text overlays
        create_text_overlay = MjCambrianViewerOverlay.create_text_overlay
        overlays.append(create_text_overlay(f"Num Eyes: {self.num_eyes}"))
        overlays.append(create_text_overlay(f"FOV: {self.config.fov}"))
        overlays.append(create_text_overlay(f"Resolution: {self.config.resolution}"))

        # Add image overlays
        position = MjCambrianCursor.Position.BOTTOM_LEFT
        layer = MjCambrianCursor.Layer.BACK
        cursor = MjCambrianCursor(position=position, x=0, y=0, layer=layer)
        image = generate_composite(images) * 255.0
        image_overlay = MjCambrianViewerOverlay.create_image_overlay(
            image, cursor=cursor
        )
        overlays.append(image_overlay)

        return overlays

    @property
    def observation_space(self) -> spaces.Space:
        """Constructs the observation space for the multi-eye."""
        if self._config.flatten_observations:
            shape = (
                self._config.resolution[0] * self.num_eyes,
                self._config.resolution[1],
                3,
            )
            observation_space = spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        else:
            observation_space = {}
            for name, eye in self._eyes.items():
                observation_space[name] = eye.observation_space
            observation_space = spaces.Dict(observation_space)
        return observation_space

    @property
    def prev_obs(self) -> ObsType:
        """The last observations from all eyes."""
        obs = {}
        for name, eye in self._eyes.items():
            obs[name] = eye.prev_obs
        return obs

    @property
    def eyes(self) -> Dict[str, MjCambrianEye]:
        """Returns the dictionary of eyes."""
        return self._eyes

    @property
    def name(self) -> str:
        """Returns the base name of the multi-eye system."""
        return self._name

    @property
    def num_eyes(self) -> int:
        """Returns the number of eyes."""
        return len(self._eyes)
