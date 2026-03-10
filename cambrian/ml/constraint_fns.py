"""These are constraint functions for the optimizers. These functions are used to prune
experiments from the search space."""

from typing import Any, Dict, Tuple

import numpy as np

from cambrian.utils import is_number


def nevergrad_constraint_fn(
    parameterization: Dict[str, Any], /, *, fn: str, **parameters
) -> bool:
    """This function is used to prune experiments for nevergrad sweepers. It will
    return False if the experiment should be pruned."""
    from hydra.utils import get_method

    arguments: Dict[str, Any] = {}
    for argument_key, key_or_value in parameters.items():
        if isinstance(key_or_value, str) and key_or_value in parameterization:
            arguments[argument_key] = parameterization[key_or_value]
        else:
            arguments[argument_key] = key_or_value
    return get_method(fn)(**arguments)


def constrain_total_pixels(
    *,
    num_eyes_to_generate: Tuple[int, int] | int,
    resolution: Tuple[int, int] | int,
    max_num_pixels: int,
):
    """This constraint method will check whether the total number of pixels generated
    is less than a certain threshold."""
    if is_number(num_eyes_to_generate):
        num_eyes_to_generate = (1, num_eyes_to_generate)
    if is_number(resolution):
        resolution = (resolution, 1)
    pixels_per_eye = resolution[0] * resolution[1]
    number_of_eyes = num_eyes_to_generate[0] * num_eyes_to_generate[1]
    return pixels_per_eye * number_of_eyes <= max_num_pixels


def constrain_total_memory_throughput(
    *,
    num_eyes_to_generate: Tuple[int, int] | int,
    resolution: Tuple[int, int] | int,
    stack_size: int,
    max_pixels_in_memory: int,
):
    """This constraint method will check whether the total number of pixels generated
    is less than a certain threshold."""
    if is_number(num_eyes_to_generate):
        num_eyes_to_generate = (1, num_eyes_to_generate)
    if is_number(resolution):
        resolution = (resolution, 1)
    pixels_per_eye = resolution[0] * resolution[1]
    number_of_eyes = num_eyes_to_generate[0] * num_eyes_to_generate[1]
    return pixels_per_eye * number_of_eyes * stack_size <= max_pixels_in_memory


def constrain_morphologically_feasible_eyes(
    *,
    num_eyes_to_generate: int,
    resolution: Tuple[int, int] | int,
    lon_range: Tuple[int, int] | int,
    radius: float = 0.1,
    pixel_size: float = 5e-3,
    **_,
):
    """This constraint method will check whether the eye config, if placed
    num_eyes_to_generate along the longitude of the agent, would be
    morphologically feasible. Morphologically feasible in this approximated case is
    basically whether all the eyes would fit. There are two primary factors here:

    1. sensorsize and number of eyes. We want to make sure, along the horizontal axis,
    that the eyes don't overlap.

    2. The total number of pixels. We want to make sure that the total number of pixels
    generated is less than a certain threshold.

    Going to approximate the agent as a circle and the eyes as a line with a length
    equal to the sensorsize width. Then we'll check whether the eyes fit in the allowed
    longitude range.


    Args:
        num_eyes_to_generate (int): The number of eyes to generate along
            the longitude of the agent.
        resolution (Tuple[int, int] | int): The resolution of the eye.
        lon_range (Tuple[int, int] | int): The range of longitudes in which to generate
            the eyes. This is in degrees.

    Keyword Args:
        radius (float): The radius of the agent. Default is 0.2.
        pixel_size (float): The pixel size of the eye. This is used to calculate the
            total width of the eyes. Default is 0.01.
    """
    if is_number(num_eyes_to_generate):
        num_eyes_to_generate = (1, num_eyes_to_generate)
    if is_number(resolution):
        resolution = (resolution, 1)
    if is_number(lon_range):
        lon_range = (-abs(lon_range), abs(lon_range))

    # Total width of each eye
    sensor_width = resolution[0] * pixel_size
    total_width = sensor_width * num_eyes_to_generate[1]

    # Check whether the total width is less than the circumference of the agent
    # Only checked in the lon range
    lon_circumference = (lon_range[1] - lon_range[0]) * np.pi / 180 * radius
    lon_feasibility = total_width < lon_circumference

    return lon_feasibility


def constrain_total_num_eyes(
    *,
    num_eyes_to_generate: Tuple[int, int],
    max_num_eyes: int,
):
    """This constraint method will check whether the total number of eyes generated
    is less than a certain threshold."""
    return num_eyes_to_generate[0] * num_eyes_to_generate[1] <= max_num_eyes


def _resample_profile(values: Tuple[float, ...] | list[float], length: int) -> np.ndarray:
    profile = np.asarray(values, dtype=float)
    if profile.size == 0:
        return np.zeros(length, dtype=float)
    if profile.size == 1:
        return np.repeat(profile.item(), length)

    src = np.linspace(0.0, 1.0, profile.size)
    dst = np.linspace(0.0, 1.0, length)
    return np.interp(dst, src, profile)


def build_height_profile(
    *,
    height_map: Tuple[float, ...] | list[float] = (),
    height_profile_ctrl: Tuple[float, ...] | list[float] = (),
    height_profile_poly: Tuple[float, ...] | list[float] = (),
    height_profile_mix: float = 0.5,
    height_profile_ctrl_0: float | None = None,
    height_profile_ctrl_1: float | None = None,
    height_profile_ctrl_2: float | None = None,
    height_profile_ctrl_3: float | None = None,
    height_profile_poly_c2: float | None = None,
    height_profile_poly_c4: float | None = None,
    height_profile_poly_c6: float | None = None,
    length: int = 64,
) -> np.ndarray:
    """Reconstruct the normalized radial lens profile used by the optics module."""
    scalar_ctrl = [
        value
        for value in (
            height_profile_ctrl_0,
            height_profile_ctrl_1,
            height_profile_ctrl_2,
            height_profile_ctrl_3,
        )
        if value is not None
    ]
    scalar_poly = [
        value
        for value in (
            height_profile_poly_c2,
            height_profile_poly_c4,
            height_profile_poly_c6,
        )
        if value is not None
    ]
    if scalar_ctrl:
        height_profile_ctrl = scalar_ctrl
    if scalar_poly:
        height_profile_poly = scalar_poly

    base = _resample_profile(height_map, length) if len(height_map) else np.full(length, 0.5)
    ctrl = (
        _resample_profile(height_profile_ctrl, length)
        if len(height_profile_ctrl)
        else base.copy()
    )

    r = np.linspace(0.0, 1.0, length)
    poly = np.ones(length, dtype=float)
    for index, coefficient in enumerate(height_profile_poly, start=1):
        poly -= float(coefficient) * np.power(r, 2 * index)
    poly = np.clip(poly, 0.0, 1.0)

    mix = float(np.clip(height_profile_mix, 0.0, 1.0))
    profile = (1.0 - mix) * poly + mix * ctrl

    if not len(height_profile_ctrl) and not np.any(np.abs(height_profile_poly) > 0):
        profile = base

    return np.clip(profile, 0.0, 1.0)


def build_grin_profile(
    *,
    refractive_index: float,
    refractive_index_edge: float | None = None,
    grin_profile_ctrl: Tuple[float, ...] | list[float] = (),
    grin_profile_ctrl_0: float | None = None,
    grin_profile_ctrl_1: float | None = None,
    grin_profile_ctrl_2: float | None = None,
    grin_profile_ctrl_3: float | None = None,
    length: int = 64,
) -> np.ndarray:
    """Reconstruct the radial GRIN profile used by the optics module."""
    scalar_ctrl = [
        value
        for value in (
            grin_profile_ctrl_0,
            grin_profile_ctrl_1,
            grin_profile_ctrl_2,
            grin_profile_ctrl_3,
        )
        if value is not None
    ]
    if scalar_ctrl:
        grin_profile_ctrl = scalar_ctrl

    if len(grin_profile_ctrl) and np.any(
        np.abs(np.asarray(grin_profile_ctrl, dtype=float) - refractive_index) > 1e-9
    ):
        return np.maximum(_resample_profile(grin_profile_ctrl, length), 1.0)

    edge = refractive_index if refractive_index_edge is None else refractive_index_edge
    r = np.linspace(0.0, 1.0, length)
    profile = edge + (refractive_index - edge) * (1.0 - np.square(r))
    return np.maximum(profile, 1.0)


def constrain_monotonic_nonincreasing(
    *,
    profile: Tuple[float, ...] | list[float],
    tolerance: float = 1e-6,
) -> bool:
    """Require the radial profile to flatten or decrease from center to edge."""
    values = np.asarray(profile, dtype=float)
    if values.size < 2:
        return True
    return bool(np.all(np.diff(values) <= tolerance))


def constrain_profile_smoothness(
    *,
    profile: Tuple[float, ...] | list[float],
    max_second_difference: float,
) -> bool:
    """Bound roughness to avoid highly oscillatory lens surfaces."""
    values = np.asarray(profile, dtype=float)
    if values.size < 3:
        return True
    return bool(np.max(np.abs(np.diff(values, n=2))) <= max_second_difference)


def constrain_center_dominant_profile(
    *,
    profile: Tuple[float, ...] | list[float],
    min_center_minus_edge: float = 0.0,
) -> bool:
    """Require the lens center to be at least as strong as the edge."""
    values = np.asarray(profile, dtype=float)
    if values.size == 0:
        return True
    return bool(values[0] - values[-1] >= min_center_minus_edge)


def constrain_height_profile(
    *,
    height_map: Tuple[float, ...] | list[float] = (),
    height_profile_ctrl: Tuple[float, ...] | list[float] = (),
    height_profile_poly: Tuple[float, ...] | list[float] = (),
    height_profile_mix: float = 0.5,
    height_profile_ctrl_0: float | None = None,
    height_profile_ctrl_1: float | None = None,
    height_profile_ctrl_2: float | None = None,
    height_profile_ctrl_3: float | None = None,
    height_profile_poly_c2: float | None = None,
    height_profile_poly_c4: float | None = None,
    height_profile_poly_c6: float | None = None,
    max_second_difference: float = 0.075,
    min_center_minus_edge: float = 0.0,
    length: int = 64,
) -> bool:
    """Validate a radial lens profile reconstructed from optics parameters."""
    profile = build_height_profile(
        height_map=height_map,
        height_profile_ctrl=height_profile_ctrl,
        height_profile_poly=height_profile_poly,
        height_profile_mix=height_profile_mix,
        height_profile_ctrl_0=height_profile_ctrl_0,
        height_profile_ctrl_1=height_profile_ctrl_1,
        height_profile_ctrl_2=height_profile_ctrl_2,
        height_profile_ctrl_3=height_profile_ctrl_3,
        height_profile_poly_c2=height_profile_poly_c2,
        height_profile_poly_c4=height_profile_poly_c4,
        height_profile_poly_c6=height_profile_poly_c6,
        length=length,
    )
    return constrain_monotonic_nonincreasing(
        profile=profile
    ) and constrain_profile_smoothness(
        profile=profile, max_second_difference=max_second_difference
    ) and constrain_center_dominant_profile(
        profile=profile, min_center_minus_edge=min_center_minus_edge
    )


def constrain_grin_profile(
    *,
    refractive_index: float,
    refractive_index_edge: float | None = None,
    grin_profile_ctrl: Tuple[float, ...] | list[float] = (),
    grin_profile_ctrl_0: float | None = None,
    grin_profile_ctrl_1: float | None = None,
    grin_profile_ctrl_2: float | None = None,
    grin_profile_ctrl_3: float | None = None,
    max_second_difference: float = 0.05,
    min_center_minus_edge: float = 0.0,
    min_refractive_index: float = 1.0,
    max_refractive_index: float = 3.0,
    length: int = 64,
) -> bool:
    """Validate a radial GRIN profile."""
    profile = build_grin_profile(
        refractive_index=refractive_index,
        refractive_index_edge=refractive_index_edge,
        grin_profile_ctrl=grin_profile_ctrl,
        grin_profile_ctrl_0=grin_profile_ctrl_0,
        grin_profile_ctrl_1=grin_profile_ctrl_1,
        grin_profile_ctrl_2=grin_profile_ctrl_2,
        grin_profile_ctrl_3=grin_profile_ctrl_3,
        length=length,
    )
    in_bounds = bool(
        np.all(profile >= min_refractive_index) and np.all(profile <= max_refractive_index)
    )
    return (
        in_bounds
        and constrain_monotonic_nonincreasing(profile=profile)
        and constrain_profile_smoothness(
            profile=profile, max_second_difference=max_second_difference
        )
        and constrain_center_dominant_profile(
            profile=profile, min_center_minus_edge=min_center_minus_edge
        )
    )
