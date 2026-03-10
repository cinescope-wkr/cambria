"""Penalty functions applied on top of the final train/eval fitness."""

from typing import TYPE_CHECKING, Dict

import numpy as np

from cambrian.ml.constraint_fns import build_grin_profile, build_height_profile

if TYPE_CHECKING:
    from cambrian.config import MjCambrianConfig


def optics_morphology_penalty(
    config: "MjCambrianConfig",
    *,
    penalty_scale: float = 1000.0,
    height_roughness_weight: float = 1.0,
    grin_roughness_weight: float = 1.0,
    concavity_weight: float = 5.0,
    edge_bulge_weight: float = 5.0,
    refractive_bounds_weight: float = 10.0,
    min_refractive_index: float = 1.0,
    max_refractive_index: float = 3.0,
) -> float:
    """Return a positive penalty for non-physical or overly rough optics morphologies."""
    eye = config.env.agents.agent.eyes.eye

    height_profile = build_height_profile(
        height_map=eye.height_map,
        height_profile_ctrl=eye.height_profile_ctrl,
        height_profile_poly=eye.height_profile_poly,
        height_profile_mix=eye.height_profile_mix,
        height_profile_ctrl_0=eye.height_profile_ctrl_0,
        height_profile_ctrl_1=eye.height_profile_ctrl_1,
        height_profile_ctrl_2=eye.height_profile_ctrl_2,
        height_profile_ctrl_3=eye.height_profile_ctrl_3,
        height_profile_poly_c2=eye.height_profile_poly_c2,
        height_profile_poly_c4=eye.height_profile_poly_c4,
        height_profile_poly_c6=eye.height_profile_poly_c6,
    )
    grin_profile = build_grin_profile(
        refractive_index=eye.refractive_index,
        refractive_index_edge=eye.refractive_index_edge,
        grin_profile_ctrl=eye.grin_profile_ctrl or (),
        grin_profile_ctrl_0=eye.grin_profile_ctrl_0,
        grin_profile_ctrl_1=eye.grin_profile_ctrl_1,
        grin_profile_ctrl_2=eye.grin_profile_ctrl_2,
        grin_profile_ctrl_3=eye.grin_profile_ctrl_3,
    )

    height_first = np.diff(height_profile)
    height_second = np.diff(height_profile, n=2)
    grin_second = np.diff(grin_profile, n=2)

    concavity_penalty = np.maximum(height_first, 0.0).sum()
    edge_bulge_penalty = max(height_profile[-1] - height_profile[0], 0.0)
    height_roughness_penalty = np.abs(height_second).sum()
    grin_roughness_penalty = np.abs(grin_second).sum()
    refractive_low_penalty = np.maximum(min_refractive_index - grin_profile, 0.0).sum()
    refractive_high_penalty = np.maximum(grin_profile - max_refractive_index, 0.0).sum()

    total_penalty = penalty_scale * (
        concavity_weight * concavity_penalty
        + edge_bulge_weight * edge_bulge_penalty
        + height_roughness_weight * height_roughness_penalty
        + grin_roughness_weight * grin_roughness_penalty
        + refractive_bounds_weight * (refractive_low_penalty + refractive_high_penalty)
    )
    return float(total_penalty)


def optics_morphology_penalty_breakdown(
    config: "MjCambrianConfig",
    *,
    min_refractive_index: float = 1.0,
    max_refractive_index: float = 3.0,
) -> Dict[str, float]:
    """Return penalty components for logging/debugging."""
    eye = config.env.agents.agent.eyes.eye

    height_profile = build_height_profile(
        height_map=eye.height_map,
        height_profile_ctrl=eye.height_profile_ctrl,
        height_profile_poly=eye.height_profile_poly,
        height_profile_mix=eye.height_profile_mix,
        height_profile_ctrl_0=eye.height_profile_ctrl_0,
        height_profile_ctrl_1=eye.height_profile_ctrl_1,
        height_profile_ctrl_2=eye.height_profile_ctrl_2,
        height_profile_ctrl_3=eye.height_profile_ctrl_3,
        height_profile_poly_c2=eye.height_profile_poly_c2,
        height_profile_poly_c4=eye.height_profile_poly_c4,
        height_profile_poly_c6=eye.height_profile_poly_c6,
    )
    grin_profile = build_grin_profile(
        refractive_index=eye.refractive_index,
        refractive_index_edge=eye.refractive_index_edge,
        grin_profile_ctrl=eye.grin_profile_ctrl or (),
        grin_profile_ctrl_0=eye.grin_profile_ctrl_0,
        grin_profile_ctrl_1=eye.grin_profile_ctrl_1,
        grin_profile_ctrl_2=eye.grin_profile_ctrl_2,
        grin_profile_ctrl_3=eye.grin_profile_ctrl_3,
    )

    return {
        "concavity": float(np.maximum(np.diff(height_profile), 0.0).sum()),
        "edge_bulge": float(max(height_profile[-1] - height_profile[0], 0.0)),
        "height_roughness": float(np.abs(np.diff(height_profile, n=2)).sum()),
        "grin_roughness": float(np.abs(np.diff(grin_profile, n=2)).sum()),
        "refractive_low": float(np.maximum(min_refractive_index - grin_profile, 0.0).sum()),
        "refractive_high": float(np.maximum(grin_profile - max_refractive_index, 0.0).sum()),
    }
