import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from hydra_config import run_hydra

from cambrian import MjCambrianConfig, MjCambrianTrainer
from cambrian.envs.env import MjCambrianEnv
from cambrian.eyes.eye import MjCambrianEye
from cambrian.eyes.multi_eye import MjCambrianMultiEye
from cambrian.eyes.optics import MjCambrianOpticsEye
from cambrian.ml.fitness_fns import (
    compute_vision_quality_components,
    vision_quality_score,
)
from cambrian.renderer.render_utils import generate_composite, resize_with_aspect_fill


def _get_primary_eye(env: MjCambrianEnv) -> MjCambrianEye | None:
    agent = next((agent for agent in env.agents.values() if agent.trainable), None)
    if agent is None or len(agent.eyes) == 0:
        return None
    return next(iter(agent.eyes.values()))


def _get_primary_optics_eye(env: MjCambrianEnv) -> MjCambrianOpticsEye | None:
    eye = _get_primary_eye(env)
    if isinstance(eye, MjCambrianOpticsEye):
        return eye
    if isinstance(eye, MjCambrianMultiEye):
        for sub_eye in eye.eyes.values():
            if isinstance(sub_eye, MjCambrianOpticsEye):
                return sub_eye
    return None


def _get_vision_source(env: MjCambrianEnv) -> torch.Tensor | None:
    eye = _get_primary_eye(env)
    if eye is None:
        return None
    if isinstance(eye, MjCambrianMultiEye):
        images = {}
        for sub_eye in eye.eyes.values():
            lat, lon = sub_eye.config.coord
            images.setdefault(lat, {})[lon] = sub_eye.prev_obs
        return generate_composite(images)
    return eye.prev_obs


def _compute_psf_report(optics_eye: MjCambrianOpticsEye) -> dict:
    if len(optics_eye._depths) > 0:
        depth = optics_eye._depths[0]
    else:
        depth = torch.tensor(
            10.0 * max(optics_eye.config.focal), device=optics_eye.prev_obs.device
        )
    psf = optics_eye._resize(optics_eye._get_psf(depth)).detach().cpu().numpy()
    sums = psf.sum(axis=(1, 2))
    peak_coords = []
    centroids = []
    symmetry_error = []
    for channel in psf:
        peak = np.unravel_index(np.argmax(channel), channel.shape)
        peak_coords.append([int(peak[0]), int(peak[1])])
        yy, xx = np.indices(channel.shape)
        total = max(channel.sum(), 1e-12)
        centroids.append(
            [float((yy * channel).sum() / total), float((xx * channel).sum() / total)]
        )
        flip_h = np.flip(channel, axis=1)
        flip_v = np.flip(channel, axis=0)
        base = max(float(np.mean(np.abs(channel))), 1e-12)
        symmetry_error.append(
            float(
                0.5
                * (np.mean(np.abs(channel - flip_h)) + np.mean(np.abs(channel - flip_v)))
                / base
            )
        )

    center = [(psf.shape[1] - 1) / 2.0, (psf.shape[2] - 1) / 2.0]
    centroid_offsets = [
        float(np.linalg.norm(np.array(centroid) - np.array(center)))
        for centroid in centroids
    ]

    return {
        "psf_shape": list(psf.shape),
        "channel_sums": [float(value) for value in sums],
        "peak_coords": peak_coords,
        "centroids": centroids,
        "centroid_offsets": centroid_offsets,
        "symmetry_error": symmetry_error,
    }


def _save_frame(path: Path, frame: torch.Tensor) -> None:
    array = (torch.clamp(frame, 0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
    array = np.flip(array, axis=0)
    iio.imwrite(path, array)


def main(config: MjCambrianConfig):
    trainer = MjCambrianTrainer(config)
    vec_env = trainer._make_env(config.eval_env, 1, monitor=None)
    env: MjCambrianEnv = vec_env.envs[0].unwrapped
    diagnostics_dir = config.expdir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    obs = vec_env.reset()
    scene = env.render()
    if isinstance(scene, tuple):
        scene = scene[0]
    _save_frame(diagnostics_dir / "scene_0.png", scene)

    optics_eye = _get_primary_optics_eye(env)
    assert optics_eye is not None, "Expected an optics eye for diagnostics."

    psf_report = _compute_psf_report(optics_eye)
    quality_series = []

    vision = _get_vision_source(env)
    if vision is not None:
        _save_frame(
            diagnostics_dir / "vision_0.png",
            resize_with_aspect_fill(vision, scene.shape[0], scene.shape[1]),
        )
        quality_series.append(compute_vision_quality_components(vision.detach().cpu().numpy()))

    zero_action = np.zeros(
        (vec_env.num_envs,) + vec_env.action_space.shape,
        dtype=vec_env.action_space.dtype,
    )

    for step in range(1, 9):
        obs, _, dones, _ = vec_env.step(zero_action)
        scene = env.render()
        if isinstance(scene, tuple):
            scene = scene[0]
        vision = _get_vision_source(env)
        if vision is not None:
            quality_series.append(
                compute_vision_quality_components(vision.detach().cpu().numpy())
            )
            if step in (1, 4, 8):
                _save_frame(
                    diagnostics_dir / f"vision_{step}.png",
                    resize_with_aspect_fill(vision, scene.shape[0], scene.shape[1]),
                )
        if bool(np.any(dones)):
            break

    quality_report = {
        key: float(np.mean([row[key] for row in quality_series]))
        for key in quality_series[0]
    }
    quality_report["score"] = vision_quality_score(quality_report)

    warnings = []
    if quality_report["mean_intensity"] < 0.03:
        warnings.append("Vision is severely underexposed.")
    if quality_report["detail"] < 0.01:
        warnings.append("Vision retains very little spatial structure.")
    if max(psf_report["centroid_offsets"]) > 3.0:
        warnings.append("PSF centroid is substantially off-center.")

    report = {
        "expdir": str(config.expdir),
        "example": getattr(config, "example", None),
        "psf": psf_report,
        "vision_quality": quality_report,
        "warnings": warnings,
    }
    with open(diagnostics_dir / "optics_report.json", "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    config_path = (Path(__file__).resolve().parents[2] / "cambrian" / "configs").resolve()
    run_hydra(main, config_path=config_path, config_name="base")
