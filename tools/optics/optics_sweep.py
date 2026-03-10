from pathlib import Path

import numpy as np
import torch
from hydra_config import run_hydra

from cambrian import MjCambrianConfig, MjCambrianTrainer
from cambrian.envs.env import MjCambrianEnv
from cambrian.eyes.multi_eye import MjCambrianMultiEye
from cambrian.eyes.optics import (
    MjCambrianCircularApertureConfig,
    MjCambrianMaskApertureConfig,
    MjCambrianOpticsEye,
    MjCambrianOpticsEyeConfig,
)
from cambrian.renderer.render_utils import add_text, resize_with_aspect_fill
from cambrian.utils import RenderFrame


def sine_wave(step, min=0.1, max=1):
    """Does one full sine wave over the range of steps.
    Step=value between 0 and 1 where 1 is done and 0 is first step.
    The wave starts at max, goes to min, and returns to max."""
    return min + (max - min) * (1 + np.sin(2 * np.pi * step - np.pi / 2)) / 2


def _optics_render_override(
    self: MjCambrianOpticsEye, *, color=(255, 0, 0)
) -> RenderFrame:
    image = super(MjCambrianOpticsEye, self).render()
    if self._config.scale_intensity:
        # Artificially increase the intensity of the image
        image = 1 - torch.exp(-image * 10)  # scale for visualization
    image = add_text(image, "POV", fill=color)
    images = [image]

    pupil = self._pupil
    pupil = torch.clip(torch.abs(pupil), 0, 1).permute(1, 2, 0)
    pupil = resize_with_aspect_fill(pupil, *image.shape[:2])
    pupil = add_text(pupil, "Pupil", fill=color)
    if isinstance(self._config.aperture, MjCambrianCircularApertureConfig):
        pupil = add_text(
            pupil,
            f"Radius: {self._config.aperture.radius:0.2f}",
            (0, 12),
            fill=color,
            size=8,
        )
    images.append(pupil)

    psf = self._get_psf(self._depths[0])
    psf = self._resize(psf)
    psf = torch.clip(psf, 0, 1).permute(1, 2, 0)
    psf = resize_with_aspect_fill(psf, *image.shape[:2])
    psf = 1 - torch.exp(-psf * 100)  # scale for visualization
    psf = add_text(psf, "PSF", fill=color)
    images.append(psf)

    return torch.cat(images, dim=1)


# Monkey patch the render method to add additional images
MjCambrianOpticsEye.render = _optics_render_override


def step_callback(env: MjCambrianEnv):
    eye = env.agents["agent"].eyes["eye"]
    if isinstance(eye, MjCambrianMultiEye):
        assert len(eye.eyes) == 1
        eye = next(iter(eye.eyes.values()))
    assert isinstance(
        eye, MjCambrianOpticsEye
    ), f"Expected MjCambrianOpticsEye, got {type(eye)}"
    config: MjCambrianOpticsEyeConfig = eye.config

    step, max_steps = env.episode_step, env.max_episode_steps
    initialize = True
    if isinstance(config.aperture, MjCambrianCircularApertureConfig):
        config.aperture.radius = sine_wave(step / max_steps)
    if isinstance(config.aperture, MjCambrianMaskApertureConfig):
        # Only initialize the mask every 20 steps
        initialize = step % 20 == 0

    if initialize:
        eye.initialize()


def main(config: MjCambrianConfig):
    assert "agent" in config.env.agents
    assert len(config.env.agents["agent"].eyes) == 1
    assert "eye" in config.env.agents["agent"].eyes

    runner = MjCambrianTrainer(config)
    return runner.eval(step_callback=step_callback)


if __name__ == "__main__":
    config_path = (Path(__file__).resolve().parent / "configs").resolve()
    run_hydra(main, config_path=config_path, config_name="optics_sweep")
