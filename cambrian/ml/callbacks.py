"""Callbacks used during training and/or evaluation."""

import csv
import glob
import inspect
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.experimental.callbacks import Callback as HydraCallback
from omegaconf import DictConfig
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
    ProgressBarCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from cambrian.envs import MjCambrianEnv
from cambrian.eyes.eye import MjCambrianEye
from cambrian.eyes.multi_eye import MjCambrianMultiEye
from cambrian.eyes.optics import MjCambrianOpticsEye
from cambrian.ml.fitness_fns import (
    compute_vision_quality_components,
    vision_quality_score,
)
from cambrian.ml.model import MjCambrianModel
from cambrian.renderer.render_utils import (
    add_border,
    add_text,
    generate_composite,
    resize_with_aspect_fill,
)
from cambrian.utils.logger import get_logger


class MjCambrianPlotMonitorCallback(BaseCallback):
    """Should be used with an EvalCallback to plot the evaluation results.

    This callback will take the monitor.csv file produced by the VecMonitor and
    plot the results and save it as an image. Should be passed as the
    `callback_after_eval` for the EvalCallback.

    Args:
        logdir (Path | str): The directory where the evaluation results are stored. The
            evaluations.npz file is expected to be at `<logdir>/<filename>.csv`. The
            resulting plot is going to be stored at
            `<logdir>/evaluations/<filename>.png`.
        filename (Path | str): The filename of the monitor file. The saved file will be
            `<logdir>/<filename>.csv`. And the resulting plot will be saved as
            `<logdir>/evaluations/<filename>.png`.
    """

    parent: EvalCallback

    def __init__(self, logdir: Path | str, filename: Path | str, n_episodes: int = 1):
        self.logdir = Path(logdir)
        self.filename = Path(filename)
        self.filename_csv = self.filename.with_suffix(".csv")
        self.filename_png = self.filename.with_suffix(".png")
        self.evaldir = self.logdir / "evaluations"
        self.evaldir.mkdir(parents=True, exist_ok=True)

        self.n_episodes = n_episodes
        self.n_calls = 0

    def _on_step(self) -> bool:
        if not (self.logdir / self.filename_csv).exists():
            get_logger().warning(f"No {self.filename_csv} file found.")
            return

        # Temporarily set the monitor ext so that the right file is read
        old_ext = Monitor.EXT
        Monitor.EXT = str(self.filename_csv)
        x, y = ts2xy(load_results(self.logdir), "timesteps")
        Monitor.EXT = old_ext
        if len(x) <= 20 or len(y) <= 20:
            get_logger().warning(f"Not enough {self.filename} data to plot.")
            return True
        original_x, original_y = x.copy(), y.copy()

        get_logger().info(f"Plotting {self.filename} results at {self.evaldir}")

        def moving_average(data, window=1):
            return np.convolve(data, np.ones(window), "valid") / window

        n = min(len(y) // 10, 1000)
        y = y.astype(float)

        if self.n_episodes > 1:
            assert len(y) % self.n_episodes == 0, (
                "n_episodes must be a common factor of the"
                f" number of episodes in the {self.filename} data."
            )
            y = y.reshape(-1, self.n_episodes).mean(axis=1)
        else:
            y = moving_average(y, window=n)

        x = moving_average(x, window=n).astype(int)

        # Make sure the x, y are of the same length
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]

        plt.plot(x, y)
        plt.plot(original_x, original_y, color="grey", alpha=0.3)

        plt.xlabel("Number of Timesteps")
        plt.ylabel("Rewards")
        plt.savefig(self.evaldir / self.filename.with_suffix(".png"))
        plt.cla()

        return True


class MjCambrianEvalCallback(EvalCallback):
    """Overwrites the default EvalCallback to support saving visualizations at the same
    time as the evaluation.

    Note:
        Only the first environment is visualized
    """

    def _init_callback(self):
        self.log_path = Path(self.log_path)
        self.n_evals = 0

        # Delete all the existing renders
        for f in glob.glob(str(self.log_path / "vis_*")):
            get_logger().info(f"Deleting {f}")
            Path(f).unlink()

        super()._init_callback()

    def _on_step(self) -> bool:
        # Early exit
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        env: MjCambrianEnv = self.eval_env.envs[0].unwrapped

        # Add some overlays
        # env.overlays["Exp"] = env.config.expname # TODO
        env.overlays["Best Mean Reward"] = f"{self.best_mean_reward:.2f}"
        env.overlays["Total Timesteps"] = f"{self.num_timesteps}"

        # Run the evaluation
        get_logger().info(f"Starting {self.n_eval_episodes} evaluation run(s)...")
        env.record(self.render)
        continue_training = super()._on_step()

        if self.render:
            # Save the visualization
            filename = Path(f"vis_{self.n_evals}")
            env.save(self.log_path / filename)
            env.record(False)

        if self.render:
            # Copy the most recent gif to latest.gif so that we can just watch this file
            for f in self.log_path.glob(str(filename.with_suffix(".*"))):
                shutil.copy(f, f.with_stem("latest"))

        self.n_evals += 1
        return continue_training

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]):
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        env: MjCambrianEnv = self.eval_env.envs[0].unwrapped

        # If done, do some logging
        if locals_["done"]:
            run = locals_["episode_counts"][locals_["i"]]
            cumulative_reward = env.stashed_cumulative_reward
            get_logger().info(f"Run {run} done. Cumulative reward: {cumulative_reward}")

        super()._log_success_callback(locals_, globals_)


class MjCambrianGPUUsageCallback(BaseCallback):
    """This callback will log the GPU usage at the end of each evaluation.
    We'll log to a csv."""

    parent: EvalCallback

    def __init__(
        self,
        logdir: Path | str,
        logfile: Path | str = "gpu_usage.csv",
        *,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.logfile = self.logdir / logfile
        with open(self.logfile, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timesteps",
                    "memory_reserved",
                    "max_memory_reserved",
                    "memory_available",
                ]
            )

    def _on_step(self) -> bool:
        if torch.cuda.is_available():
            # Get the GPU usage, log it and save it to the file
            device = torch.cuda.current_device()
            memory_reserved = torch.cuda.memory_reserved(device)
            max_memory_reserved = torch.cuda.max_memory_reserved(device)
            memory_available = torch.cuda.get_device_properties(device).total_memory

            # Log to the output file
            with open(self.logfile, "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        self.num_timesteps,
                        memory_reserved,
                        max_memory_reserved,
                        memory_available,
                    ]
                )

            # Log to stdout
            if self.verbose > 0:
                get_logger().debug(subprocess.getoutput("nvidia-smi"))
                get_logger().debug(torch.cuda.memory_summary())

        return True


class MjCambrianSavePolicyCallback(BaseCallback):
    """Should be used with an EvalCallback to save the policy.

    This callback will save the policy at the end of each evaluation. Should be passed
    as the `callback_after_eval` for the EvalCallback.

    Args:
        logdir (Path | str): The directory to store the generated visualizations. The
            resulting visualizations are going to be stored at
            `<logdir>/evaluations/visualization.gif`.
    """

    parent: EvalCallback

    def __init__(
        self,
        logdir: Path | str,
        *,
        save_history: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.logdir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_history = save_history

        self.model: MjCambrianModel = None

    def _on_step(self) -> bool:
        self.model.save_policy(self.logdir)
        if self.save_history:
            src = self.logdir / "policy.pt"
            dst = self.checkpoint_dir / f"policy_{self.num_timesteps}.pt"
            shutil.copy(src, dst)

        return True


class MjCambrianMultiViewRenderCallback(BaseCallback):
    """Save a compact scene/vision/optics triptych after each evaluation."""

    parent: EvalCallback

    def __init__(
        self,
        logdir: Path | str,
        *,
        fps: int = 20,
        max_steps: int | None = None,
        deterministic: bool = True,
        scene_subdir: str = "scene",
        vision_subdir: str = "vision",
        optics_subdir: str = "optics",
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.logdir = Path(logdir)
        self.evaldir = self.logdir / "evaluations"
        self.scene_dir = self.evaldir / scene_subdir
        self.vision_dir = self.evaldir / vision_subdir
        self.optics_dir = self.evaldir / optics_subdir
        for directory in (self.scene_dir, self.vision_dir, self.optics_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.fps = fps
        self.max_steps = max_steps
        self.deterministic = deterministic
        self._render_index = 0
        self._text_fill = (246, 246, 242)
        self._base_title_size = 36
        self._base_body_size = 30
        self._base_meta_size = 28
        self._base_card_size = 24

    def _on_step(self) -> bool:
        scene_frames, vision_frames, optics_frames, quality_stats = self._rollout_multiview()
        stem = f"vis_{self._render_index}"

        self._save_animation(self.scene_dir / stem, scene_frames)
        self._save_animation(self.vision_dir / stem, vision_frames)
        self._save_animation(self.optics_dir / stem, optics_frames)
        self._save_vision_quality(stem, quality_stats)

        for directory in (self.scene_dir, self.vision_dir, self.optics_dir):
            latest = directory / "latest.webp"
            latest.write_bytes((directory / f"{stem}.webp").read_bytes())

        self._render_index += 1
        return True

    def _rollout_multiview(
        self,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[dict[str, float]],
    ]:
        vec_env = self.parent.eval_env
        env: MjCambrianEnv = vec_env.envs[0].unwrapped
        max_steps = self.max_steps or env.max_episode_steps

        obs = vec_env.reset()
        scene_frames: list[torch.Tensor] = []
        vision_frames: list[torch.Tensor] = []
        optics_frames: list[torch.Tensor] = []
        quality_stats: list[dict[str, float]] = []

        self._append_multiview_frames(
            env, scene_frames, vision_frames, optics_frames, quality_stats
        )

        for _ in range(max_steps):
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, _, dones, _ = vec_env.step(action)
            self._append_multiview_frames(
                env, scene_frames, vision_frames, optics_frames, quality_stats, action
            )
            if bool(np.any(dones)):
                break

        return scene_frames, vision_frames, optics_frames, quality_stats

    def _append_multiview_frames(
        self,
        env: MjCambrianEnv,
        scene_frames: list[torch.Tensor],
        vision_frames: list[torch.Tensor],
        optics_frames: list[torch.Tensor],
        quality_stats: list[dict[str, float]],
        action: np.ndarray | None = None,
    ) -> None:
        scene = env.render()
        if isinstance(scene, tuple):
            scene = scene[0]

        height, width = scene.shape[:2]
        style = self._build_ui_style(width=width, height=height)
        scene_frames.append(self._format_scene_frame(env, scene, action, style=style))
        raw_vision, vision_label = self._get_vision_source(env)
        if raw_vision is None:
            vision_frames.append(
                self._blank_frame(
                    height,
                    width,
                    "Vision unavailable",
                    fill=style["text_fill"],
                )
            )
        else:
            components = compute_vision_quality_components(raw_vision.detach().cpu().numpy())
            quality_stats.append(components)
            vision_frames.append(
                self._format_vision_frame(
                    env,
                    raw_vision,
                    vision_label,
                    components,
                    height,
                    width,
                    style=style,
                )
            )
        optics_frames.append(self._capture_optics_frame(env, height, width, style=style))

    def _get_vision_source(
        self, env: MjCambrianEnv
    ) -> tuple[torch.Tensor | None, str]:
        eye = self._get_primary_eye(env)
        if eye is None:
            return None, "Vision unavailable"

        if isinstance(eye, MjCambrianMultiEye):
            images: Dict[float, Dict[float, torch.Tensor]] = {}
            for sub_eye in eye.eyes.values():
                lat, lon = sub_eye.config.coord
                images.setdefault(lat, {})[lon] = sub_eye.prev_obs
            return generate_composite(images), f"Agent Vision: {eye.num_eyes} eyes"

        return eye.prev_obs, "Agent Vision"

    def _format_vision_frame(
        self,
        env: MjCambrianEnv,
        frame: torch.Tensor,
        label: str,
        components: dict[str, float],
        height: int,
        width: int,
        *,
        style: dict[str, int | float | tuple[int, int, int]],
    ) -> torch.Tensor:
        frame = resize_with_aspect_fill(frame, height, width)
        frame = self._label_frame(
            frame,
            label,
            fill=style["text_fill"],
            size=style["title_size"],
        )
        eye = self._get_primary_eye(env)
        num_eyes = eye.num_eyes if isinstance(eye, MjCambrianMultiEye) else int(eye is not None)
        fov = getattr(getattr(eye, "config", None), "fov", None)
        resolution = getattr(getattr(eye, "config", None), "resolution", None)
        info = [
            f"Eval: {self._render_index}",
            f"Episode Step: {env.episode_step}/{env.max_episode_steps}",
            f"Eyes: {num_eyes}",
        ]
        if resolution is not None:
            info.append(f"Resolution: {list(resolution)}")
        if fov is not None:
            info.append(
                f"FOV: [{self._fmt_num(float(fov[0]))}, {self._fmt_num(float(fov[1]))}]"
            )
        info.append(f"Mean: {self._fmt_num(components['mean_intensity'])}")
        info.append(f"Contrast: {self._fmt_num(components['contrast'])}")
        info.append(f"Detail: {self._fmt_num(components['detail'])}")
        return add_text(
            frame,
            "\n".join(info),
            (style["text_x"], style["body_y"]),
            size=style["body_size"],
            fill=style["text_fill"],
        )

    def _format_scene_frame(
        self,
        env: MjCambrianEnv,
        frame: torch.Tensor,
        action: np.ndarray | None,
        style: dict[str, int | float | tuple[int, int, int]],
    ) -> torch.Tensor:
        frame = self._label_frame(
            frame,
            "Scene: Shallow Cambria",
            fill=style["text_fill"],
            size=style["title_size"],
        )
        info = [
            f"Name: {env.name}",
            f"Eval: {self._render_index}",
            f"Train Timesteps: {self.num_timesteps}",
            f"Episode Step: {env.episode_step}/{env.max_episode_steps}",
            f"Cumulative Reward: {env.cumulative_reward:.2f}",
        ]
        if action is not None:
            flattened = np.asarray(action).reshape(-1)
            action_text = ", ".join(self._fmt_num(float(value)) for value in flattened[:4])
            info.append(f"Action: [{action_text}]")
        return add_text(
            frame,
            "\n".join(info),
            (style["text_x"], style["body_y"]),
            size=style["body_size"],
            fill=style["text_fill"],
        )

    def _capture_optics_frame(
        self,
        env: MjCambrianEnv,
        height: int,
        width: int,
        style: dict[str, int | float | tuple[int, int, int]],
    ) -> torch.Tensor:
        optics_eye = self._get_primary_optics_eye(env)
        if optics_eye is None:
            return self._blank_frame(
                height,
                width,
                "Optics panel unavailable",
                fill=style["text_fill"],
            )

        # avoid the aspect-ratio-induced crop: build the optics 2x2 board first,
        # then place it inside a fixed header/content canvas.
        header_h = int(style["header_height"])
        header_h = max(header_h, 1)
        content_h = max(height - header_h, 1)
        content_w = width
        panel_gap = int(style["panel_gap"])
        side_gap = int(style["side_gap"])
        content_board_h = max(content_h - side_gap, 1)
        content_board_w = max(content_w - 2 * side_gap, 1)
        panel_height = max((content_board_h - panel_gap) // 2, 1)
        panel_width = max((content_board_w - panel_gap) // 2, 1)

        pupil = resize_with_aspect_fill(
            torch.clip(torch.abs(optics_eye._pupil), 0, 1).permute(1, 2, 0),
            panel_height,
            panel_width,
        )
        pupil = self._card_frame(
            pupil,
            "Pupil",
            fill=style["text_fill"],
            text_size=int(style["card_label_size"]),
        )

        if len(optics_eye._depths) > 0:
            depth = optics_eye._depths[0]
        else:
            depth = torch.tensor(
                10.0 * max(optics_eye.config.focal), device=optics_eye.prev_obs.device
            )
        psf = optics_eye._resize(optics_eye._get_psf(depth))
        psf = torch.clip(psf, 0, 1).permute(1, 2, 0)
        psf = 1 - torch.exp(-psf * 100)
        psf = resize_with_aspect_fill(psf, panel_height, panel_width)
        psf = self._card_frame(
            psf,
            "PSF",
            fill=style["text_fill"],
            text_size=int(style["card_label_size"]),
        )

        height_profile = self._make_profile_card(
            optics_eye._radial_height_profile,
            panel_height,
            panel_width,
            "Lens Height",
            color=(0.91, 0.83, 0.33),
            fill=style["text_fill"],
            footer=f"max {optics_eye.config.max_height_um:.2f}um",
            style=style,
        )
        refractive_profile = self._make_profile_card(
            optics_eye._radial_refractive_index_profile,
            panel_height,
            panel_width,
            "Refractive Index",
            color=(0.38, 0.90, 0.83),
            fill=style["text_fill"],
            footer=(
                f"n {optics_eye.config.refractive_index:.2f}"
                f" -> {optics_eye.config.refractive_index_edge:.2f}"
            ),
            style=style,
        )

        board = torch.full((content_h, width, 3), 0.02, dtype=torch.float32, device=pupil.device)
        board_x = max((content_w - (2 * panel_width + panel_gap)) // 2, 0)
        board_y = max((content_h - (2 * panel_height + panel_gap)) // 2, 0)
        board[board_y : board_y + panel_height, board_x : board_x + panel_width] = pupil
        board[
            board_y : board_y + panel_height,
            board_x + panel_width + panel_gap : board_x + 2 * panel_width + panel_gap,
        ] = psf
        board[
            board_y + panel_height + panel_gap : board_y + 2 * panel_height + panel_gap,
            board_x : board_x + panel_width,
        ] = height_profile
        board[
            board_y + panel_height + panel_gap : board_y + 2 * panel_height + panel_gap,
            board_x + panel_width + panel_gap : board_x + 2 * panel_width + panel_gap,
        ] = refractive_profile

        header = torch.full(
            (header_h, width, 3),
            0.02,
            dtype=torch.float32,
            device=pupil.device,
        )
        frame = torch.cat([header, board], dim=0)

        radius = getattr(optics_eye.config.aperture, "radius", None)
        meta = (
            f"FOV [{self._fmt_num(float(optics_eye.config.fov[0]))}, "
            f"{self._fmt_num(float(optics_eye.config.fov[1]))}]"
        )
        if radius is not None:
            meta += f"  aperture {self._fmt_num(float(radius))}"
        meta += f"  mix {self._fmt_num(float(optics_eye.config.height_profile_mix))}"
        frame = add_text(
            frame,
            meta,
            (style["text_x"], int(header_h * 0.45)),
            size=style["meta_size"],
            fill=style["text_fill"],
        )
        return self._label_frame(
            frame,
            "Optical System",
            fill=style["text_fill"],
            size=style["title_size"],
        )

    def _build_ui_style(
        self, width: int, height: int
    ) -> dict[str, int | float | tuple[int, int, int]]:
        scale = math.sqrt((width * height) / (1600.0 * 900.0))
        scale = min(max(scale, 0.75), 2.2)

        return {
            "scale": scale,
            "title_size": max(int(round(self._base_title_size * scale)), 28),
            "body_size": max(int(round(self._base_body_size * scale)), 22),
            "meta_size": max(int(round(self._base_meta_size * scale)), 20),
            "card_label_size": max(int(round(self._base_card_size * scale)), 18),
            "header_height": max(min(int(round(height * 0.16)), 240), 140),
            "side_gap": max(int(round(12 * scale)), 10),
            "panel_gap": max(int(round(14 * scale)), 10),
            "text_x": max(int(round(20 * scale)), 16),
            "body_y": max(int(round(18 * scale)), 14)
            + max(int(round(self._base_title_size * scale)), 28),
            "text_fill": self._text_fill,
        }

    def _get_primary_eye(self, env: MjCambrianEnv) -> MjCambrianEye | None:
        agent = next((agent for agent in env.agents.values() if agent.trainable), None)
        if agent is None or len(agent.eyes) == 0:
            return None
        return next(iter(agent.eyes.values()))

    def _get_primary_optics_eye(self, env: MjCambrianEnv) -> MjCambrianOpticsEye | None:
        eye = self._get_primary_eye(env)
        if eye is None:
            return None
        if isinstance(eye, MjCambrianOpticsEye):
            return eye
        if isinstance(eye, MjCambrianMultiEye):
            for sub_eye in eye.eyes.values():
                if isinstance(sub_eye, MjCambrianOpticsEye):
                    return sub_eye
        return None

    def _blank_frame(
        self, height: int, width: int, text: str, *, fill: tuple[int, int, int]
    ) -> torch.Tensor:
        frame = torch.full((height, width, 3), 0.06, dtype=torch.float32)
        return self._label_frame(frame, text, fill=fill)

    def _card_frame(
        self,
        frame: torch.Tensor,
        label: str,
        *,
        fill: tuple[int, int, int],
        text_size: int | None = None,
    ) -> torch.Tensor:
        frame = add_border(torch.clamp(frame, 0.0, 1.0), 2, color=(0.08, 0.09, 0.08))
        return add_text(
            frame,
            label,
            (12, 10),
            size=text_size or self._base_card_size,
            fill=fill,
        )

    def _make_profile_card(
        self,
        profile: torch.Tensor,
        height: int,
        width: int,
        label: str,
        *,
        color: tuple[float, float, float],
        fill: tuple[int, int, int],
        style: dict[str, int | float | tuple[int, int, int]],
        footer: str,
    ) -> torch.Tensor:
        canvas = torch.full((height, width, 3), 0.05, dtype=torch.float32, device=profile.device)
        scale = float(style["scale"])
        left_pad = max(12, int(round(18 * scale)))
        right_pad = left_pad
        top_pad = max(24, int(round(36 * scale)))
        bottom_pad = max(30, int(round(46 * scale)))
        usable_width = max(width - left_pad - right_pad, 2)
        usable_height = max(height - top_pad - bottom_pad, 2)
        normalized = profile.detach().float()
        min_value = float(normalized.min())
        max_value = float(normalized.max())
        if max_value - min_value < 1e-8:
            normalized = torch.zeros_like(normalized)
        else:
            normalized = (normalized - min_value) / (max_value - min_value)

        x_values = torch.linspace(left_pad, left_pad + usable_width - 1, normalized.numel(), device=profile.device)
        y_values = top_pad + (1.0 - normalized) * (usable_height - 1)

        for index in range(max(normalized.numel() - 1, 0)):
            x0 = int(round(float(x_values[index])))
            y0 = int(round(float(y_values[index])))
            x1 = int(round(float(x_values[index + 1])))
            y1 = int(round(float(y_values[index + 1])))
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            xs = torch.linspace(x0, x1, steps + 1, device=profile.device).round().to(torch.long)
            ys = torch.linspace(y0, y1, steps + 1, device=profile.device).round().to(torch.long)
            xs = torch.clamp(xs, 0, width - 1)
            ys = torch.clamp(ys, 0, height - 1)
            canvas[ys, xs] = torch.tensor(color, device=profile.device)

        canvas[top_pad : top_pad + usable_height, left_pad] = 0.35
        canvas[top_pad + usable_height - 1, left_pad : left_pad + usable_width] = 0.35
        canvas = self._card_frame(canvas, label, fill=fill)
        footer_size = max(16, int(round(0.8 * float(style["meta_size"]))))
        canvas = add_text(
            canvas,
            f"{footer}\nmin {self._fmt_num(min_value)}  max {self._fmt_num(max_value)}",
            (12, height - max(32, int(round(44 * scale)))),
            size=footer_size,
            fill=self._text_fill,
        )
        return canvas

    def _label_frame(
        self,
        frame: torch.Tensor,
        label: str,
        *,
        fill: tuple[int, int, int],
        size: int | None = None,
    ) -> torch.Tensor:
        frame = torch.clamp(frame, 0.0, 1.0)
        return add_text(
            frame,
            label,
            (16, 12),
            size=size or self._base_title_size,
            fill=self._text_fill,
        )

    def _fmt_num(self, value: float) -> str:
        return f"{value:.2f}"

    def _save_animation(self, path: Path, frames: list[torch.Tensor]) -> None:
        array = (torch.stack(frames) * 255.0).to(torch.uint8).cpu().numpy()
        array = np.flip(array, axis=1)
        imageio.mimwrite(path.with_suffix(".webp"), array, fps=self.fps, lossless=True)

    def _save_vision_quality(
        self, stem: str, quality_stats: list[dict[str, float]]
    ) -> None:
        if not quality_stats:
            return

        aggregate = {
            key: float(np.mean([stats[key] for stats in quality_stats]))
            for key in quality_stats[0]
        }
        aggregate["score"] = vision_quality_score(aggregate)
        aggregate["num_frames"] = len(quality_stats)

        indexed_path = self.evaldir / f"{stem}_vision_quality.json"
        latest_json = self.evaldir / "vision_quality.json"
        latest_txt = self.evaldir / "vision_quality.txt"

        with open(indexed_path, "w") as f:
            json.dump(aggregate, f, indent=2, sort_keys=True)
        with open(latest_json, "w") as f:
            json.dump(aggregate, f, indent=2, sort_keys=True)
        with open(latest_txt, "w") as f:
            f.write(str(aggregate["score"]))


class MjCambrianProgressBarCallback(ProgressBarCallback):
    """Overwrite the default progress bar callback to flush the pbar on deconstruct."""

    def __del__(self):
        """This string will restore the terminal back to its original state."""
        if hasattr(self, "pbar"):
            print("\x1b[?25h")


class MjCambrianCallbackListWithSharedParent(CallbackList):
    def __init__(self, callbacks: Iterable[BaseCallback] | Dict[str, BaseCallback]):
        if isinstance(callbacks, dict):
            callbacks = callbacks.values()

        self.callbacks = []
        super().__init__(list(callbacks))

    @property
    def parent(self):
        return getattr(self.callbacks[0], "parent", None)

    @parent.setter
    def parent(self, parent):
        for cb in self.callbacks:
            cb.parent = parent


# ==================


class MjCambrianSaveConfigCallback(HydraCallback):
    """This callback will save the resolved hydra config to the logdir."""

    def on_run_start(self, config: DictConfig, **kwargs):
        self._save_config(config)

    def on_multirun_start(self, config: DictConfig, **kwargs):
        self._save_config(config)

    def _save_config(self, config: DictConfig):
        from omegaconf import OmegaConf

        config.logdir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, config.logdir / "full.yaml")


class MjCambrianNevergradBestLogger:
    """Log the current best candidate for each Nevergrad generation.

    This complements Nevergrad's ParametersLogger by maintaining a compact summary file
    that is directly usable for plotting evolution trajectories.
    """

    def __init__(
        self,
        jsonl_path: Path | str,
        csv_path: Path | str,
        *,
        append: bool = False,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.csv_path = Path(csv_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not append:
            for path in (self.jsonl_path, self.csv_path):
                if path.exists():
                    path.unlink()
        self._best_by_generation: Dict[int, Dict[str, Any]] = {}
        self._best_loss: float | None = None

    def __call__(self, optimizer, candidate, loss: float) -> None:
        generation = int(candidate.generation)
        current_best = self._best_by_generation.get(generation)
        if current_best is not None and loss >= current_best["loss"]:
            return

        record = self._make_record(optimizer, candidate, loss)
        record["is_overall_best"] = self._best_loss is None or loss < self._best_loss
        self._best_by_generation[generation] = record
        if record["is_overall_best"]:
            self._best_loss = loss

        with self.jsonl_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        self._rewrite_csv()

    def _make_record(self, optimizer, candidate, loss: float) -> Dict[str, Any]:
        from nevergrad.parametrization import helpers

        record: Dict[str, Any] = {
            "generation": int(candidate.generation),
            "num_ask": int(optimizer.num_ask),
            "num_tell": int(optimizer.num_tell),
            "uid": candidate.uid,
            "lineage": candidate.heritage["lineage"],
            "loss": float(loss),
        }
        for name, param in helpers.flatten(candidate, with_containers=False, order=1):
            value = param.value
            if isinstance(value, (np.float64, np.int_, np.bool_)):
                value = value.item()
            if inspect.ismethod(value):
                value = repr(value.__self__)
            key = name if name else "0"
            if isinstance(value, np.ndarray):
                record[key] = value.tolist()
            else:
                record[key] = value
        return record

    def _rewrite_csv(self) -> None:
        rows = [self._best_by_generation[g] for g in sorted(self._best_by_generation)]
        if not rows:
            return

        fieldnames = sorted({key for row in rows for key in row.keys()})
        with self.csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value) if isinstance(value, list) else value
                        for key, value in row.items()
                    }
                )
