<<<<<<< HEAD
# Cambrian: Co-Optimizing Vision and Control in MuJoCo

Cambrian is a reinforcement-learning framework for end-to-end experiments that jointly optimize locomotion/control policies and visual front-end morphology. It combines MuJoCo simulation, Hydra-driven configuration, Stable-Baselines3 PPO training, and optional evolutionary search over optics-related parameters. The repository is designed for three operating modes: standard policy training, deterministic/controlled evaluation, and parameter search/evolution.

This README focuses on practical usage and what can be configured: purpose, implementation, runnable modes, task families, environment parameters, and render modes.

## Fork and attribution

- Original project: [Artificial Cambrian Intelligence (ACI)](https://github.com/cambrian-org/ACI), developed by the original ACI team.
- This repository is a fork of that project.
- Maintainer of this fork: Jinwoo Lee (`cinescope@kaist.ac.kr`).

## What this project does

- Trains and evaluates agents in MuJoCo environments with multi-agent support.
- Represents each agent’s perception through an eye module (RGB/depth vision pipeline).
- Supports biologically inspired optics (pupil, aperture, blur, noise, depth-dependent point-spread) via an `Optics` eye model.
- Optionally evolves optics-related structure and training hyperparameters with Hydra + Nevergrad.
- Captures artifacts and reports for reproducibility (checkpoints, videos/images, configs, eval summaries).

## High-level architecture

- **`cambrian/main.py`**: CLI entrypoint (`--train` or `--eval`), Hydra composition, and dispatch.
- **`cambrian/envs/env.py`**: Core environment wrapper (`MjCambrianEnv`) with multi-agent step/reward/done handling.
- **`cambrian/agents/`**: Agent definition and control interfaces.
- **`cambrian/eyes/`**: Base eye, multi-eye, and optics eye implementations.
- **`cambrian/renderer/`**: MuJoCo rendering adapter with support for multiple render and save modes.
- **`cambrian/ml/`**: PPO model wrapper, training loop, constraints, fitness and evolution hooks.
- **`cambrian/configs/`**: Fully declarative experiment definitions via Hydra.
- **`scripts/`**: Reusable launch templates for training/eval/evolution.

## Core design

### 1) Environment + agent contract

- The environment is structured as a vector-like multi-agent interface (`obs`, `reward`, `terminated`, `truncated`, `infos`).
- Agent control is policy-driven with action spaces defined in `agents` configs.
- Environment lifecycle supports train/eval modes separately and supports deterministic resets and episode bookkeeping.

### 2) Perception stack

`Eye` modules produce observations from MuJoCo camera data:

- `MjCambrianEye`: single camera, RGB/depth based path.
- `MjCambrianMultiEye`: configurable eye lattice (`lat_range`, `lon_range`, `num_eyes`) with optional flattening into observation arrays.
- `MjCambrianOpticsEye`: optics pipeline including pupil model, aperture model, depth-aware blur, optional Poisson/Gaussian noise, and optional precomputed depth support.

### 3) Vision model and physics link

- Rendering is delegated to `cambrian/renderer/renderer.py`.
- All rendered observations are standardized through config-driven metadata and output mode selection.
- Renderer outputs feed directly into policy observation space used by PPO.

### 4) Trainer + optimization loop

- `MjCambrianTrainer` orchestrates training/evaluation.
- Uses `EvalCallback`/`CheckpointCallback`-style lifecycle (periodic eval, best-model tracking, periodic saves).
- Model is SB3 PPO by default (`MultiInputPolicy`, shared defaults are overrideable via config).
- Checkpoint policy naming and save path are deterministic and organized by experiment run.

### 5) Evolution

- Separate evolution pipeline is available and activated through Hydra (`evo` group)
- Search space is controlled by mutation presets in `cambrian/configs/evo/mutations/*.yaml`.
- Constraints and penalties can restrict generated candidates before simulation, including optics-morphology and resource limits.
- Supports generation/rank based runs and distributed launchers via Hydra.

## Modes of operation

### Standard training

Run training with a trainer config and selected task/environment.

```bash
python cambrian/main.py --train env=... trainer=... task=... [+optional overrides]
```

### Evaluation

Run deterministic or stochastic policy evaluation against a fixed environment config.

```bash
python cambrian/main.py --eval env=... eval_env=... trainer=... task=... model=... [+overrides]
```

### Multi-run sweeps / hyperparameter search

Hydra multi-run is supported for batched experiments.

```bash
python cambrian/main.py --train -m hydra/sweeper=... trainer=... [other overrides]
```

### Evolution mode

Activate evolution/evolver configs to search over parameter space while using PPO as the base learning loop.

```bash
python cambrian/main.py --train hydra/launcher=... evo=all trainer=... task=... env=...
```

## Command-line patterns

- `--train` and `--eval` are mutually exclusive.
- Use Hydra override syntax for any nested key.
- Most experiments are launchable directly from an experiment alias in `cambrian/configs/exp/`.
- Typical run scripts under `scripts/` encode stable baselines and launcher-specific environment variables.

## Experiment families

The following are validated and documented by config:

- `example/navigator` family
  - `navigation.yaml`: path/navigation benchmark in open mazes.
  - `tracking.yaml`: moving-target tracking with reward shaping.
  - `detection.yaml`: detection-style objective with adversarial distractors.
- Cambrian-specific suites
  - `cambria_detection.yaml`
  - `cambria_detection_preview.yaml`
  - `cambria_detection_vision_quality.yaml`
  - `cambria_detection_diagnostic.yaml`
  - `cambria_shallows.yaml`
  - `cambria_shallow_vision_quality.yaml`
- Task-level overrides are further specialized in `configs/task/`.

A practical way to browse all variants is `configs/exp/` and the `configs/task/*.yaml` hierarchy.

## Environment configuration knobs

### Core environment settings

- `frame_skip`: simulation step multiplier between policy actions.
- `max_episode_steps`: rollout termination horizon.
- `n_eval_episodes`: number of episodes per evaluation phase.
- `n_envs`: parallel environments for SB3 vectorized training.
- `eval_render_mode`: rendering mode used specifically during evaluation runs.
- `overlay` / `overlay_options`: optional on-frame visual overlays.

### Maze and map-based environments

- Maze tasks are configured through `maze_env.yaml` and concrete map configs.
- Common map families include
  - `SHALLOWS`
  - `COMPLEX`
  - `OPEN`
  - `MAZE`
  - `NARROW`
- `maze_num` / spawn / geometry options are controlled in maze-aware task configs.

### Agent configuration

Available `env.agents` presets include point/object variants and texturing options.

- `agents/point.yaml` baseline point agent.
- `agents/point_cambria.yaml` Cambria-compatible point baseline.
- `agents/point_seeker.yaml` and seeker variants for task variants.
- `agents/object*.yaml` textured and goal-object variants.

### Eye and optics presets

Under `configs/env/agents/eyes/` and `configs/env/agents/eyes/aperture/`:

- `eye.yaml`: base eye defaults.
- `multi_eye.yaml`: lattice/multi-camera setup.
- `optics.yaml`: enables optics stack.
- `aperture/circular.yaml`, `aperture/elliptical.yaml`, `aperture/mask.yaml`, `aperture/aperture.yaml` for aperture style swaps.

## Render modes and output formats

Renderer metadata supports

- `human`
  - usually mapped to live viewer.
- `rgb_array`
  - off-screen RGB frame output used by learning and recording.
- `depth_array`
  - depth map stream for depth-aware training and optics diagnostics.

Save modes in renderer config include

- `NONE`
- `GIF`
- `MP4`
- `PNG`
- `WEBP`
- `USD`

`rgb_array` is required for frame saving in most modes because video/image writers consume RGB sequences.

## Typical trainer defaults

Common defaults (can be overridden per experiment)

- `trainer.trainer.yaml`
  - `total_timesteps`: `500_000`
  - `n_envs`: `1`
  - `max_episode_steps`: `256`
- `trainer.model.model.yaml`
  - PPO policy: `MultiInputPolicy`
  - `n_steps`: `2048`
  - `batch_size`: derived from `n_steps * n_envs // 32`
  - learning rate: `1e-3`

Cambrian preview/evolution configs often use shorter or custom schedules in `configs/exp/cambria_*`.

## Model and artifact outputs

Training/evaluation runs typically produce:

- `checkpoints/` and `best_model`
- `policy.pt` (TorchScript/checkpoint compatibility path)
- `config.yaml` and `eval_config.yaml`
- `fitness.csv` and `best_fitness.csv` for evolution/eval pipelines
- `eval_env.xml` and `env.xml` style scene export when enabled
- rendered videos/images when `save_mode` is enabled

## Where to add new experiments

1. Define or clone an experiment YAML under `configs/exp/`.
2. Compose reusable pieces from `base.yaml`, `env/`, `task/`, `agent/`, `eyes/`, `renderer/`, `trainer/`, and `evo/`.
3. Add run aliases or script entries in `scripts/`.
4. Validate via short `--train` runs before long evolution sweeps.

## Recommended workflow for reproducibility

- Keep one canonical base config and apply overrides for each condition.
- Log every run with the generated `run` directory and store exact Hydra override set.
- Use deterministic seeds explicitly in task/env/optimizer configs.
- Prefer short pilot runs before launching multi-GPU / multi-node sweeps.

## Useful launch scripts

- `scripts/run.sh`
- `scripts/run_cambria_preview_experiment.sh`
- `scripts/run_cambria_principled_experiment.sh`
- `scripts/run_cambria_signal_experiment.sh`
- `scripts/smoke_test_evo.sh`
- `scripts/run_vast1_train.sh`

## Repository structure (at a glance)

```text
configs/
  base.yaml
  env/
  task/
  trainer/
  renderer/
  agent/
  exp/
  evo/
  hydra/
scripts/
cambrian/
  main.py
  envs/
  agents/
  eyes/
  renderer/
  ml/
```

## Notes

Cambrian is highly config-driven: most behavior changes should be done by editing/overriding YAML rather than changing source code. Use the provided presets as templates and layer overrides for controlled experiments.


>>>>>>> 8fc50a08e825bb394f2c017204a9e77e03794462
