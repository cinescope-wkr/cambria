"""The configuration module for the ``cambrian`` module."""

from pathlib import Path
from typing import Any, List, Optional

from hydra_config import HydraContainerConfig, config_wrapper, register_new_resolver
from omegaconf import DictConfig, OmegaConf

from cambrian.envs.env import MjCambrianEnvConfig
from cambrian.ml.evo import MjCambrianEvoConfig
from cambrian.ml.trainer import MjCambrianTrainerConfig
from cambrian.utils import is_integer

# =============================================================================


@config_wrapper
class MjCambrianConfig(HydraContainerConfig):
    """The base config for the mujoco cambrian environment. Used for type hinting.

    Attributes:
        logdir (Path): The primary directory which simulation data is stored in. This is
            the highest level directory used for the experiment. `expdir` is the
            subdirectory used for a specific experiment.
        expdir (Path): The directory used for a specific experiment. This is the
            directory where the experiment's data is stored. Should evaluate to
            `logdir / `expsubdir`
        expsubdir (Path): The subdirectory relative to logdir where the experiment's
            data is stored. This is the directory where the experiment's data is stored.
        expname (str): The name of the experiment. Used to name the logging
            subdirectory.

        seed (int): The base seed used when initializing the default thread/process.
            Launched processes should use this seed value to calculate their own seed
            values. This is used to ensure that each process has a unique seed.

        training (MjCambrianTrainingConfig): The config for the training process.
        env (MjCambrianEnvConfig): The config for the environment.
        eval_env (MjCambrianEnvConfig): The config for the evaluation environment.
    """

    logdir: Path
    expdir: Path
    expsubdir: Path
    expname: Any

    seed: int

    trainer: MjCambrianTrainerConfig
    evo: Optional[MjCambrianEvoConfig] = None
    env: MjCambrianEnvConfig
    eval_env: MjCambrianEnvConfig | Any


# =============================================================================


@register_new_resolver("package")
def package_resolver(package: str = "cambrian") -> Path:
    """Get the path to installed package directory."""
    import importlib.util

    package_path = importlib.util.find_spec(package).submodule_search_locations[0]
    return Path(package_path)


@register_new_resolver("clean_overrides")
def clean_overrides_resolver(
    overrides: List[str],
    ignore_after_override: Optional[str] = "",
    use_seed_as_subfolder: bool = True,
) -> str:
    cleaned_overrides: List[str] = []

    seed: Optional[Any] = None
    for override in overrides:
        if "=" not in override or override.count("=") > 1:
            continue
        if ignore_after_override and override == ignore_after_override:
            break

        key, value = override.split("=", 1)
        if key == "exp":
            continue
        if key == "seed" and use_seed_as_subfolder:
            seed = value
            continue

        # Special key cases that we want the second key rather than the first
        if (
            key.startswith("env.reward_fn")
            or key.startswith("env.truncation_fn")
            or key.startswith("env.termination_fn")
            or key.startswith("env.step_fn")
            or is_integer(key.split(".")[-1])
        ):
            key = "_".join(key.split(".")[-2:])
        else:
            key = key.split("/")[-1].split(".")[-1]

        # Clean the key and value
        key = (
            key.replace("+", "")
            .replace("[", "")
            .replace("]", "")
            .replace(",", "_")
            .replace(" ", "")
        )
        value = (
            value.replace(".", "p")
            .replace("-", "n")
            .replace("[", "")
            .replace("]", "")
            .replace(",", "_")
            .replace(" ", "")
        )

        cleaned_overrides.append(f"{key}_{value}")

    return "_".join(cleaned_overrides) + (f"/seed_{seed}" if seed is not None else "")


@register_new_resolver("num_cpus")
def num_cpus_resolver() -> int:
    import multiprocessing

    return multiprocessing.cpu_count()


@register_new_resolver("unresolved")
def unresolved_resolver(path: str, *, _parent_: DictConfig) -> str:
    return OmegaConf.to_container(OmegaConf.select(_parent_, path))
