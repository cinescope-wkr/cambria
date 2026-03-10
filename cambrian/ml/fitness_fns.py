"""Fitness functions for evaluating the fitness of agents. These functions are used by
the optimizers to evaluate the fitness of the agents. The fitness functions are
responsible for loading the evaluations and monitor files and calculating the fitness
of the agent based on the evaluations."""

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
from scipy.stats import zscore

if TYPE_CHECKING:
    from cambrian.config import MjCambrianConfig

# ========================
# Utils


def parse_evaluations_npz(evaluations_npz: Path) -> Dict[str, np.ndarray]:
    """Parse the evaluations npz file and return the rewards."""
    assert (
        evaluations_npz.exists()
    ), f"Evaluations file {evaluations_npz} does not exist."
    data = np.load(evaluations_npz, allow_pickle=True)
    return {k: data[k] for k in data}


def parse_monitor_csv(monitor_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse the monitor csv file and return the timesteps and rewards."""
    assert monitor_csv.exists(), f"Monitor file {monitor_csv} does not exist."
    timesteps, rewards = [], []
    with open(monitor_csv, "r") as f:
        # Skip the comment line
        f.readline()

        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            timesteps.append(float(row["t"]))
            rewards.append(float(row["r"]))

    return np.array(timesteps), np.array(rewards)


def top_n_percent(
    data: np.ndarray,
    percent: float,
    use_outliers: bool,
    max_zscore: float = 3.0,
) -> float:
    """Calculate the mean of the top `percent` of the data, optionally excluding
    outliers."""
    if not use_outliers:
        z_scores = zscore(data)
        filtered_data = data[abs(z_scores) < max_zscore]
        if len(filtered_data) == 0:
            filtered_data = data
        data = filtered_data

    n_top = int(len(data) * (percent / 100.0))
    return float(np.median(np.sort(data)[-n_top:]))


# ========================


def compute_vision_quality_components(
    image: np.ndarray,
    *,
    target_mean: float = 0.22,
    exposure_sigma: float = 0.10,
) -> Dict[str, float]:
    """Compute a compact no-reference image quality proxy for agent vision.

    The goal is not photographic fidelity; it is to estimate whether the retinal
    signal is usable for downstream behavior. We score:
    - exposure: whether the image is neither too dark nor too bright
    - contrast: whether there is tonal variation
    - detail: whether spatial structure survives the optics stack
    - clipping: whether the image avoids being mostly crushed or saturated
    """

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected image with shape [H, W, C], got {image.shape}.")
    if image.shape[-1] > 3:
        image = image[..., :3]
    if image.max() > 1.0:
        image = image / 255.0

    gray = np.clip(image.mean(axis=-1), 0.0, 1.0)
    mean_intensity = float(gray.mean())
    contrast = float(gray.std())

    grad_x = float(np.abs(np.diff(gray, axis=1)).mean()) if gray.shape[1] > 1 else 0.0
    grad_y = float(np.abs(np.diff(gray, axis=0)).mean()) if gray.shape[0] > 1 else 0.0
    detail = 0.5 * (grad_x + grad_y)

    dark_fraction = float((gray < 0.02).mean())
    bright_fraction = float((gray > 0.98).mean())

    exposure_score = float(
        np.exp(-((mean_intensity - target_mean) ** 2) / (2.0 * exposure_sigma**2))
    )
    contrast_score = float(np.tanh(contrast / 0.12))
    detail_score = float(np.tanh(detail / 0.06))
    clipping_score = float(max(0.0, 1.0 - 0.5 * (dark_fraction + bright_fraction)))

    return {
        "mean_intensity": mean_intensity,
        "contrast": contrast,
        "detail": detail,
        "dark_fraction": dark_fraction,
        "bright_fraction": bright_fraction,
        "exposure_score": exposure_score,
        "contrast_score": contrast_score,
        "detail_score": detail_score,
        "clipping_score": clipping_score,
    }


def vision_quality_score(components: Dict[str, float]) -> float:
    """Aggregate the image quality proxy into a single scalar fitness."""

    return float(
        0.35 * components["exposure_score"]
        + 0.25 * components["contrast_score"]
        + 0.30 * components["detail_score"]
        + 0.10 * components["clipping_score"]
    )


# ========================


def fitness_from_evaluations(
    config: "MjCambrianConfig",
    evaluations_npz: Path,
    *,
    return_data: bool = False,
    use_outliers: bool = False,
    percent: float = 25.0,
    quartiles: tuple[float, float] = (25, 75),
) -> float | Tuple[float, np.ndarray]:
    """
    Calculate the fitness of the agent based on evaluation results. The fitness is
    determined by taking the mean of the top `percent` of the evaluation rewards,
    optionally excluding outliers.

    Args:
        config (MjCambrianConfig): Configuration for the evaluation.
        evaluations_npz (Path): Path to the `.npz` file containing evaluation data.

    Keyword Args:
        return_data (bool): If True, returns the evaluation data along with the fitness.
        use_outliers (bool): If True, includes outliers in the fitness calculation.
            Defaults to False.
        percent (float): Defines the top 'n' percent of rewards to be used for
            calculating fitness. Defaults to 75.0.

    Returns:
        float | Tuple[float, np.ndarray]: The calculated fitness value. If `return_data`
        is True, returns a tuple containing the fitness value and the full evaluation
        data as a numpy array.

    Notes:
        - The rewards array should be a 2D array where each row represents an evaluation
          run and each column represents the rewards for each evaluation step.
        - If the evaluations file does not exist, returns negative infinity as the
          fitness value.
    """

    # Return negative infinity if the evaluations file doesn't exist
    if not evaluations_npz.exists():
        return -float("inf")

    # The rewards array will be stored in a 2D array where each row represents each
    # evaluation run and each column represents the rewards for each evaluation step.
    # We may run multiple steps of the same or slightly different environment to reduce
    # variance. We will average the rewards across each row to get the final rewards.
    evaluations = parse_evaluations_npz(evaluations_npz)
    rewards = evaluations["results"]
    rewards = np.mean(rewards, axis=1)

    fitness = top_n_percent(rewards, percent, use_outliers, quartiles)
    if return_data:
        return fitness, evaluations
    return top_n_percent(rewards, percent, use_outliers, quartiles)


def fitness_from_monitor(
    config: "MjCambrianConfig",
    monitor_csv: Path,
    *,
    return_data: bool = False,
    percent: float = 25.0,
    n_episodes: int = 1,
) -> float | Tuple[float, Tuple[np.ndarray, np.ndarray]]:
    """
    Calculate the fitness of the agent based on monitor data. The fitness is determined
    by taking the specified `percent` of the cumulative monitor rewards.

    Args:
        config (MjCambrianConfig): Configuration for the evaluation.
        monitor_csv (Path): Path to the CSV file containing monitor data.

    Keyword Args:
        return_data (bool): If True, returns the monitor data along with the fitness.
            Defaults to False.
        percent (float): Defines the top 'n' percent of rewards to be used for
            calculating fitness. Defaults to 75.0 (3rd quartile).
        n_episodes (int): The number of episodes to use sum rewards for. Defaults to 1.
            n_episodes must be a common factor of the number of episodes in the monitor
            data.

    Returns:
        float | Tuple[float, Tuple[np.ndarray, np.ndarray]]: The calculated fitness
        value. If `return_data` is True, returns a tuple containing the fitness value
        and the monitor data (timesteps, rewards) as numpy arrays.

    Notes:
        - If the rewards are empty, returns negative infinity as the fitness value.
        - The monitor data should contain timesteps and corresponding rewards.
    """
    timesteps, rewards = parse_monitor_csv(monitor_csv)

    if len(rewards) == 0:
        return -float("inf")

    assert len(rewards) % n_episodes == 0, (
        "n_episodes must be a common factor of the"
        " number of episodes in the monitor data."
    )
    timesteps = timesteps.reshape(-1, n_episodes).mean(axis=1)
    rewards = rewards.reshape(-1, n_episodes).sum(axis=1)

    fitness = top_n_percent(rewards, percent, use_outliers=False)
    if return_data:
        return fitness, (timesteps, rewards)
    return fitness


def fitness_from_txt(config: "MjCambrianConfig", txt_file: Path) -> float:
    """Calculate the fitness of the agent. Uses the 3rd quartile of the cumulative
    monitor rewards."""
    with open(txt_file, "r") as f:
        fitness = float(f.read().strip())

    return fitness


# ========================
# Fake eval fns which are used to test optimizers
# They should be realistic in the sense that they return a fitness value which is
# similar to the fitness value returned by the real eval fns. This means they should
# be fairly noisy and a similar correlation with the genotype as the real eval fns.


def fitness_num_eyes(
    config: "MjCambrianConfig",
    *,
    pattern: str,
    mean: float = 0,
    std: float = 5,
    assume_one: bool = True,
) -> float:
    """This fitness function will return higher rewards generally for agents with more
    eyes.

    Args:
        pattern (str): The path to the number of eyes in the config.
    """
    num_eyes = config.glob(pattern, flatten=True, assume_one=assume_one)

    # Set the seed such that when loading later, the same random values are generated
    # Get the seed based on the agent's id
    seed = config.seed
    if config.evo is not None:
        rank = config.evo.rank
        generation = config.evo.generation
        population_size = config.evo.population_size
        seed = generation * population_size + rank
    return np.random.default_rng(seed).normal(mean + np.prod(num_eyes), std)


def fitness_num_eyes_and_fov(
    config: "MjCambrianConfig",
    *,
    num_eyes_pattern: str,
    fov_pattern: str,
    mean: float = 0,
    std: float = 5,
    assume_one: bool = True,
    optimal_fov: float = 45,
) -> float:
    """This fitness function will return higher rewards generally for agents with more
    eyes and a fov closer to the `optimal_fov`.
    """
    num_eyes = config.glob(num_eyes_pattern, flatten=True, assume_one=assume_one)
    fov = config.glob(fov_pattern, flatten=True, assume_one=assume_one)

    # Set the seed such that when loading later, the same random values are generated
    # Get the seed based on the agent's id
    seed = config.seed
    if config.evo is not None:
        rank = config.evo.rank
        generation = config.evo.generation
        population_size = config.evo.population_size
        seed = generation * population_size + rank

    rng = np.random.default_rng(seed)
    num_eyes = rng.normal(mean + np.prod(num_eyes), std)
    fov = rng.normal(mean + optimal_fov / max(abs(optimal_fov - fov), 1), std)
    return num_eyes + fov
