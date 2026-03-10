"""Cambrian package init file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Optional

import cambrian.agents  # noqa
import cambrian.envs  # noqa
import cambrian.eyes  # noqa
from cambrian.config import MjCambrianConfig  # noqa
from cambrian.ml.trainer import MjCambrianTrainer  # noqa


def run_hydra(
    main_fn: Optional[Callable[..., Any]] = lambda *_, **__: None,
    /,
    *,
    parser: Optional[argparse.ArgumentParser] = None,
    config_path: Path | str | None = None,
    config_name: str = "base",
    instantiate: bool = True,
    **kwargs: Any,
):
    """Project default wrapper around :func:`hydra_config.run_hydra`.

    Notes:
        The upstream helper uses :func:`hydra.main`, whose ``config_path`` semantics
        differ across Hydra versions. To be robust (and avoid filesystem path issues),
        we default to loading configs from the installed package via
        ``pkg://cambrian.configs``.
    """

    from hydra_config import run_hydra as _run_hydra

    if parser is None:
        parser = argparse.ArgumentParser()

    if config_path is None:
        config_path = "pkg://cambrian.configs"

    normalized_config_path: str
    if isinstance(config_path, Path):
        config_dir = config_path.expanduser()
        if not config_dir.is_absolute():
            config_dir = config_dir.resolve()
        if not config_dir.is_dir():
            raise FileNotFoundError(f"Hydra config directory not found: {config_dir}")
        normalized_config_path = f"file://{config_dir}"
    else:
        if config_path.startswith(("pkg://", "file://")):
            normalized_config_path = config_path
        else:
            # Interpret as a filesystem path (absolute or relative to this package).
            candidate = Path(config_path).expanduser()
            if not candidate.is_absolute():
                candidate = (Path(__file__).resolve().parent / candidate).resolve()
            if not candidate.is_dir():
                raise FileNotFoundError(
                    f"Hydra config directory not found: {candidate}. "
                    "Pass a directory path, or use e.g. `pkg://cambrian.configs`."
                )
            normalized_config_path = f"file://{candidate}"

    return _run_hydra(
        main_fn,
        parser=parser,
        config_path=normalized_config_path,
        config_name=config_name,
        instantiate=instantiate,
        **kwargs,
    )

__author__ = "Camera Culture (a2cc@media.mit.edu)"
"""Camera Culture (a2cc@media.mit.edu)"""
__license__ = "BSD3"
"""BSD3"""
