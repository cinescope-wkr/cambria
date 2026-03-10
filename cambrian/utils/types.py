"""Defines the types used in ``cambrian`` package."""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Concatenate,
    Dict,
    Self,
    Tuple,
    TypeAlias,
)

import numpy as np
import torch

if TYPE_CHECKING:
    from cambrian.agents.agent import MjCambrianAgent
    from cambrian.envs.env import MjCambrianEnv

# ======================

ObsType: TypeAlias = np.ndarray | torch.Tensor | Dict[str, Self]
RewardType: TypeAlias = float | Dict[str, Self]
TerminatedType: TypeAlias = bool | Dict[str, Self]
TruncatedType: TypeAlias = bool | Dict[str, Self]
InfoType: TypeAlias = Dict[str, Any] | Dict[str, Self]
ActionType: TypeAlias = np.ndarray | torch.Tensor | Dict[str, Self]
RenderFrame: TypeAlias = torch.Tensor | np.ndarray | Dict[str, Self] | None

# ======================

MjCambrianStepFn: TypeAlias = Callable[
    [Concatenate["MjCambrianEnv", ObsType, InfoType, ...]],
    Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]],
]

MjCambrianTerminationFn: TypeAlias = Callable[
    Concatenate["MjCambrianEnv", "MjCambrianAgent", InfoType, ...],
    bool,
]

MjCambrianTruncationFn: TypeAlias = Callable[
    Concatenate["MjCambrianEnv", "MjCambrianAgent", InfoType, ...],
    bool,
]

MjCambrianRewardFn: TypeAlias = Callable[
    Concatenate[
        "MjCambrianEnv",
        "MjCambrianAgent",
        TerminatedType,
        TruncatedType,
        InfoType,
        ...,
    ],
    float,
]

# ======================

__all__ = [
    "ObsType",
    "RewardType",
    "TerminatedType",
    "TruncatedType",
    "InfoType",
    "ActionType",
    "RenderFrame",
    "MjCambrianStepFn",
    "MjCambrianTerminationFn",
    "MjCambrianTruncationFn",
    "MjCambrianRewardFn",
]
