"""Done fns. These can be used either with termination or truncation."""

from typing import Any, Dict, List, Optional

import numpy as np

from cambrian.agents import MjCambrianAgent
from cambrian.envs import MjCambrianEnv
from cambrian.utils import agent_selected

# ======================


def done_never(
    env: MjCambrianEnv, agent: MjCambrianAgent, info: Dict[str, Any]
) -> bool:
    """Never done."""
    return False


def done_if_exceeds_max_episode_steps(
    env: MjCambrianEnv, agent: MjCambrianAgent, info: Dict[str, Any]
) -> bool:
    """Done if episode step exceeds max episode steps."""
    return env.episode_step >= env.max_episode_steps - 1


def done_if_low_reward(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    info: Dict[str, Any],
    *,
    threshold: float,
    disable: bool = False,
) -> bool:
    """Done if agent has low reward."""
    if disable:
        return False

    return env.cumulative_reward < threshold


def done_if_has_contacts(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    info: Dict[str, Any],
    *,
    for_agents: Optional[List[str]] = None,
    disable: bool = False,
) -> bool:
    """Done if agent has contacts."""
    if not agent_selected(agent, for_agents) or disable:
        return False

    return info["has_contacts"]


def done_if_close_to_agents(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    info: Dict[str, Any],
    *,
    to_agents: Optional[List[str]] = None,
    for_agents: Optional[List[str]] = None,
    distance_threshold: float,
    disable: bool = False,
) -> bool:
    """Done if agent is close to another agent."""
    # Early exit if the agent is not in the for_agents list
    if not agent_selected(agent, for_agents) or disable:
        return False

    for other_agent in env.agents.values():
        if not agent_selected(other_agent, to_agents) or other_agent.name == agent.name:
            continue

        if np.linalg.norm(other_agent.pos - agent.pos) < distance_threshold:
            return True
    return False


def done_combined(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    info: Dict[str, Any],
    **done_fns,
) -> bool:
    """Combine multiple done functions."""
    return any(done_fn(env, agent, info) for done_fn in done_fns.values())
