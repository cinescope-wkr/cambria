"""Step fns. These can be used to modify the observation and info dictionaries."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cambrian.agents import MjCambrianAgent
from cambrian.envs import MjCambrianEnv, MjCambrianMazeEnv

# ======================
# Helpers


def respawn_agent(
    env: MjCambrianMazeEnv,
    agent: MjCambrianAgent,
) -> Dict[str, Any]:
    """Respawn agent at given position."""
    agent.init_pos = env.maze.generate_reset_pos(agent.name)
    obs = agent.reset(env.spec)
    return obs


# ======================
# Step Functions


def step_respawn_agents_if_close_to_agents(
    env: MjCambrianMazeEnv,
    obs: Dict[str, Any],
    info: Dict[str, Dict[str, Any]],
    *,
    distance_threshold: float,
    for_agents: Optional[List[str]] = None,
    to_agents: Optional[List[str]] = None,
    from_agents: Optional[List[str]] = None,
):
    """

    Keywords Args:
        for_agents: List of agent names to check for proximity.
        to_agents: List of agent names to check distance to
        from_agents: List of agent names to check distance from
    """
    for agent_name, agent in env.agents.items():
        if for_agents is not None and agent_name not in for_agents:
            continue
        if from_agents is not None and agent_name not in from_agents:
            continue

        for other_agent_name, other_agent in env.agents.items():
            if to_agents is not None and other_agent_name not in to_agents:
                continue
            if agent_name == other_agent_name:
                continue

            info[agent_name]["respawned"] = False
            if np.linalg.norm(agent.pos - other_agent.pos) < distance_threshold:
                obs[agent_name] = respawn_agent(env, agent)
                info[agent_name]["respawned"] = True

    return obs, info


def step_add_agent_qpos_to_info(
    env: MjCambrianEnv,
    obs: Dict[str, Any],
    info: Dict[str, Dict[str, Any]],
    *,
    for_agents: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Add agent qpos to observation."""
    for agent_name, agent in env.agents.items():
        if for_agents is not None and agent_name not in for_agents:
            continue
        info[agent_name]["qpos"] = agent.qpos
    return obs, info


def step_combined(
    env: MjCambrianEnv,
    obs: Dict[str, Any],
    info: Dict[str, Dict[str, Any]],
    **step_fns,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Combine multiple step functions."""
    for step_fn in step_fns.values():
        obs, info = step_fn(env, obs, info)
    return obs, info
