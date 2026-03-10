"""Point agents."""

from functools import cached_property
from typing import Optional, Tuple

import numpy as np
from gymnasium import spaces

from cambrian.agents.agent import MjCambrianAgent2D, MjCambrianAgentConfig
from cambrian.envs.maze_env import MjCambrianMazeEnv
from cambrian.utils import get_logger
from cambrian.utils.types import ActionType, ObsType


class MjCambrianAgentPoint(MjCambrianAgent2D):
    """
    This is a hardcoded class which implements the agent as actuated by a forward
    velocity and a rotational position. In mujoco, to the best of my knowledge, all
    translational joints are actuated in reference to the _global_ frame rather than
    the local frame. This means a velocity actuator applied along the x-axis will move
    the agent along the global x-axis rather than the local x-axis. Therefore, the
    agent will have 3 actuators: two for x and y global velocities and one for
    rotational position. From the perspective the calling class (i.e. MjCambrianEnv),
    this agent has two actuators: a forward velocity and a rotational position. We will
    calculate the global velocities and rotational position from these two "actuators".

    Todo:
        Will create an issue on mujoco and see if it's possible to implement this in
        xml. The issue right now is that mujoco doesn't support relative positions for
        hinge joints, so we have to implement the heading joint as a velocity actuator
        which is not ideal.
    """

    def __init__(
        self,
        config: MjCambrianAgentConfig,
        name: str,
        *,
        kp: float = 0.75,
    ):
        super().__init__(config, name)

        self._kp = kp

        assert np.all(self._actuators[0].ctrlrange == self._actuators[1].ctrlrange), (
            f"Forward velocity and lateral velocity must have the same control range, "
            f"got {self._actuators[0].ctrlrange} and {self._actuators[1].ctrlrange}"
        )
        self._v_ctrlrange = np.array([0, 1])
        self._theta_ctrlrange = self._actuators[2].ctrlrange

    def _update_obs(self, obs: ObsType) -> ObsType:
        """Creates the entire obs dict."""
        obs = super()._update_obs(obs)

        # Update the action obs
        # Calculate the global velocities
        if self._config.use_action_obs:
            v, theta = self._calc_v_theta(self._last_action)
            v = np.interp(v, self._v_ctrlrange, [-1, 1])
            theta = np.interp(theta, self._theta_ctrlrange, [-1, 1])
            obs["action"] = np.array([v, theta], dtype=np.float32)

        return obs

    def _calc_v_theta(self, action: Tuple[float, float, float]) -> Tuple[float, float]:
        """Calculates the v and theta from the action."""
        vx, vy, _ = action
        v = np.hypot(vx, vy)
        theta = np.arctan2(vy, vx) - self.qpos[2]
        return v, theta

    def apply_action(self, action: ActionType):
        """Calls the appropriate apply action method based on the heading joint type."""
        assert len(action) == 2, f"Action must have two elements, got {len(action)}."

        # Calculate global velocities
        v = np.interp(action[0], [-1, 1], self._v_ctrlrange)
        current_heading = self.qpos[2]
        vx = v * np.cos(current_heading)
        vy = v * np.sin(current_heading)

        super().apply_action([vx, vy, action[1]])

    @cached_property
    def action_space(self) -> spaces.Space:
        """Overrides the base implementation to only have two elements."""
        return spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)


class MjCambrianAgentPointSeeker(MjCambrianAgentPoint):
    """This is an agent which is non-trainable and defines a custom policy which
    acts as a 'homing' agent in the maze environment. This agent will attempt to reach
    a target (which is either randomly placed or another specific agent) by taking
    actions that best map to the optimal trajectory which is calculated from the maze
    using bfs (or just by choosing the action which minimizes the distance to
    the target).

    Keyword Args:
        target (Optional[str]): The name of the target agent to home in on. If
            None, a random free space in the maze will be chosen as the target.
        speed (float): The speed at which the agent moves. Defaults to -0.75.
        distance_threshold (float): The distance threshold at which the agent will
            consider itself to have reached the target. Defaults to 2.0.
        use_optimal_trajectory (bool): Whether to use the optimal trajectory to the
            target. Defaults to False.
    """

    def __init__(
        self,
        config: MjCambrianAgentConfig,
        name: str,
        *,
        target: Optional[str],
        speed: float = -0.75,
        distance_threshold: float = 2.0,
        use_optimal_trajectory: bool = False,
    ):
        super().__init__(config, name)

        self._target = target
        self._speed = speed
        self._distance_threshold = distance_threshold

        self._optimal_trajectory: np.ndarray = None
        self._use_optimal_trajectory = use_optimal_trajectory

        self._prev_target_pos: np.ndarray = None

    def reset(self, *args) -> ObsType:
        """Resets the optimal_trajectory."""
        self._optimal_trajectory = None
        self._prev_target_pos = None
        return super().reset(*args)

    def get_action_privileged(self, env: MjCambrianMazeEnv) -> ActionType:
        if self._target is None:
            if self._optimal_trajectory is None or len(self._optimal_trajectory) == 0:
                # Generate a random position to navigate to
                # Chooses one of the empty spaces in the maze
                rows, cols = np.where(env.maze.map == "0")
                assert rows.size > 0, "No empty spaces in the maze"
                index = np.random.randint(rows.size)
                target_pos = env.maze.rowcol_to_xy((rows[index], cols[index]))
            else:
                target_pos = self._optimal_trajectory[0]
        else:
            assert self._target in env.agents, f"Target {self._target} not found in env"
            target_pos = env.agents[self._target].pos

        if self._prev_target_pos is None:
            self._prev_target_pos = target_pos

        # Calculate the optimal trajectory if the current trajectory is None
        if (
            self._optimal_trajectory is None
            or np.linalg.norm(target_pos - self._prev_target_pos) > 0.1
        ):
            if self._use_optimal_trajectory:
                obstacles = []
                for agent_name, agent in env.agents.items():
                    if agent_name != self._target:
                        obstacles.append(tuple(env.maze.xy_to_rowcol(agent.pos)))
                try:
                    self._optimal_trajectory = env.maze.compute_optimal_path(
                        self.pos, target_pos, obstacles=obstacles
                    )
                except IndexError:
                    # Happens if there's no path to the target
                    get_logger().warning(
                        f"No path to target {target_pos} from {self.pos}"
                    )
                    self._optimal_trajectory = np.array([target_pos[:2]])
            else:
                self._optimal_trajectory = np.array([target_pos[:2]])

        # If the optimal trajectory is empty, then set the optimal trajectory to the
        # target position
        if len(self._optimal_trajectory) == 0:
            self._optimal_trajectory = np.array([target_pos[:2]])

        # Get the current target. If the distance between the current position and the
        # target is less than the threshold, then remove the target from the optimal
        # trajectory
        target = self._optimal_trajectory[0]
        target_vector = target - self.pos[:2]
        distance = np.linalg.norm(target - self.pos[:2])
        if distance < self._distance_threshold:
            self._optimal_trajectory = self._optimal_trajectory[1:]

        # Update the previous target position
        target_theta = np.arctan2(target_vector[1], target_vector[0])
        theta_action = np.interp(target_theta, [-np.pi, np.pi], [-1, 1])

        return [self._speed, theta_action]
