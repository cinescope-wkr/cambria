"""Defines a static agent which does not learn. This is useful for defining objects in
the environment."""

from typing import TYPE_CHECKING, List

from cambrian.agents.agent import MjCambrianAgent2D, MjCambrianAgentConfig

if TYPE_CHECKING:
    from cambrian.envs.env import MjCambrianEnv


class MjCambrianAgentObject(MjCambrianAgent2D):
    """This is a class which defines an object agent. An object agent is
    essentially a non-trainable agent. It's simply an object in the environment which
    has no observations or actions.
    """

    def __init__(
        self,
        config: MjCambrianAgentConfig,
        name: str,
    ):
        assert not config.trainable, "Object agents cannot be trainable"
        super().__init__(config, name)

    def get_action_privileged(self, _: "MjCambrianEnv") -> List[float]:
        return []
