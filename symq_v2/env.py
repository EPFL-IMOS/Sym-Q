import numpy as np
import gymnasium as gym
from gymnasium import spaces


from util import get_seq2action
from expression import AgentExpression


class SymbolicWorldEnv(gym.Env):
    def __init__(self, cfg):
        self.cfg = cfg
        self._max_step = cfg["max_step"]  # The maximal number of steps in an episode

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the operation.
        """
        self._seq2action = get_seq2action(cfg)
        self._action_to_operation = {v: k for k, v in self._seq2action.items()}

        # Observations
        self.observation_space = spaces.Dict(
            {
                "tree": spaces.MultiBinary(
                    [1, self._max_step, len(self._action_to_operation)]
                ),
            }
        )
        self.action_space = spaces.Discrete(len(self._action_to_operation))

        self._agent_expr = None  # The agent's expression

        self.n_step = 0
        self.tree = np.zeros((1, self._max_step, len(self._action_to_operation)))

    def get_obs(self):
        return {
            "tree": self.tree,
        }

    def get_info(self):
        return {
            "agent_expr": self._agent_expr,
            "terminated": None,
        }

    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Initialize the step counter
        self.n_step = 0
        self.tree = np.zeros((1, self._max_step, len(self._action_to_operation)))

        # agent expression
        self._agent_expr = AgentExpression()

        observation = self.get_obs()
        info = self.get_info()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3,...}) to the operation
        operation = self._action_to_operation[action]

        # Apply the operation to the agent expression
        all_replaced = self._agent_expr.add_opt(operation)

        # Buid the tree
        self.tree[0, self.n_step, action] = 1

        # Increment the step counter
        self.n_step += 1

        observation = self.get_obs()
        info = self.get_info()
        reward = 0
        terminated = False

        if all_replaced:
            terminated = True
            info["terminated"] = "all_replaced"

        elif self.n_step == self._max_step:
            terminated = True
            info["terminated"] = "time_out"

        return (
            observation,
            reward,
            terminated,
            False,
            info,
        )


if __name__ == "__main__":
    from util import load_cfg

    cfg = load_cfg("cfg_3var.yaml")
    env = SymbolicWorldEnv(cfg)
    obs, info = env.reset()
    print(obs)
    obs, _, done, _, info = env.step(1)
    obs, _, done, _, info = env.step(0)
    print(obs)
    print(info)
