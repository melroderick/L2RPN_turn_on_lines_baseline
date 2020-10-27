import numpy as np
from grid2op.Agent import BaseAgent


class TurnOnLinesAgent(BaseAgent):
    def __init__(self, action_space):
        BaseAgent.__init__(self, action_space=action_space)

    def act(self, observation, reward, done=False):
        action = {}
        
        # turn lines that are off back on
        lines_to_turn_on = np.argwhere(np.logical_and(np.logical_not(observation.line_status),
                                                      observation.time_before_cooldown_line == 0))
        if lines_to_turn_on.shape[0] > 0:
            action['change_line_status'] = (lines_to_turn_on[0, 0])

        return self.action_space(action)


def make_agent(env, this_directory_path):
    my_agent = TurnOnLinesAgent(env.action_space)
    return my_agent
