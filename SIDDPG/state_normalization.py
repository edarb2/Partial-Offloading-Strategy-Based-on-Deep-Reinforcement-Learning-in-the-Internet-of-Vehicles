import numpy as np
from mec_state import MEC_STATE


class StateNormalization1(object):
    def __init__(self):
        env = MEC_STATE()
        M = env.M
        self.high_state = np.append(np.ones(M) * 1000 * 5, np.ones(M) * 262144 )
        self.high_state = np.append(self.high_state, np.ones(M) * 262144)

        # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        # self.high_state = np.array(
        #     [500000, 100, 100, 60 * 1048576, 100, 100, 100, 100, 100, 100, 100, 100, 2097152, 2097152, 2097152, 2097152,
        #      1, 1, 1, 1])  # uav loc, ue loc, task size, block_flag
        self.low_state = np.append(np.zeros(M),np.ones(M) * 109715)
        self.low_state = np.append(self.low_state, np.ones(M) * 109715)
        #self.low_state = np.zeros(3 * M )  # uav loc, ue loc, task size, block_flag

    def state_normal(self, state):
        return (state - self.low_state) / (self.high_state - self.low_state)

class StateNormalization2(object):
    def __init__(self):
        env = MEC_STATE()
        M = env.M
        self.high_state = np.append(np.ones(M) * 262144, np.ones(M) * 262144)

        # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        # self.high_state = np.array(
        #     [500000, 100, 100, 60 * 1048576, 100, 100, 100, 100, 100, 100, 100, 100, 2097152, 2097152, 2097152, 2097152,
        #      1, 1, 1, 1])  # uav loc, ue loc, task size, block_flag
        #self.low_state = np.zeros(2 * M)  # uav loc, ue loc, task size, block_flag
        self.low_state = np.append(np.ones(M) * 109715, np.ones(M) * 109715)

    def state_normal(self, state):
        return (state - self.low_state) / (self.high_state - self.low_state)
