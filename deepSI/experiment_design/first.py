

#  SISO experiment design

class experiment_generators(object):
    """docstring for experiment_generators"""
    def __init__(self, sys):
        super(experiment_generators, self).__init__()
        self.sys = sys
        self.data = []

if __name__ == '__main__':
    import gym
    space = gym.spaces.Box(-1,1,tuple())
    print(type(space.sample()))