

import deepSI
from deepSI.systems.System import System, System_deriv, System_data
import numpy as np
from gym.spaces import Box

assert False, 'not yet implemented'

class ball_in_box_system(System): #discrate system single system
    """docstring for double_well_system

    V(x) = 1/2*min((x-a)**2,(x+a)**2)
    v' = -(x-a) if x>0 else (x+a) + u #+ resistance 
    x' = v
    Fmax < a
    Fmin > -a
    """
    def __init__(self, Nx=25, Ny=25):
        '''Noise, system setting and x0 settings'''
        self.Nx = Nx
        self.Ny = Ny
        super(ball_in_box_system, self).__init__(dt=dt,nx=2)
        self.action_space = Box(float(-1),float(1),shape=(2,))

    def reset(self):
        self.x = np.zeros((self.Nx, self.Ny)) #[x,y,vx,vy]
        return self.h(self.x) #return position

    def step(self,u):
        px, py = u
        self.x


    def h(self,x):
        return x