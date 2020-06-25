
import deepSI
from deepSI.systems.System import System_SS, System_data, System_Deriv
import numpy as np


class nonlindrag_sys(System_Deriv):
    def __init__(self,dt=1, Fdrag = lambda v: -0.1*v):
        super(nonlindrag_sys,self).__init__(dt=dt,nx=1)
        self.Fdrag = Fdrag

    def deriv(self, x, u):
        v, = x
        return (self.Fdrag(v) + u,)

    def h(self, x):
        return x[0]

    def get_train_data(self):
        exp = System_data(u=self.random.uniform(-1,1,size=10**5))
        return self.apply_experiment(exp)

    def get_test_data(self):
        exp = System_data(u=self.random.uniform(-1,1,size=10**4))
        return self.apply_experiment(exp)
    
class CED_sim(nonlindrag_sys):
    def __init__(self):
        import math
        sign = lambda v: 1 if v>0 else (-1 if v<0 else 0)
        super(CED_sim,self).__init__(Fdrag = lambda v: -0.1*v-0.2*sign(v)*math.exp(-abs(v)/1))

    def h(self, x):
        return abs(x[0])

if __name__=='__main__':
    sys_sim = CED_sim()
    train_full,test = sys_sim.get_train_data(), sys_sim.get_test_data()
    train_full.plot(show=True)