

import deepSI
from deepSI.systems.System import System_ss, System_data
import numpy as np

class NarendraLiBenchmark(System_ss): #https://arxiv.org/pdf/2003.14162.pdf
    """docstring for test_system"""
    def __init__(self):
        '''Noise, system setting and x0 settings'''
        super(NarendraLiBenchmark, self).__init__(nx=2)

    def f(self,x,u):
        x1,x2 = x
        x1new = (x1/(1+x1**2)+1)*np.sin(x2)
        x2new = x2*np.cos(x2) + x1*np.exp(-(x1**2+x2**2)/8) + u**3/(1+u**2+0.5*np.cos(x1+x2))
        return [x1new,x2new]

    def h(self,x):
        x1,x2 = x
        return x1/(1+0.5*np.sin(x2)) + x2/(1+0.5*np.sin(x1)) + self.random.normal(scale=0.1)

    def get_train_data(self):
        exp = System_data(u=self.random.uniform(low=-2.5,high=2.5,size=(2000,)))
        return self.apply_experiment(exp)

    def get_test_data(self):
        exp = System_data(u=self.random.uniform(low=-2.5,high=2.5,size=(2000,)))
        return self.apply_experiment(exp)

if __name__ == '__main__':
    from deepSI import fit_systems
    sys = NarendraLiBenchmark()
    sys_data = sys.get_train_data()

    SYS = fit_systems.System_IO_fit_linear
    # sys_fit, score, kwargs = fit_systems.fit_system_tuner(SYS, sys_data, dict(na=range(0,7),nb=range(1,7)))
    score, sys_fit, kwargs, _ = fit_systems.grid_search(SYS, sys_data, dict(na=range(0,7),nb=range(1,7)))
    sys_data_predict = sys_fit.apply_experiment(sys_data)
    sys_data.plot()
    sys_data_predict.plot(show=True)
