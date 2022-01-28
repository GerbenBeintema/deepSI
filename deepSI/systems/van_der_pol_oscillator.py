
import deepSI
from deepSI.systems.system import System, System_deriv, System_data
import numpy as np

class Van_der_pol_oscillator(System_deriv):
    """docstring for Van_der_pol_oscillator"""
    def __init__(self, dt=0.2,mu=2.5):
        self.mu = mu
        super(Van_der_pol_oscillator, self).__init__(nx=2,dt=dt)

    def reset_state(self):
        x = np.random.uniform(low=[-3,-3],high=[3,3],size=(2,))
        for i in range(500):
            x = self.f(x,0)
        self.x = x

    def h(self,x,u):
        return x[0]

    def deriv(self,state,u): #will be converted by Deriv system
        x, y = state #unpack
        xp = self.mu*(x-x**3/3-y)
        yp = 1/self.mu*(x-u)
        return np.array([xp,yp])


if __name__ == '__main__':
    sys = Van_der_pol_oscillator()
    sys_data = sys.apply_experiment(System_data(u=0*np.sin(np.linspace(0,2*np.pi*20,num=10000))))
    sys_data[:1000].plot(show=True)