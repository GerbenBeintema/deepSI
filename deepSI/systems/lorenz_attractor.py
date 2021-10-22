
# Model structures and fitting criteria for system
# identification with neural networks

import deepSI
from deepSI.systems.system import System, System_deriv, System_data
import numpy as np


class Lorenz_attractor(System_deriv): #discrate system single system
    """
    Description:
        lorenz attractor learning envoriement
        x' = sigma*(y-x) + a
        y' = x*(rho-z) - y
        z' = x*y - beta*z
        with as goal maximizing the number of x-crossings with the control a 
        with as constraint that |a|<=1 

    Source:
        Gerben Beintema from TU/e

    Observation: 
        Type: Box(3,)
        Num Observation                 Min         Max
        0   current x                  -30          30
        1   current y                  -40          40
        2   current z                    0          70

    Actions:
        Type: Discrete(3) or continues Box(1,) (min=-1, max=1)
        If Discrate(3): 
            action = 0 -> a = -1
            action = 1 -> a = 0
            action = 2 -> a = 1
        if continues ie Box(1,):
            a = action (with -1<=action<=1)

    Reward:
        1 for every crossing of the x-axis else 0

    Starting State:
        random position on the uncontrolled lorenz attractor (done by taking 500 steps)

    Episode Termination:
        1000 cycles

    """
    def __init__(self,sigma=10,beta=8/3.,rho=28):
        '''Noise, system setting and x0 settings'''
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        super(Lorenz_attractor, self).__init__(nx=3,dt=0.05)
        # self.dt = 5e-7 #0.5 mus #dt is quite short so prediction error will not perform

        # self.settings = {**self.settings,**dict(name=self.name,omega0=omega0,Fc=Fc,lin=lin)} #todo seed?

    def reset(self):
        x = np.random.uniform(low=[-20,-25,0.9],high=[20,25,50],size=(3,))
        for i in range(500):
            x = self.f(x,0)
        self.x = x
        return self.h(self.x)

    def h(self,x):
        return x[0]

    def deriv(self,x,u): #will be converted by Deriv system
        '''
        x' = sigma*(y-x) + a
        y' = x*(rho-z) - y (+ b)
        z' = x*y - beta*z
        '''
        x,y,z = x
        xp = self.sigma*(y-x) 
        yp = x*(self.rho-z)-y + u
        zp = x*y-self.beta*z
        return np.array([xp,yp,zp])

class Lorenz_attractor_sincos(Lorenz_attractor):
    def h(self,x):
        x,y,z = x

        A = x*np.sin(x)
        B = y*np.cos(y)
        C = z*np.sin(z)
        return A,B,C


if __name__=='__main__':
    from matplotlib import pyplot as plt
    sys = Lorenz_attractor()
    train = sys.get_train_data()
    test = sys.get_test_data()
    fit_sys = deepSI.fit_systems.System_Torch_IO(na=5,nb=5)
    fit_sys.fit(train,verbose=2,epochs=200)
    test_predict = fit_sys.simulation(test)
    test.plot(show=False)
    test_predict.plot(show=False)
    plt.legend(['real',f'lin {test_predict.BFR(test)/100:.2%}'])
    plt.show()

    # # iLtest = np.linspace(0,20,num=200)
    # # plt.plot(iLtest,sys.L(iLtest))
    # # plt.show()
    # from scipy import signal
    # band = 150e3
    # order = 6
    # self.b, self.a = signal.butter(order,band,analog=False,fs=1/self.dt)
    # u0 = np.random.normal(scale=80,size=4000)
    # u = signal.lfilter(self.b,self.a,u0)
    # from scipy.fftpack import *
    # exp.plot()

    # U = fft(u)/len(u)
    # feq = fftfreq(len(u),d=self.dt)
    # plt.plot(feq,abs(U),'.')
    # ylim = plt.ylim()
    # plt.plot([band,band],plt.ylim())
    # plt.ylim(ylim)
    # plt.show()
