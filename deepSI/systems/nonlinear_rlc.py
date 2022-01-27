# taken from:
# Model structures and fitting criteria for system
# identification with neural networks


import deepSI
from deepSI.systems.system import System, System_deriv, System_data
import numpy as np


class Nonlinear_rlc(System_deriv): #discrate system single system
    """docstring for nonlinear_RLC

    th,omega = x
    dth/dt = omega
    domega/dt = Tau / I - Fomega
    Tau = sum(r x F) = u - L m g sin(th)
    I = m L^2, 
    domega/dt = u -  g/L sin(th) - Fc omega


    """
    def __init__(self):
        '''Noise, system setting and x0 settings'''
        super(Nonlinear_rlc, self).__init__(nx=2,dt=5e-7)
        # self.dt = 5e-7 #0.5 mus #dt is quite short so prediction error will not perform
        self.L0 = 50e-6 #50 muH
        self.C = 270e-9 #270 nF
        self.R = 3 #3 Omh
        self.L = lambda iL: self.L0*((0.9/np.pi*np.arctan(-5*(abs(iL)-5))+0.5)+0.1)
        # self.settings = {**self.settings,**dict(name=self.name,omega0=omega0,Fc=Fc,lin=lin)} #todo seed?


    def deriv(self,x,u): #will be converted by Deriv system
        vC,iL = x
        vin = u
        L = self.L(iL)
        dvC_dt = iL/self.C
        diL_dt = (-vC - self.R*iL + vin)/L
        return [dvC_dt,diL_dt]

    def h(self,x,u):
        vC,iL = x
        return iL #or vC

    def get_train_data(self,N=4000):
        from scipy import signal
        band = 150e3
        order = 6
        self.b, self.a = signal.butter(order,band,analog=False,fs=1/self.dt)
        u0 = np.random.normal(scale=80,size=N)
        u = signal.lfilter(self.b,self.a,u0)
        exp = deepSI.System_data(u=u)
        return self.apply_experiment(exp)

    def get_test_data(self,N=4000):
        from scipy import signal
        band = 200e3
        order = 6
        self.b, self.a = signal.butter(order,band,analog=False,fs=1/self.dt)
        u0 = np.random.normal(scale=60,size=N)
        u = signal.lfilter(self.b,self.a,u0)
        exp = deepSI.System_data(u=u)
        return self.apply_experiment(exp)



if __name__=='__main__':
    from matplotlib import pyplot as plt
    sys = Nonlinear_rlc()
    train = sys.get_train_data()
    test = sys.get_test_data()
    # fit_sys = deepSI.fit_systems.NN_ARX_multi(nf=60)
    # fit_sys.fit(train,verbose=2,epochs=200)
    # test_predict = fit_sys.simulation(test)
    test.plot(show=False)
    # test_predict.plot(show=False)
    # plt.legend(['real',f'lin {test_predict.BFR(test)/100:.2%}'])
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
