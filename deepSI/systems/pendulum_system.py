
import deepSI
from deepSI.systems.System import System, System_Deriv, System_data
import numpy as np


class pendulum_system(System_Deriv): #discrate system single system
    """docstring for pendulum_system

    th,omega = x
    dth/dt = omega
    domega/dt = Tau / I - Fomega
    Tau = sum(r x F) = u - L m g sin(th)
    I = m L^2, 
    domega/dt = u -  g/L sin(th) - Fc omega


    """
    def __init__(self, dt=None, omega0 = 1, Fc=0.1,lin=False,):
        '''Noise, system setting and x0 settings'''
        dt = min(Fc/5,1/omega0/5) if dt is None else dt
        super(pendulum_system, self).__init__(dt=dt,nx=2)
        self.omega0 = omega0 #tau = 1/omega = 1
        self.Fc = Fc #tau = 10
        self.lin = lin


    def deriv(self,x,u): #will be converted by 
        th,omega = x
        dthdt = omega
        if self.lin:
            domegadt = -self.omega0**2*th + u - self.Fc*omega
        else:
            domegadt = -self.omega0**2*np.sin(th) + u - self.Fc*omega
        return [dthdt,domegadt]

    def h(self,x):
        th,omega = x
        return th

if __name__=='__main__':
    from matplotlib import pyplot as plt
    sys = pendulum_system()
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
    # exp = uxyeye.Experiment(u=u)
    # from scipy.fftpack import *
    # exp.plot()

    # U = fft(u)/len(u)
    # feq = fftfreq(len(u),d=self.dt)
    # plt.plot(feq,abs(U),'.')
    # ylim = plt.ylim()
    # plt.plot([band,band],plt.ylim())
    # plt.ylim(ylim)
    # plt.show()
