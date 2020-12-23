
import numpy as np
from matplotlib import pyplot as plt
import deepSI

class Controller(object):
    def __init__(self): #todo
        pass

    def __call__(self,y):
        raise NotImplementedError

class Controller_pid(Controller):
    def __init__(self,kP,kD=0,kI=0):
        super(Controller_pid,self).__init__()
        self.kD = kD
        self.kP = kP
        self.kI = kI
        # self.r = np.zeros(self.N_samples) #can be an experiment?
        self.reset()

    def reset(self):
        self.t = 0
        self.E_old = None
        self.E_I = None

    def __call__(self,y):
        E = y-self.r[self.t]
        if self.E_old is None:
            self.E_old = E
            self.E_I = E
        else:
            self.E_I += E
        dE = E-self.E_old
        u = -self.kP*E-self.kD*dE-self.kI*self.E_I
        self.E_old = E
        self.t += 1
        return u

if __name__=='__main__':
    controller = Controller_pid(30,1000,0)
    controller.N_samples = 200
    # controller.r = np.sin(np.arange(controller.N_samples)/10)/2
    controller.r = (np.arange(controller.N_samples)>20)/5
    sys = deepSI.systems.Pendulum()
    data = sys.apply_controller(controller,N_samples=controller.N_samples)
    plt.plot(data.y)
    plt.plot(controller.r)
    # plt.plot(data.u)
    plt.show()
