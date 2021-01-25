import deepSI
import numpy as np
from matplotlib import pyplot as plt

class My_system(deepSI.System_ss): 
    def __init__(self):
        super(My_system,self).__init__(nx=2)
    def f(self, x, u):
        return x[0]/(1.2+x[1]**2)+x[1]*0.4, x[1]/(1.2+x[0]**2)+x[0]*0.4+u #some non-linear function
    def h(self, x):
        return x[0]+self.random.normal(scale=0.1,size=())


if __name__ == '__main__':
    sys = My_system()

    train = deepSI.System_data(u=np.random.uniform(low=-2,high=2,size=10**4))
    train = sys.apply_experiment(train) #fill train.y
    test = deepSI.System_data(u=np.random.uniform(low=-2,high=2,size=5*10**3))
    test = sys.apply_experiment(test) #fill train.y

    fit_sys_ss_lin = deepSI.fit_systems.SS_linear(nx=2)
    fit_sys_ss_lin.fit(train, SS_f=20)

    test_lin = fit_sys_ss_lin.apply_experiment(test)

    plt.plot(test.y,label='real')
    plt.plot(test_lin.y-test.y,label='diff')
    plt.legend()
    plt.show()
    print(f'NRMS = {test_lin.NRMS(test):.2%}')

    fit_sys_encoder = deepSI.fit_systems.SS_encoder(nx=4,na=10,nb=10)
    fit_sys_encoder.fit(train[1000:],sim_val=train[:1000])

    test_encoder = fit_sys_encoder.apply_experiment(test)
    plt.plot(test.y,label='real')
    plt.plot(test_lin.y-test.y,label='diff lin')
    plt.plot(test_encoder.y-test.y,label='diff encoder')
    plt.legend()
    plt.show()
    print(f'ss linear NRMS = {test_lin.NRMS(test):.2%}')
    print(f'encoder NRMS = {test_encoder.NRMS(test):.2%}')