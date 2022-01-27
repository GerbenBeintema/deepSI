
from deepSI.fit_systems.fit_system import System_fittable
from deepSI.systems.system import System, System_io, System_data, load_system

class Sklearn_io(System_fittable, System_io):
    def __init__(self, na, nb, reg, feedthrough=False):
        super(Sklearn_io, self).__init__(na, nb,feedthrough=feedthrough)
        self.reg = reg

    def _fit(self, sys_data):
        #sys_data is already normed fitted on 
        hist, y = sys_data.to_IO_data(na=self.na,nb=self.nb,feedthrough=self.feedthrough)
        self.reg.fit(hist, y)

    def io_step(self,uy):
        return self.reg.predict([uy])[0] if uy.ndim==1 else self.reg.predict(uy)

from sklearn import linear_model 
class Sklearn_io_linear(Sklearn_io):
    def __init__(self,na,nb,feedthrough=False):
        super(Sklearn_io_linear,self).__init__(na,nb,linear_model.LinearRegression(),feedthrough)


if __name__ == '__main__':
    import numpy as np
    from matplotlib import pyplot as plt
    import deepSI
    sys = deepSI.systems.Wiener_sysid_book()
    sys_data = sys.apply_experiment(System_data(u=np.random.normal(scale=0.1,size=400)))
    sys = Sklearn_io_linear(na=2,nb=1)
    sys.fit(sys_data)


    # sys, score, kwargs = deepSI.fit_systems.fit_system_tuner(Sklearn_io_linear,sys_data,dict(na=range(1,10),nb=range(1,10)))
    # print(score,kwargs)
    # sys.apply_experiment(sys_data).plot()
    # sys_data.plot(show=True)
    print(f'len(sys_data)={len(sys_data)}')
    plt.plot(sys.n_step_error(sys_data))
    plt.show()
    sys.one_step_ahead(sys_data).plot(show=True)