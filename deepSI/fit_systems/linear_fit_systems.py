
import deepSI
from deepSI.system_data.System_data import System_data,System_data_list

from deepSI.fit_systems.Fit_system import System_IO_fit_sklearn, System_fittable


from sklearn import linear_model 
class System_IO_fit_linear(System_IO_fit_sklearn):
    def __init__(self,na,nb):
        super(System_IO_fit_linear,self).__init__(na,nb,linear_model.LinearRegression())


if __name__ == '__main__':
    import numpy as np
    from matplotlib import pyplot as plt
    sys = deepSI.systems.Wiener_sys_ID_book()
    sys_data = sys.apply_experiment(System_data(u=np.random.normal(scale=0.1,size=400)))
    sys = System_IO_fit_linear(na=2,nb=1)
    sys.fit(sys_data)


    # sys, score, kwargs = deepSI.fit_systems.fit_system_tuner(System_IO_fit_linear,sys_data,dict(na=range(1,10),nb=range(1,10)))
    # print(score,kwargs)
    # sys.apply_experiment(sys_data).plot()
    # sys_data.plot(show=True)
    print(f'len(sys_data)={len(sys_data)}')
    plt.plot(sys.n_step_error(sys_data))
    plt.show()
    sys.one_step_ahead(sys_data).plot(show=True)
