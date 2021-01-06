
#! /usr/bin/env python3
#
# This is an example  shows application of linear system identification





import deepSI
import numpy as np
from matplotlib import pyplot as plt


def linear_SI_example():

    
    ## Generate system
    A = np.array([[0.89, 0.], [0., 0.45]]) #(2,2) x<-x
    B = np.array([[0.3], [2.5]]) #(2,1) x<-u
    C = np.array([[0.7, 1.]]) #(1,2) y<-u
    D = np.array([[0.0]]) #(1,1) 
    sys0 = deepSI.fit_systems.SS_linear(A=A,B=B,C=C,D=D)

    ## Generate experiment
    sys_data_train = deepSI.System_data(u=np.random.normal(size=1000))
    sys_data_test = deepSI.System_data(u=np.random.normal(size=1000))

    ## apply experiment
    sys_data_train = sys0.apply_experiment(sys_data_train)
    sys_data_train.y += np.random.normal(scale=1e-2,size=1000) #add a bit of noise to the training set
    sys_data_test = sys0.apply_experiment(sys_data_test)


    #create linear fit systems
    fit_sys_SS = deepSI.fit_systems.SS_linear(nx=2)
    fit_sys_IO = deepSI.fit_systems.System_IO_fit_linear(na=4,nb=4)

    fit_sys_SS.fit(sys_data_train)
    fit_sys_IO.fit(sys_data_train)

    #use the fitted system
    sys_data_test_predict_SS = fit_sys_SS.apply_experiment(sys_data_test)
    sys_data_test_res_SS = sys_data_test - sys_data_test_predict_SS #this will subtract the y between the two data sets
    sys_data_test_predict_IO = fit_sys_IO.apply_experiment(sys_data_test)
    sys_data_test_res_IO = sys_data_test - sys_data_test_predict_IO

    #plot results
    plt.subplot(2,1,1)
    sys_data_test.plot()
    sys_data_test_res_SS.plot()
    sys_data_test_res_IO.plot()
    plt.legend(['real','SS','IO'])
    plt.grid()
    plt.subplot(2,1,2)
    sys_data_test_res_SS.plot()
    sys_data_test_res_IO.plot()
    plt.ylim(-0.02,0.02)
    plt.legend(['SS','IO'])
    plt.grid()
    plt.show()




# testing here is a normal line


if __name__ == '__main__':
    linear_SI_example()

    
