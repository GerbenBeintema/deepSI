
#! /usr/bin/env python3
#
# This is an example shows application of linear system identification



import deepSI
import numpy as np
from matplotlib import pyplot as plt

def linear_SI_benchmark():
    sys_data_train, sys_data_test = deepSI.datasets.sista_database.evaporator() #downloads and splits data into 75% train and 25% test
    #data sets consist of 3 inputs and 3 outputs and has 6k samples

    #create linear fit systems
    # best_score, best_sys, best_sys_dict, best_fit_dict = \
    #         deepSI.fit_systems.grid_search(deepSI.fit_systems.System_IO_fit_linear, \
    #                     sys_data_train, \
    #                     sys_dict_choices=dict(na=[1,2,3],nb=[9,10,11,12,13,15,20,25,30]), \
    #                     fit_dict_choices=dict())
    # print(best_score,best_sys_dict)

    # best_score, best_sys, best_sys_dict, best_fit_dict = \
    #         deepSI.fit_systems.grid_search(deepSI.fit_systems.statespace_linear_system, \
    #                     sys_data_train, \
    #                     sys_dict_choices=dict(nx=[1,2,4,5,10,15,20,30,50]), \
    #                     fit_dict_choices=dict(SS_A_stability=[True],SS_f=[10,20,40,60,80]))
    # print(best_score,best_sys_dict)

    fit_sys_SS = deepSI.fit_systems.statespace_linear_system(nx=8)
    fit_sys_IO = deepSI.fit_systems.System_IO_fit_linear(na=1,nb=10)

    fit_sys_SS.fit(sys_data_train,SS_f=40)
    fit_sys_IO.fit(sys_data_train)

    #use the fitted system
    sys_data_test_simulation_SS = fit_sys_SS.apply_experiment(sys_data_test)
    sys_data_test_res_SS = sys_data_test - sys_data_test_simulation_SS #this will subtract the y between the two data sets
    sys_data_test_simulation_IO = fit_sys_IO.apply_experiment(sys_data_test)
    sys_data_test_res_IO = sys_data_test - sys_data_test_simulation_IO

    #plot results
    plt.subplot(2,1,1)
    plt.title('IO NRMS')
    plt.plot(sys_data_test.y[:,0])
    plt.plot(sys_data_test_res_SS.y[:,0])
    plt.plot(sys_data_test_res_IO.y[:,0])
    plt.legend(['real',f'SS NRMS={sys_data_test_simulation_IO.NRMS(sys_data_test):.2%}',f'IO NRMS={sys_data_test_simulation_SS.NRMS(sys_data_test):.2%}'])
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(sys_data_test_res_SS.y[:,0])
    plt.plot(sys_data_test_res_IO.y[:,0])
    # plt.ylim(-0.02,0.02)
    plt.legend(['SS','IO'])
    plt.grid()
    plt.show()

if __name__ == '__main__':
    linear_SI_benchmark()