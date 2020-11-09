


# http://www.nonlinearbenchmark.org/
# A. Janot, M. Gautier and M. Brunot, Data Set and Reference Models of EMPS, 2019 Workshop on Nonlinear System Identification Benchmarks, Eindhoven, The Netherlands, April 10-12, 2019.

import deepSI
from deepSI.system_data.System_data import System_data, System_data_list
import os
from scipy.io import loadmat

import tempfile
import os.path
from pathlib import Path

from deepSI.system_data.datasets.dataset_utils import *


def EMPS(dir_placement=None,vir_as_u=True,force_download=False,split_data=True):
    '''The Electro-Mechanical Positioning System is a standard configuration of a drive system for prismatic joint of robots or machine tools. The main source of nonlinearity is caused by friction effects that are present in the setup. Due to the presence of a pure integrator in the system, the measurements are obtained in a closed-loop setting.

    The provided data is described in this link. The provided Electro-Mechanical Positioning System datasets are available for download here. This zip-file contains the system description and available data sets .mat file format.

    Please refer to the Electro-Mechanical Positioning System as:

    A. Janot, M. Gautier and M. Brunot, Data Set and Reference Models of EMPS, 2019 Workshop on Nonlinear System Identification Benchmarks, Eindhoven, The Netherlands, April 10-12, 2019.

    Special thanks to Alexandre Janot for making this dataset available.'''
    #q_cur current measured position
    #q_ref target/reference potion
    #non-linear due to singed friction force Fc ~ sing(dq/dt)
    #t time
    #vir applied the vector of motor force expressed in the load side i.e. in N;

    url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/EMPS/EMPS.zip'
    download_size = 1949929
    save_dir = cashed_download(url,'EMPS',dir_placement=dir_placement,download_size=download_size,force_download=force_download)
    matfile = loadmat(os.path.join(save_dir,'DATA_EMPS.mat'))
    q_cur, q_ref, t, vir = [matfile[a][:,0] for a in ['qm','qg','t','vir']] #qg is reference, either, q_ref is input or vir is input
    out_data = System_data(u=vir,y=q_cur) if vir_as_u else System_data(u=q_ref,y=q_cur)
    return out_data.train_test_split() if split_data else out_data


def CED(dir_placement=None,force_download=False,split_data=True):
    '''The coupled electric drives consists of two electric motors that drive a pulley using a flexible belt. The pulley is held by a spring, resulting in a lightly damped dynamic mode. The electric drives can be individually controlled allowing the tension and the speed of the belt to be simultaneously controlled. The drive control is symmetric around zero, hence both clockwise and counter clockwise movement is possible. The focus is only on the speed control system. The angular speed of the pulley is measured as an output with a pulse counter and this sensor is insensitive to the sign of the velocity. The available data sets are short, which constitute a challenge when performing identification.

    The provided data is part of a technical note available online through this link. The provided Coupled Electric Drives datasets are available for download here. This zip-file contains the system description and available data sets, both in the .csv and .mat file format.

    Please refer to the Coupled Electric Drives dataset as:

    T. Wigren and M. Schoukens, Coupled Electric Drives Data Set and Reference Models, Technical Report, Department of Information Technology, Uppsala University, Department of Information Technology, Uppsala University, 2017.

    Previously published results on the Coupled Electric Drives benchmark are listed in the history section of this webpage.

    Special thanks to Torbjön Wigren for making this dataset available.

    NOTE: We are re-evaluating the continuous-time models reported in the technical note. For now, the discrete-time model reported in eq. (9) of the technical note can be used in combination of PRBS dataset with amplitude 1.'''

    #http://www.it.uu.se/research/publications/reports/2017-024/2017-024-nc.pdf
    url = 'http://www.it.uu.se/research/publications/reports/2017-024/CoupledElectricDrivesDataSetAndReferenceModels.zip'
    download_size=278528
    save_dir = cashed_download(url,'CED',dir_placement=dir_placement,download_size=download_size,force_download=force_download)
    
    # there is something not right with this data set. 
    datasets = []
    # d = 'DATAPRBS.MAT'
    # # for d in [,'DATAUNIF.MAT']:
    # matfile = loadmat(os.path.join(save_dir,d))
    # u1,u2,u3,z1,z2,z3 = [matfile[a][:,0] for a in ['u1','u2','u3','z1','z2','z3']]
    # datasets.extend([System_data(u=u,y=y) for (u,y) in [(u1,z1),(u2,z2),(u3,z3)]])


    d = 'DATAUNIF.MAT'
    matfile = loadmat(os.path.join(save_dir,d))
    u11,u12,z11,z12 = [matfile[a][:,0] for a in ['u11','u12','z11','z12']]
    datasets = System_data_list([System_data(u=u11,y=z11), System_data(u=u12,y=z12)])
    return datasets if not split_data else datasets.sdl

def F16(dir_placement=None,yn=0,force_download=False,split_data=True):
    '''The F-16 Ground Vibration Test benchmark features a high order system with clearance and friction nonlinearities at the mounting interface of the payloads.

    The experimental data made available to the Workshop participants were acquired on a full-scale F-16 aircraft on the occasion of the Siemens LMS Ground Vibration Testing Master Class, held in September 2014 at the Saffraanberg military basis, Sint-Truiden, Belgium.

    During the test campaign, two dummy payloads were mounted at the wing tips to simulate the mass and inertia properties of real devices typically equipping an F-16 in ﬂight. The aircraft structure was instrumented with accelerometers. One shaker was attached underneath the right wing to apply input signals. The dominant source of nonlinearity in the structural dynamics was expected to originate from the mounting interfaces of the two payloads. These interfaces consist of T-shaped connecting elements on the payload side, slid through a rail attached to the wing side. A preliminary investigation showed that the back connection of the right-wing-to-payload interface was the predominant source of nonlinear distortions in the aircraft dynamics, and is therefore the focus of this benchmark study.

    A detailed formulation of the identification problem can be found here. All the provided files and information on the F-16 aircraft benchmark system are available for download here. This zip-file contains a detailed system description, the estimation and test data sets, and some pictures of the setup. The data is available in the .csv and .mat file format.

    Please refer to the F16 benchmark as:

    J.P. Noël and M. Schoukens, F-16 aircraft benchmark based on ground vibration test data, 2017 Workshop on Nonlinear System Identification Benchmarks, pp. 19-23, Brussels, Belgium, April 24-26, 2017.

    Previously published results on the F-16 Ground Vibration Test benchmark are listed in the history section of this webpage.

    Special thanks to Bart Peeters (Siemens Industry Software) for his help in creating this benchmark.'''
    #todo this is still broken for some mat files
    # assert False, 'this is still broken for some files where y has many more dimensions than expected'
    url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/F16/F16GVT_Files.zip'
    download_size=148455295
    save_dir = cashed_download(url,'F16',dir_placement=dir_placement,download_size=download_size,force_download=force_download)
    save_dir = os.path.join(save_dir,'F16GVT_Files/BenchmarkData') #matfiles location
    matfiles = [os.path.join(save_dir,a).replace('\\','/') for a in os.listdir(save_dir) if a.split('.')[-1]=='mat']
    datasets = []
    for file in sorted(matfiles):
        out = loadmat(file)
        Force, Voltage, (y1,y2,y3),Fs = out['Force'][0], out['Voltage'][0], out['Acceleration'], out['Fs'][0,0]
        #u = Force
        #y = one of the ys, multi objective regression?
        name = file.split('/')[-1]
        if 'SpecialOddMSine' not in name:
            datasets.append(System_data(u=Force,y=[y1,y2,y3][yn],system_dict={'name':name}))
    datasets = System_data_list(datasets)
    return datasets if not split_data else datasets.train_test_split()


def Cascaded_Tanks(dir_placement=None,force_download=False,split_data=True):
    url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/CASCADEDTANKS/CascadedTanksFiles.zip'
    download_size=7520592
    save_dir = cashed_download(url,'Cascaded_Tanks',dir_placement=dir_placement,download_size=download_size,force_download=force_download)
    save_dir = os.path.join(save_dir,'CascadedTanksFiles')

    out = loadmat(os.path.join(save_dir,'dataBenchmark.mat'))

    uEst, uVal, yEst, yVal, Ts = out['uEst'][:,0],out['uVal'][:,0],out['yEst'][:,0],out['yVal'][:,0],out['Ts'][0,0]
    datasets = [System_data(u=uEst,y=yEst),System_data(u=uVal,y=yVal)]
    datasets = System_data_list(datasets)
    return datasets if not split_data else (datasets.sdl[0], datasets.sdl[1])

def WienerHammerstein_Process_Noise(dir_placement=None, force_download=False, split_data=True):
    '''Warning this is a quite a bit of data'''
    url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/WIENERHAMMERSTEINPROCESS/WienerHammersteinFiles.zip'
    download_size=423134764
    save_dir = cashed_download(url,'WienHammer',dir_placement=dir_placement,download_size=download_size,force_download=force_download)
    save_dir = os.path.join(save_dir,'WienerHammersteinFiles') #matfiles location
    matfiles = [os.path.join(save_dir,a).replace('\\','/') for a in os.listdir(save_dir) if a.split('.')[-1]=='mat']
    datafiles = []
    datafiles_test = []

    #file = 'WH_CombinedZeroMultisineSinesweep.mat' #'WH_Triangle_meas.mat'
    for file in matfiles:
        out = loadmat(os.path.join(save_dir,file))
        r,u,y,fs = out['dataMeas'][0,0]
        fs = fs[0,0]
        data = [System_data(u=ui,y=yi) for ui,yi in zip(u.T,y.T)]
        if split_data and 'Test' in file:
            datafiles_test.extend(data)
        else:
            datafiles.extend(data)

    datasets = System_data_list(datasets)
    datafiles_test = System_data_list(datafiles_test)
    return (datafiles, datafiles_test) if split_data else datafiles  #brackets required if before ,

def BoucWen(dir_placement=None, force_download=False, split_data=True):

    #todo: dot p file integration as system for training data
    #generate more data
    url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/BOUCWEN/BoucWenFiles.zip'
    download_size=5284363
    save_dir = cashed_download(url,'BoucWen',dir_placement=dir_placement,download_size=download_size,force_download=force_download)
    save_dir = os.path.join(save_dir,'BoucWenFiles\\Test signals\\Validation signals') #matfiles location

    datafiles = []

    out = loadmat(os.path.join(save_dir,'uval_multisine.mat'))
    u_multisine = out['uval_multisine'][0]
    out = loadmat(os.path.join(save_dir,'yval_multisine.mat'))
    y_multisine = out['yval_multisine'][0]
    datafiles.append(System_data(u=u_multisine,y=y_multisine))

    out = loadmat(os.path.join(save_dir,'uval_sinesweep.mat'))
    u_sinesweep = out['uval_sinesweep'][0]
    out = loadmat(os.path.join(save_dir,'yval_sinesweep.mat'))
    y_sinesweep = out['yval_sinesweep'][0]
    datafiles.append(System_data(u=u_sinesweep,y=y_sinesweep))
    datafiles = System_data_list(datafiles)
    if not split_data:
        return datafiles
    else:
        return datafiles.train_test_split()


def ParWHF(dir_placement=None,force_download=False, split_data=True):
    '''Parallel Wienner-Hammerstein'''
    url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/PARWH/ParWHFiles.zip'
    download_size=58203304
    save_dir = cashed_download(url,'ParWHF',dir_placement=dir_placement,download_size=download_size,force_download=force_download)
    save_dir = os.path.join(save_dir,'ParWHFiles') #matfiles location

    out = loadmat(os.path.join(save_dir,'ParWHData.mat'))
    # print(out.keys())
    # print(out['amp'][0]) #5 values 
    # print(out['fs'][0,0])
    # print(out['lines'][0]) #range 2:4096
    # print('uEst',out['uEst'].shape) #(16384, 2, 20, 5), (N samplees, P periods, M Phase changes, nAmp changes)
    # print('uVal',out['uVal'].shape) #(16384, 2, 1, 5)
    # print('uValArr',out['uValArr'].shape) #(16384, 2)
    # print('yEst',out['yEst'].shape) #(16384, 2, 20, 5)
    # print('yVal',out['yVal'].shape) #(16384, 2, 1, 5)
    # print('yValArr',out['yValArr'].shape) #(16384, 2)

    datafiles = []
    datafiles_test = []
    #todo split train, validation and test
    uEst = out['uEst'].reshape((16384,-1))
    yEst = out['yEst'].reshape((16384,-1))
    datafiles.extend([System_data(u=ui,y=yi) for ui,yi in zip(uEst.T,yEst.T)])
    uVal = out['uVal'].reshape((16384,-1))
    yVal = out['yVal'].reshape((16384,-1))
    data = [System_data(u=ui,y=yi) for ui,yi in zip(uVal.T,yVal.T)]
    datafiles_test.extend(data) if split_data else datafiles.extend(data)
    uValArr = out['uValArr'].reshape((16384,-1))
    yValArr = out['yValArr'].reshape((16384,-1))
    data = [System_data(u=ui,y=yi) for ui,yi in zip(uValArr.T,yValArr.T)]
    datafiles_test.extend(data) if split_data else datafiles.extend(data)
    datasets = System_data_list(datasets)
    if split_data:
        datafiles_test = System_data_list(datafiles_test)
        return (datafiles, datafiles_test)
    else:
        return datafiles

def WienerHammerBenchMark(dir_placement=None,force_download=False, split_data=True):
    url = 'http://www.ee.kth.se/~hjalmars/ifac_tc11_benchmarks/2009_wienerhammerstein/WienerHammerBenchMark.mat'
    download_size=1707601
    save_dir = cashed_download(url,'WienerHammerBenchMark',dir_placement=dir_placement,download_size=download_size,force_download=force_download,zipped=False)

    out = loadmat(os.path.join(save_dir,'WienerHammerBenchMark.mat'))
    u,y,fs = out['uBenchMark'][:,0],out['yBenchMark'][:,0],out['fs'][0,0]
    out = System_data(u=u[5200:184000],y=y[5200:184000]) #fine only were u is active
    # out = System_data(u=u,y=y)
    return (out[:100000],out[100000:]) if split_data else out

def Silverbox(dir_placement=None,force_download=False, split_data=True):
    '''The Silverbox system can be seen as an electronic implementation of the Duffing oscillator. It is build as a 
    2nd order linear time-invariant system with a 3rd degree polynomial static nonlinearity around it in feedback. 
    This type of dynamics are, for instance, often encountered in mechanical systems.

    The provided data is part of a previously published ECC paper available online. A technical note describing the 
    Silverbox benchmark can be found here. All the provided data (.mat file format) on the Silverbox system is available
    for download here. This .zip file contains the Silverbox dataset as specified in the benchmark document (V1 is the
    input record, while V2 is the measured output), extended with .csv version of the same data and an extra data record 
    containing a Schroeder phase multisine measurement.

    Please refer to the Silverbox benchmark as:

    T. Wigren and J. Schoukens. Three free data sets for development and benchmarking in nonlinear system identification. 
    2013 European Control Conference (ECC), pp.2933-2938 July 17-19, 2013, Zurich, Switzerland.

    Previously published results on the Silverbox benchmark are listed in the history section of this webpage.

    Special thanks to Johan Schoukens for creating this benchmark, and to Torbjörn Wigren for hosting this benchmark.
    '''
    url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/SILVERBOX/SilverboxFiles.zip'
    download_size=5793999
    save_dir = cashed_download(url,'Silverbox',dir_placement=dir_placement,download_size=download_size,force_download=force_download)
    save_dir = os.path.join(save_dir,'SilverboxFiles') #matfiles location


    out = loadmat(os.path.join(save_dir,'Schroeder80mV.mat'))

    u,y = out['V1'][0], out['V2'][0]
    data1 = System_data(u=u,y=y)
    out = loadmat(os.path.join(save_dir,'SNLS80mV.mat')) #train test
    u,y = out['V1'][0], out['V2'][0]
    data2 = System_data(u=u,y=y)


    if split_data:
        data_out = System_data(u=data2.u[40650:127400],y=data2.y[40650:127400])
        return data_out.train_test_split()
    return System_data_list([data1, data2])


if __name__=='__main__':
    # clear_cache()

    clear_cache()
    out = Cascaded_Tanks(split_data=True)
    print(out)
    out = Cascaded_Tanks(split_data=False)
    print(out)
    # out[0].plot()

    '''testing'''
    # data_train,data_test = Silverbox()
    # if isinstance(data_test,list):
    #     data_test[1].plot(show=False,nmax=-1)
    #     data_train[1].plot(nmax=-1)
    # else:
    #     data_test.plot(show=False,nmax=-1)
    #     data_train.plot(nmax=-1)
    # print([d.u.shape for d in datas])
    # print([d.y.shape for d in datas])
    # datas = BoucWen()

    # print([sys_data.y.shape for sys_data in datas])
    # print([sys_data.system_dict for sys_data in datas if len(sys_data.y.shape)==2])
    
    # print([sys_data.u.shape for sys_data in datas])
    # fit_sys = deepSI.fit_systems.Linear_fit_system()
    # print('fitting...',end='')
    # fit_sys.fit(datas)
    # print('done')
    # try:
    #     data = datas[0]
    # except:
    #     data = datas
    # sys_open = fit_sys.simulation(data)
    # sys_one = fit_sys.one_step_ahead(data)
    # print(sys_one.BFR(data),sys_one.RMSE(data))
    # print(sys_open.BFR(data),sys_open.RMSE(data))
    # sys_open.plot(show=False,nmax=-1)
    # data.plot(nmax=-1)