
#https://homes.esat.kuleuven.be/~tokka/daisydata.html

from deepSI import System_data, System_data_list
from deepSI.system_data.datasets.dataset_utils import *

import os
import numpy as np

import tempfile
import os.path



def DaISy_download(url,dir_placement=None,download_size=None,force_download=False):    
    dir_name = 'DaISy_data'
    save_dir = cashed_download(url,dir_name,dir_placement=dir_placement,download_size=download_size,force_download=force_download)
    file = os.path.join(save_dir,url.split('/')[-1][:-3])
    data = np.loadtxt(file)
    return data

def destill(dir_placement=None,force_download=False,split_data=True,noise=10):
    '''This file describes the data in the destill.dat file.
    1. Contributed by:
        Peter Van Overschee
        K.U.Leuven - ESAT - SISTA
        K. Mercierlaan 94
        3001 Heverlee
        Peter.Vanoverschee@esat.kuleuven.ac.be
    2. Process/Description:
        Data of a simulation (not real !) related to the identification
        of an ethane-ethylene destillationcolumn. The series consists of 4
        series: 
            U_dest, Y_dest:     without noise (original series)
            U_dest_n10, Y_dest_n10: 10 percent additive white noise
            U_dest_n20, Y_dest_n20: 20 percent additive white noise
            U_dest_n30, Y_dest_n30: 30 percent additive white noise
    3. Sampling time 
        15 min.
    4. Number of samples: 
        90 samples
    5. Inputs:
        a. ratio between the reboiler duty and the feed flow
        b. ratio between the reflux rate and the feed flow
        c. ratio between the distillate and the feed flow
        d. input ethane composition
        e. top pressure
    6. Outputs:
        a. top ethane composition
        b. bottom ethylene composition
        c. top-bottom differential pressure.
    7. References:
        R.P. Guidorzi, M.P. Losito, T. Muratori, The range error test in the
        structural identification of linear multivariable systems,
        IEEE transactions on automatic control, Vol AC-27, pp 1044-1054, oct.
        1982.
    8. Known properties/peculiarities
        
    9. Some MATLAB-code to retrieve the data
        !gunzip destill.dat.Z
        load destill.dat
        U=destill(:,1:20);
        Y=destill(:,21:32);
        U_dest=U(:,1:5);
        U_dest_n10=U(:,6:10);
        U_dest_n20=U(:,11:15);  
        U_dest_n30=U(:,16:20);
        Y_dest=Y(:,1:3);
        Y_dest_n10=Y(:,4:6);
        Y_dest_n20=Y(:,7:9);
        Y_dest_n30=Y(:,10:12);
    '''
    url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/process_industry/destill.dat.gz'
    destill = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)
    U=destill[:,:20]
    Y=destill[:,20:]
    U_dest=U[:,:5]
    U_dest_n10=U[:,5:10]
    U_dest_n20=U[:,10:15]  
    U_dest_n30=U[:,15:20]
    Y_dest=Y[:,0:3]
    Y_dest_n10=Y[:,3:6]
    Y_dest_n20=Y[:,6:9]
    Y_dest_n30=Y[:,9:12]
    if noise==0:
        data = System_data(u=U_dest,y=Y_dest)
    elif noise==10:
        data = System_data(u=U_dest_n10,y=Y_dest_n10)
    elif noise==20:
        data = System_data(u=U_dest_n20,y=Y_dest_n20)
    elif noise==30:
        data = System_data(u=U_dest_n30,y=Y_dest_n30)
    return data.train_test_split() if split_data else data

def destill_all(dir_placement=None,force_download=False,split_data=True):
    out = System_data_list([destill(dir_placement=dir_placement,force_download=force_download,split_data=False,noise=n) for n in [0,10,20,30]])
    return out.train_test_split() if split_data else out


def glassfurnace(dir_placement=None,force_download=False,split_data=True):
    '''This file describes the data in the glassfurnace.dat file.
    1. Contributed by:
        Peter Van Overschee
        K.U.Leuven - ESAT - SISTA
        K. Mercierlaan 94
        3001 Heverlee
        Peter.Vanoverschee@esat.kuleuven.ac.be
    2. Process/Description:
        Data of a glassfurnace (Philips)
    3. Sampling time 
        
    4. Number of samples: 
        1247 samples
    5. Inputs:
        a. heating input
        b. cooling input
        c. heating input
    6. Outputs:
        a. 6 outputs from temperature sensors in a cross section of the 
        furnace
    7. References:
        a. Van Overschee P., De Moor B., N4SID : Subspace Algorithms for 
        the Identification of Combined Deterministic-Stochastic Systems, 
        Automatica, Special Issue on Statistical Signal Processing and Control, 
        Vol. 30, No. 1, 1994, pp. 75-93
        b.  Van Overschee P., "Subspace identification : Theory, 
        Implementation, Application" , Ph.D. Thesis, K.U.Leuven, February 1995. 
    8. Known properties/peculiarities
        
    9. Some MATLAB-code to retrieve the data
        !gunzip glassfurnace.dat.Z
        load glassfurnace.dat
          T=glassfurnace(:,1);
        U=glassfurnace(:,2:4);
        Y=glassfurnace(:,5:10);
    '''
    url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/process_industry/glassfurnace.dat.gz'
    glassfurnace = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)
    U=glassfurnace[:,1:4]
    Y=glassfurnace[:,4:10]
    data = System_data(u=U,y=Y)
    return data.train_test_split() if split_data else data


def powerplant(dir_placement=None,force_download=False,split_data=True):
    '''This file describes the data in the powerplant.dat file.
    1. Contributed by:
        Peter Van Overschee
        K.U.Leuven - ESAT - SISTA
        K. Mercierlaan 94
        3001 Heverlee
        Peter.Vanoverschee@esat.kuleuven.ac.be
    2. Process/Description:
        data of a power plant (Pont-sur-Sambre (France)) of 120 MW
    3. Sampling time 
        1228.8 sec
    4. Number of samples: 
        200 samples
    5. Inputs:
        1. gas flow
        2. turbine valves opening
        3. super heater spray flow
        4. gas dampers
        5. air flow
    6. Outputs:
        1. steam pressure
        2. main stem temperature
        3. reheat steam temperature
    7. References:
        a. R.P. Guidorzi, P. Rossi, Identification of a power plant from normal
        operating records. Automatic control theory and applications (Canada,
        Vol 2, pp 63-67, sept 1974.
        b. Moonen M., De Moor B., Vandenberghe L., Vandewalle J., On- and
        off-line identification of linear state-space models, International
        Journal of Control, Vol. 49, Jan. 1989, pp.219-232
    8. Known properties/peculiarities
        
    9. Some MATLAB-code to retrieve the data
        !gunzip powerplant.dat.Z
        load powerplant.dat
        U=powerplant(:,1:5);
        Y=powerplant(:,6:8);
        Yr=powerplant(:,9:11);
    '''
    url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/process_industry/powerplant.dat.gz'
    powerplant = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)
    U=powerplant[:,0:5]
    Y=powerplant[:,5:8]
    Yr=powerplant[:,8:11]
    data = System_data(u=U,y=Y)
    return data.train_test_split() if split_data else data



def evaporator(dir_placement=None,force_download=False,split_data=True):
    '''Contributed by:
        Favoreel
        KULeuven
        Departement Electrotechniek ESAT/SISTA
        Kardinaal Mercierlaan 94
        B-3001 Leuven
        Belgium
        wouter.favoreel@esat.kuleuven.ac.be
    Description:
        A four-stage evaporator to reduce the water content of a product, 
        for example milk. The 3 inputs are feed flow, vapor flow to the 
        first evaporator stage and cooling water flow. The three outputs 
        are the dry matter content, the flow and the temperature of the 
        outcoming product.
    Sampling:
    Number:
        6305
    Inputs:
        u1: feed flow to the first evaporator stage
        u2: vapor flow to the first evaporator stage
        u3: cooling water flow
    Outputs:
        y1: dry matter content
        y2: flow of the outcoming product
        y3: temperature of the outcoming product
    References:
        - Zhu Y., Van Overschee P., De Moor B., Ljung L., Comparison of 
          three classes of identification methods. Proc. of SYSID '94, 
          Vol. 1, 4-6 July, Copenhagen, Denmark, pp.~175-180, 1994.
    Properties:
    Columns:
        Column 1: input u1
        Column 2: input u2
        Column 3: input u3
        Column 4: output y1
        Column 5: output y2
        Column 6: output y3
    Category:
        Thermic systems
    Where:
    '''
    url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/process_industry/evaporator.dat.gz'
    data = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)
    data = System_data(u=data[:,1:4],y=data[:,4:7])
    return data.train_test_split() if split_data else data

def pHdata(dir_placement=None,force_download=False,split_data=True):
    '''Contributed by:
        Jairo Espinosa
        K.U.Leuven ESAT-SISTA
        K.Mercierlaan 94
        B3001 Heverlee
        Jairo.Espinosa@esat.kuleuven.ac.be

    Description:
        Simulation data of a pH neutralization process in a constant volume
        stirring tank. 
        Volume of the tank 1100 liters 
        Concentration of the acid solution (HAC) 0.0032 Mol/l
        Concentration of the base solution (NaOH) 0,05 Mol/l
    Sampling:
        10 sec
    Number:
        2001
    Inputs:
        u1: Acid solution flow in liters
        u2: Base solution flow in liters

    Outputs:
        y: pH of the solution in the tank

    References:
        T.J. Mc Avoy, E.Hsu and S.Lowenthal, Dynamics of pH in controlled 
        stirred tank reactor, Ind.Eng.Chem.Process Des.Develop.11(1972)
        71-78

    Properties:
        Highly non-linear system.

    Columns:
        Column 1: time-steps
        Column 2: input u1
        Column 3: input u2
        Column 4: output y

    Category:
        Process industry systems
    '''
    url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/process_industry/pHdata.dat.gz'
    data = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)
    data = System_data(u=data[:,1:3],y=data[:,3])
    return data.train_test_split() if split_data else data

# def distill2(dir_placement=None,force_download=False,split_data=True): #is a step responce
#     ''' This file describes the data in the distill_col.dat file.
#     1. Contributed by:
#         Jan Maciejowski
#         Cambridge University, Engineering Department
#         Trumpington Street, Cambridge
#         CB2 1PZ, England.
#         jmm@eng.cam.ac.uk
#     2. Process/Description:
#         step response of a fractional distillation column 
#         (Simulation data supplied by SAST Ltd)
#             Steps are applied to each input, one at a time.
#         Input (a) is a step of amplitude 1.0
#         Input (b) is a step of amplitude 1.0 
#         Input (c) is a step of amplitude 1.0 
#     3. Sampling time:
#         10 sec
#     4. Number of samples:
#         250 samples
#     5. Inputs:
#         a. input cooling temperature
#         b. reboiling temperature
#         c. pressure
#     6. Outputs:
#         a. top product flow rate
#         b. C4 concentration
#     7. References:
#         a. Maciejowski J.M., Parameter estimation of multivariable
#             systems using balanced realizations, in:
#             Bittanti,S. (ed), Identification,
#             Adaptation, and Learning, Springer (NATO ASI Series), 1996.
#         b. Chou C.T., Maciejowski J.M., System Identification Using
#         Balanced Parametrizations, IEEE Transactions on Automatic Control,
#         vol. 42, no. 7, July 1997, pp. 956-974.
#     8. Known properties/peculiarities:

#     9. Some MATLAB-code to retrieve the data
#         !gunzip distill_col.dat.Z
#         load distill_col.dat
#         u1y1=distill_col(:,2);  % step on input 1, response on output 1
#         u1y2=distill_col(:,3);  % step on input 1, response on output 2
#         u2y1=distill_col(:,4);  % etc
#         u2y2=distill_col(:,5);
#         u3y1=distill_col(:,6);
#         u3y2=distill_col(:,7);
#     '''
#     url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/process_industry/distill2.dat.gz'
#     data = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)



def dryer2(dir_placement=None,force_download=False,split_data=True):
    '''
    This file describes the data in the dryer.dat file.
    1. Contributed by:
            Jan Maciejowski
            Cambridge University, Engineering Department
            Trumpington Street, Cambridge
            CB2 1PZ, England.
            jmm@eng.cam.ac.uk
    2. Process/Description:
            Data from an industrial dryer (by Cambridge Control Ltd)
    3. Sampling time:
            10 sec
    4. Number of samples:
            867 samples
    5. Inputs:
            a. fuel flow rate
            b. hot gas exhaust fan speed
            c. rate of flow of raw material
    6. Outputs:
            a. dry bulb temperature
            b. wet bulb temperature
            c. moisture content of raw material
    7. References:
            a. Maciejowski J.M., Parameter estimation of multivariable
            systems using balanced realizations, in:
            Bittanti,S. (ed), Identification,
            Adaptation, and Learning, Springer (NATO ASI Series), 1996.
            b. Chou C.T., Maciejowski J.M., System Identification Using
            Balanced Parametrizations, IEEE Transactions on Automatic Control,
            vol. 42, no. 7, July 1997, pp. 956-974.
    8. Known properties/peculiarities:

    9. Some MATLAB-code to retrieve the data
            !gunzip dryer.dat.Z
            load dryer.dat
            U=dryer(:,2:4);
            Y=dryer(:,5:7);
    '''
    url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/process_industry/dryer2.dat.gz'
    data = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)
    data = System_data(u=data[:,1:4],y=data[:,4:7])
    return data.train_test_split() if split_data else data



def exchanger(dir_placement=None,force_download=False,split_data=True):
    '''This file describes the data in exchanger.dat

    1. Contributed by:

       Sergio Bittanti
       Politecnico di Milano
       Dipartimento di Elettronica e Informazione,
       Politecnico di Milano, 
       Piazza Leonardo da Vinci 32, 20133 MILANO (Italy)
       bittanti@elet.polimi.it
     

    2. Process/Description:

    The process is a liquid-satured steam heat exchanger, where water is
    heated by pressurized saturated steam through a copper tube.  The
    output variable is the outlet liquid temperature. The input variables
    are the liquid flow rate, the steam temperature, and the inlet liquid
    temperature.  In this experiment the steam temperature and the inlet
    liquid temperature are kept constant to their nominal values.

    3. Sampling time:

      1 s 

    4. Number of samples:

      4000 

    5. Inputs:

      q: liquid flow rate 

    6. Outputs:

      th: outlet liquid temperature 

    7. References:

    S. Bittanti and L. Piroddi, "Nonlinear identification and control of a
    heat exchanger: a neural network approach", Journal of the Franklin
    Institute, 1996.  L. Piroddi, Neural Networks for Nonlinear Predictive
    Control. Ph.D. Thesis, Politecnico di Milano (in Italian), 1995.

    8. Known properties/peculiarities:

    The heat exchanger process is a significant benchmark for nonlinear
    control design purposes, since it is characterized by a non minimum
    phase behaviour.  In the references cited above the control problem of
    regulating the output temperature of the liquid-satured steam heat
    exchanger by acting on the liquid flow rate is addressed, and both
    direct and inverse identifications of the data are performed.

    Columns:
    Column 1: time-steps 
    Column 2: input q 
    Column 3: output th 

    '''
    url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/process_industry/exchanger.dat.gz'
    data = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)
    data = System_data(u=data[:,1],y=data[:,2])
    return data.train_test_split() if split_data else data


def winding(dir_placement=None,force_download=False,split_data=True):
    '''Contributed by:
        Favoreel
        KULeuven
        Departement Electrotechniek ESAT/SISTA
        Kardinaal Mercierlaan 94
        B-3001 Leuven
        Belgium
        wouter.favoreel@esat.kuleuven.ac.be

    Description:

    The process is a test setup of an industrial winding process.
    The main part of the plant is composed of a plastic web that 
    is unwinded from first reel (unwinding reel), goes over the 
    traction reel and is finally rewinded on the the rewinding reel.
    Reel 1 and 3 are coupled with a DC-motor that is controlled with 
    input setpoint currents I1* and I3*. The angular speed of 
    each reel (S1, S2 and S3) and the tensions in the web between
    reel 1 and 2 (T1) and between reel 2 and 3 (T3) are measured
    by dynamo tachometers and tension meters. 
    We thank Th. Bastogne from the University of Nancy for 
    providing us with these data.

    We are grateful to Thierry Bastogne of the Universite Henri Point Care, who
    provided us with these data.
       

    Sampling: 0.1 Sec

    Number: 2500

    Inputs:  u1: The angular speed of reel 1 (S1)
         u2: The angular speed of reel 2 (S2)
         u3: The angular speed of reel 3 (S3)
         u4: The setpoint current at motor 1 (I1*)
         u5: The setpoint current at motor 2 (I3*)

    Outputs: y1: Tension in the web between reel 1 and 2 (T1)
         y2: Tension in the web between reel 2 and 3 (T3)

    References:

        - Bastogne T., Identification des systemes multivariables par 
        les methodes des sous-espaces. Application a un systeme 
        d'entrainement de bande. PhD thesis. These de doctorat
        de l'Universite Henri Poincare, Nancy 1.

        - Bastogne T., Noura H., Richard A., Hittinger J.M., 
        Application of subspace methods to the identification of a 
        winding process. In: Proc. of the 4th European Control 
        Conference, Vol. 5, Brussels.

    Properties:

    Columns:
        Column 1: input u1
        Column 2: input u2
        Column 3: input u3
        Column 4: input u4
        Column 5: input u5
        Column 6: output y1
        Column 7: output y2

    Category:

        Industrial test setup
    '''
    url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/process_industry/winding.dat.gz'
    data = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)
    data = System_data(u=data[:,0:5],y=data[:,5:7])
    return data.train_test_split() if split_data else data




def cstr(dir_placement=None,force_download=False,split_data=True):
    '''Contributed by:
        Jairo ESPINOSA
        ESAT-SISTA KULEUVEN
        Kardinaal Mercierlaan 94
    B-3001 Heverlee Belgium
        espinosa@esat.kuleuven.ac.be
    Description:
        The Process is a model of a Continuous 
        Stirring Tank Reactor, where the reaction
        is exothermic and the concentration is 
        controlled by regulating the coolant 
        flow.
    Sampling:
        0.1 min
    Number:
        7500
    Inputs:
        q: Coolant Flow l/min
    Outputs:
        Ca: Concentration mol/l
        T: Temperature Kelvin degrees
    References:
        J.D. Morningred, B.E.Paden, D.E. Seborg and D.A. Mellichamp "An adaptive nonlinear predictive controller" in. Proc. of the A.C.C. vol.2 1990 pp.1614-1619
        G.Lightbody and G.W.Irwin. Nonlinear Control Structures Based on Embedded Neural System Models, IEEE Tran. on Neural Networks Vol.8 No.3 pp.553-567
        J.Espinosa and J. Vandewalle, Predictive Control Using Fuzzy Models, Submitted to the 3rd. On-Line World Conference on Soft Computing in Engineering Design and Manufacturing.
    Properties:
    Columns:
        Column 1: time-steps
        Column 2: input q
        Column 3: output Ca
        Column 4: output T
    Category:
        Process Industry Systems
    Where:
      ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/espinosa/datasets/cstr.dat

    '''
    url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/process_industry/cstr.dat.gz'
    data = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)
    data = System_data(u=data[:,1],y=data[:,2:4])
    return data.train_test_split() if split_data else data




def steamgen(dir_placement=None,force_download=False,split_data=True):
    '''Contributed by:
        Jairo Espinosa
        ESAT-SISTA KULEUVEN
        Kardinaal Mercierlaan 94
    B-3001 Heverlee Belgium
        jairo.espinosa@esat.kuleuven.ac.be
    Description:
        The data comes from a model of a Steam Generator at Abbott Power Plant in Champaign IL.
        The model is described in the paper of Pellegrineti [1].
    Sampling:
        3 sec
    Number:
        9600
    Inputs:
        u1: Fuel scaled 0-1
        u2: Air scaled 0-1
        u3: Reference level inches
        u4: Disturbance defined by the load level
    Outputs:
        y1: Drum pressure PSI
        y2: Excess Oxygen in exhaust gases %
        y3: Level of water in the drum
        y4: Steam Flow Kg./s
    References:
        [1] G. Pellegrinetti and J. Benstman, Nonlinear Control Oriented Boiler Modeling -A Benchamrk Problem for Controller Design, IEEE Tran. Control Systems Tech. Vol.4No.1 Jan.1996
        [2] J. Espinosa and J. Vandewalle Predictive Control Using Fuzzy Models Applied to a Steam Generating Unit, Submitted to FLINS 98 3rd. International Workshop on Fuzzy Logic Systems and Intelligent Technologies for Nuclear Science and Industry
    Properties:
        To make possible the open loop identification the wter level was 
        stabilized by appliying to the water flow input a feedforward action proportional to the steam flow
        with value 0.0403 and a PI action with values Kp=0.258 Ti=1.1026e-4 the reference of this controller 
        is the input u3.
    Columns:
        Column 1: time-steps
        Column 2: input fuel
        Column 3: input air
        Column 4: input level ref.
        Column 5: input disturbance
        Column 6: output drum pressure
        Column 7: output excess oxygen
        Column 8: output water level
        Column 9: output steam flow
    Category:
        Process industry systems
    Where:
        ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/espinosa/datasets/powplant.dat
    '''
    url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/process_industry/steamgen.dat.gz'
    data = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)
    data = System_data(u=data[:,1:5],y=data[:,5:])
    return data.train_test_split() if split_data else data




def ballbeam(dir_placement=None,force_download=False,split_data=True):
    '''This file describes the data in the ballbeam.dat file.
    1. Contributed by:
        Peter Van Overschee
        K.U.Leuven - ESAT - SISTA
        K. Mercierlaan 94
        3001 Heverlee
        Peter.Vanoverschee@esat.kuleuven.ac.be
    2. Process/Description:
        Data of a the ball and beam practicum at ESAT-SISTA. 
    3. Sampling time 
        0.1 sec.
    4. Number of samples: 
        1000 samples
    5. Inputs:
        a. angle of the beam
    6. Outputs:
        a. position of the ball
    7. References:
        a.  Van Overschee P., "Subspace identification : Theory, 
        Implementation, Application" , Ph.D. Thesis, K.U.Leuven, February 
        1995, pp. 200-206 
    8. Known properties/peculiarities
        
    9. Some MATLAB-code to retrieve the data
        !gunzip ballbeam.dat.Z
        load ballbeam.dat
        U=ballbeam(:,1);
        Y=ballbeam(:,2);
    '''
    url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/mechanical/ballbeam.dat.gz'
    data = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)
    data = System_data(u=data[:,0],y=data[:,1])
    return data.train_test_split() if split_data else data




def dryer(dir_placement=None,force_download=False,split_data=True):
    '''Contributed by:
        Favoreel
        KULeuven
        Departement Electrotechniek ESAT/SISTA
        Kardinaal Mercierlaan 94
        B-3001 Leuven
        Belgium
        wouter.favoreel@esat.kuleuven.ac.be
    Description:
        Laboratory setup acting like a hair dryer. Air is fanned through a tube
        and heated at the inlet. The air temperature is measured by a 
        thermocouple at the output. The input is the voltage over the heating 
        device (a mesh of resistor wires).
    Sampling:
    Number:
        1000
    Inputs:
        u: voltage of the heating device 
    Outputs:
        y: output air temperature 
    References:
        - Ljung L.  System identification - Theory for the 
          User. Prentice Hall, Englewood Cliffs, NJ, 1987.
        
        - Ljung. L. System Identification Toolbox. For Use 
           with Matlab. The Mathworks Inc., Mass., U.S.A., 1991.
    Properties:
    Columns:
        Column 1: input u
        Column 2: output y
    Category:
        mechanical systems
    Where:
    '''
    url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/mechanical/dryer.dat.gz'
    data = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)
    data = System_data(u=data[:,0],y=data[:,1])
    return data.train_test_split() if split_data else data




def CD_player_arm(dir_placement=None,force_download=False,split_data=True,data_set=0): #todo: check this function if column are correct
    '''
    This data set has two data sets which can be selected by setting the data_set argument to either 0 or 1
    Contributed by:
        Favoreel
        KULeuven
        Departement Electrotechniek ESAT/SISTA
    Kardinaal Mercierlaan 94
    B-3001 Leuven
    Belgium
        wouter.favoreel@esat.kuleuven.ac.be
    Description:
        Data from the mechanical construction of a CD player arm.  
        The inputs are the forces of the mechanical actuators
        while the outputs are related to the tracking accuracy of the arm.
        The data was measured in closed loop, and then through a two-step
        procedure converted to open loop equivalent data
            The inputs are highly colored.
    Sampling:
    Number:
        2048
    Inputs:
        u: forces of the mechanical actuators
    Outputs:
        y: tracking accuracy of the arm
    References:
        We are grateful to R. de Callafon of the
            Mechanical Engineering Systems and Control group of Delft, who
            provided us with these data.
        
        - Van Den Hof P., Schrama R.J.P., An Indirect Method for Transfer 
          Function Estimation From Closed Loop Data. Automatica, Vol. 29, 
          no. 6, pp. 1523-1527, 1993.

    Properties: 
    Columns:
        Column 1: input u1
        Column 2: input u2
        Column 1: output y1
        Column 2: output y2
    Category:
        mechanical systems

    '''
    url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/mechanical/CD_player_arm.dat.gz'
    data = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)
    data0 = System_data(u=data[:,0],y=data[:,2])
    data1 = System_data(u=data[:,1],y=data[:,3])
    data_sets = System_data_list([data0,data1])
    return data_sets.train_test_split() if split_data else data_sets
    # # data = System_data(u=data[:,0:2],y=data[:,2:4])
    # if data_set==0:
    #     return data0.train_test_split()#(data1,data2)#data.train_test_split() if split_data else data
    # else:
    #     return data1.train_test_split()

def flutter(dir_placement=None,force_download=False,split_data=True):
    '''Contributed by:
        Favoreel
        KULeuven
        Departement Electrotechniek ESAT/SISTA
        Kardinaal Mercierlaan 94
        B-3001 Leuven
        Belgium
        wouter.favoreel@esat.kuleuven.ac.be
    Description:
        Wing flutter data. Due to industrial secrecy agreements we are
        not allowed to reveal more details. Important to know is that
        the input is highly colored.
    Sampling:
    Number:
        1024
    Inputs:
        u: 
    Outputs:
        y: 
    References:
     
    Feron E., Brenner M., Paduano J. and Turevskiy A.. "Time-frequency
    analysis for transfer function estimation and application to flutter
    clearance", in AIAA J. on Guidance, Control & Dynamics, vol. 21,
    no. 3, pp. 375-382, May-June, 1998.

    Properties:
    Columns:
        Column 1: input u
        Column 2: output y
    Category:
        mechanical systems
    Where:

    '''
    url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/mechanical/flutter.dat.gz'
    data = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)
    data = System_data(u=data[:,0],y=data[:,1])
    return data.train_test_split() if split_data else data




def robot_arm(dir_placement=None,force_download=False,split_data=True):
    '''Contributed by:
        Favoreel
        KULeuven
        Departement Electrotechniek ESAT/SISTA
    Kardinaal Mercierlaan 94
    B-3001 Leuven
    Belgium
        wouter.favoreel@esat.kuleuven.ac.be
    Description:
        Data from a flexible robot arm. The arm is installed on an electrical 
        motor.  We have modeled the transfer function from the measured reaction 
        torque of the structure on the ground to the acceleration of the 
        flexible arm.  The applied input is a periodic sine sweep.
        
    Sampling:
    Number:
        1024
    Inputs:
        u: reaction torque of the structure
    Outputs:
        y: accelaration of the flexible arm
    References:
        We are grateful to Hendrik Van Brussel and Jan Swevers of the laboratory
            of Production Manufacturing and Automation of the Katholieke
            Universiteit Leuven, who provided us with these data, which were
            obtained in the framework of the Belgian Programme on
            Interuniversity Attraction Poles (IUAP-nr.50) initiated by the
            Belgian State - Prime Minister's Office - Science Policy
            Programming.
    Properties:
    Columns:
        Column 1: input u
        Column 2: output y
    Category:
        mechanical systems
    Where:

    '''
    url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/mechanical/robot_arm.dat.gz'
    data = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)
    data = System_data(u=data[:,0],y=data[:,1])
    return data.train_test_split() if split_data else data




def flexible_structure(dir_placement=None,force_download=False,split_data=True):
    '''Contributed by:
        Maher ABDELGHANI
        IRISA-INRIA
        Campus de Beaulieu
        35042 Rennes cedex
        FRANCE
        Maher.Abdelghani@irisa.fr
    Description:
        Experiment on a Steel Subframe Flexible
        structure performed at LMS-International,
        Leuven-Belgium.
        -Structure suspended with flexible rubber bands.
        -2 shakers at 2 locations were used for force input signals.
        - 28 accelerometers around the structure were used for measurements.
        - The 30 channels were simulataneously measured using the LMS-CadaX
          Data Acquisition Module.
    Sampling:
        1/1024 (s)
    Number:
        8523 samples/channel
    Inputs:
        2 inputs:
        u1= White noise Force
        u2=White noise force.
    Outputs:
        28 outputs:
        Accelerations
    References:
        1. M.Abdelghani, M.Basseville, A.Benvensite,"In-Operation Damage 
           Monitoring and Diagnosis of Vibrating Structures, with Application to 
           Offshore Structures and Rotating Machinery", IMAC-XV Feb.3-6 1997, 
           Fl. USA.
        
        2. M.Abdelghani, C.T.Chou, M. Verhaegen, "Using Subspace Methods for the 
           Identification and Modal Analysis of Structures", IMAC-XV, 
           Feb.3-6 1997, Fl.USA.
    Properties:
        Frequency Range: 10-512 Hz.
    Columns:
        colomn1= input1 (u1)
        colomn2=input2 (u2)
        
        colomns3--30: outputs1--28
    Category:
        Mechanical Structure

    '''
    url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/mechanical/flexible_structure.dat.gz'
    data = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)
    data = System_data(u=data[:,0],y=data[:,1])
    return data.train_test_split() if split_data else data




def foetal_ecg(dir_placement=None,force_download=False,split_data=True):
    '''Contributed by:
        Lieven De Lathauwer
        lieven.delathauwer@esat.kuleuven.be
    Description:
        cutaneous potential recordings of a pregnant woman (8 channels)
    Sampling:
        10 sec
    Number:
        2500 x 8
    Inputs:
    Outputs:
        1-5: abdominal
        6,7,8: thoracic
    References:
        
        Dirk Callaerts,
        "Signal Separation Methods based on Singular Value Decomposition 
        and their Application to the Real-Time Extraction of the
        Fetal Electrocardiogram from Cutaneous Recordings",
        Ph.D. Thesis, K.U.Leuven - E.E. Dept., Dec. 1989.
        
        L. De Lathauwer, B. De Moor, J. Vandewalle, ``Fetal
        Electrocardiogram Extraction by Blind Source Subspace Separation'', 
        IEEE Trans. Biomedical Engineering, Vol. 47, No. 5, May 2000, 
        Special Topic Section on Advances in Statistical Signal Processing 
        for Biomedicine, pp. 567-572.   


    Properties:
    Columns:
        Column 1: time-steps
        Column 2-9: observations
    Category:
        4
    '''
    url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/biomedical/foetal_ecg.dat.gz'
    data = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)
    data = System_data(u=None,y=data[:,1:9])
    return data.train_test_split() if split_data else data




# def tongue(dir_placement=None,force_download=False,split_data=True): #tongue dataset is not dynamic related, skipping.a
#     '''Contributed by:
#         De Lathauwer Lieven
#         K.U.Leuven, E.E. Dept.- ESAT
#         K. Mercierlaan 94
#     B-3001 Leuven (Heverlee)
#     Belgium
#         Lieven.DeLathauwer@esat.kuleuven.ac.be
#     Description:
#         The dataset is a real-valued (5x10x13)-array; basically, it was obtained as
#         follows. High-quality audio recordings and cine-fluorograms were made of
#         five English-speaking test persons while saying sentences of the form ``Say
#         h(em vowel)d again'' (substitution: ``heed, hid, hayed, head, had, hod,
#         hawed, hoed, hood, who'd''). For each of these 10 vowels an acoustic
#         reference moment was defined and the corresponding 5 (corresponding to the
#         different speakers) frames in the film located. Next, speaker-dependent
#         reference grids, taking into account the anatomy of each test person, were
#         defined and superimposed on the remaining x-ray images. The grids consisted
#         of 13 equidistant lines, in the region from epiglottis to tongue tip, more
#         or less perpendicular to the midline of the vocal tract. The array entries
#         now consist of the distance along the grid lines between the surface of the
#         tongue and the harder upper surface of the vocal tract. The values are given
#         in centimeters and have been measured to the nearest 0.5 mm. For a more
#         extensive description of the experiment, we refer to [1].
#     Sampling:
#     Number:
#     Inputs:
#     Outputs:
#         x: speakers
#         y: vowels
#         z: positions
#     References:
#         [1] R. Harshman, P. Ladefoged, L. Goldstein,
#         ``Factor Analysis of Tongue Shapes'',
#         J. Acoust. Soc. Am., Vol. 62, No. 3, Sept. 1977, pp. 693-707.
        
#         [2] L. De Lathauwer, Signal Processing based on Multilinear Algebra,
#         Ph.D. Thesis, K.U.Leuven, E.E. Dept., Sept. 1997.
#     Properties:
#     Columns:
#         Column 1: (speaker number - 1) x 10 + vowel number
#         Columns 2-14: displacement values
#     Category:
#         Biomedical systems
#     '''
#     url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/biomedical/tongue.dat.gz'
#     data = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)




def erie(dir_placement=None,force_download=False,split_data=True,noise=10):
    '''This file describes the data in the erie.dat file.
    1. Contributed by:
        Peter Van Overschee
        K.U.Leuven - ESAT - SISTA
        K. Mercierlaan 94
        3001 Heverlee
        Peter.Vanoverschee@esat.kuleuven.ac.be
    2. Process/Description:
        Data of a simulation (not real !) related to the related to the
        identification of the western basin of Lake Erie. The series consists 
        of 4 series: 
            U_erie, Y_erie:     without noise (original series)
            U_erie_n10, Y_erie_n10: 10 percent additive white noise
            U_erie_n20, Y_erie_n20: 20 percent additive white noise
            U_erie_n30, Y_erie_n30: 30 percent additive white noise
    3. Sampling time 
        1 month
    4. Number of samples: 
        57 samples
    5. Inputs:
        a. water temperature
        b. water conductivity
        c. water alkalinity
        d. NO3
        e. total hardness
    6. Outputs:
        a. dissolved oxigen
        b. algae
    7. References:
        R.P. Guidorzi, M.P. Losito, T. Muratori, On the last eigenvalue
        test in the structural identification of linear multivariable
        systems, Proceedings of the V European meeting on cybernetics and
        systems research, Vienna, april 1980.
    8. Known properties/peculiarities
        The considered period runs from march 1968 till november 1972.
    9. Some MATLAB-code to retrieve the data
        !guzip erie.dat.Z
        load erie.dat
        U=erie(:,1:20);
        Y=erie(:,21:28);
        U_erie=U(:,1:5);
        U_erie_n10=U(:,6:10);
        U_erie_n20=U(:,11:15);  
        U_erie_n30=U(:,16:20);
        Y_erie=Y(:,1:2);
        Y_erie_n10=Y(:,3:4);
        Y_erie_n20=Y(:,5:6);
        Y_erie_n30=Y(:,7:8);

    '''
    url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/environmental/erie.dat.gz'
    erie = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)
    U=erie[:,:20]
    Y=erie[:,20:]
    U_erie=U[:,:5]
    U_erie_n10=U[:,5:10]
    U_erie_n20=U[:,10:15]  
    U_erie_n30=U[:,15:20]
    Y_erie=Y[:,0:2] #two outputs
    Y_erie_n10=Y[:,2:4]
    Y_erie_n20=Y[:,4:6]
    Y_erie_n30=Y[:,6:8]
    if noise==0:
        data = System_data(u=U_erie,y=Y_erie)
    elif noise==10:
        data = System_data(u=U_erie_n10,y=Y_erie_n10)
    elif noise==20:
        data = System_data(u=U_erie_n20,y=Y_erie_n20)
    elif noise==30:
        data = System_data(u=U_erie_n30,y=Y_erie_n30)
    return data.train_test_split() if split_data else data

def erie_all(dir_placement=None,force_download=False,split_data=True):
    out = System_data_list([erie(dir_placement=dir_placement,force_download=force_download,split_data=False,noise=n) for n in [0,10,20,30]])
    return out.train_test_split() if split_data else out


def thermic_res_wall(dir_placement=None,force_download=False,split_data=True):
    '''Contributed by:
        Favoreel
        KULeuven
        Departement Electrotechniek ESAT/SISTA
        Kardinaal Mercierlaan 94
        B-3001 Leuven
        Belgium
        wouter.favoreel@esat.kuleuven.ac.be
    Description:
        Heat flow density through a two layer wall (brick and insulation 
        layer). The inputs are the internal and external temperature of 
        the wall.  The output is the heat flow density through the wall. 
    Sampling:
    Number:
        1680
    Inputs:
        u1: internal wall temperature
        u2: external wall temperature
    Outputs:
        y: heat flow density through the wall
    References:
        - System Identification Competition, Benchmark tests for estimation 
          methods of thermal characteristics of buildings and building 
          components. Organization: J. Bloem, Joint Research Centre, 
          Ispra, Italy, 1994.

    Properties:
    Columns:
        Column 1: input u1
        Column 2: input u2
        Column 3: output y
    Category:
        thermic systems
    Where:
    '''
    url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/thermic/thermic_res_wall.dat.gz'
    data = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)
    data = System_data(u=data[:,0:2],y=data[:,2])
    return data.train_test_split() if split_data else data




def heating_system(dir_placement=None,force_download=False,split_data=True):
    '''Contributed by:
        Favoreel
        KULeuven
        Departement Electrotechniek ESAT/SISTA
        Kardinaal Mercierlaan 94
        B-3001 Leuven
        Belgium
        wouter.favoreel@esat.kuleuven.ac.be
    Description:
        Heat flow density through a two layer wall (brick and insulation 
        layer). The inputs are the internal and external temperature of 
        the wall.  The output is the heat flow density through the wall. 
    Sampling:
    Number:
        1680
    Inputs:
        u1: internal wall temperature
        u2: external wall temperature
    Outputs:
        y: heat flow density through the wall
    References:
        - System Identification Competition, Benchmark tests for estimation 
          methods of thermal characteristics of buildings and building 
          components. Organization: J. Bloem, Joint Research Centre, 
          Ispra, Italy, 1994.

    Properties:
    Columns:
        Column 1: input u1
        Column 2: input u2
        Column 3: output y
    Category:
        thermic systems
    Where:
    '''
    url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/thermic/heating_system.dat.gz'
    data = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)
    data = System_data(u=data[:,0:2],y=data[:,2])
    return data.train_test_split() if split_data else data


# def mrs(dir_placement=None,force_download=False,split_data=True): #is a script
#     '''This file describes the data generated by the mrs.m file

#     1. Contributed by:
#             L. Vanhamme 
#             KUL, ESAT/SISTA
#             Kard. Mercierlaan 94, 3001 Heverlee
#             leentje.vanhamme@esat.kuleuven.ac.be
        
#     2. Process/Description:

#     This simulation signal is derived from a typical in-vivo 31P spectrum
#     measured in the human brain and consists of 11 exponentials. The 31P
#     peaks from brain tissue, phosphomonoesters, inorganic phosphate,
#     phosphodiesters, phosphocreatine, Gamma-ATP, Alpha-ATP and Beta-ATP
#     are present in this simulation signal.

#     3. Sampling interval: 

#     The time sampling interval is 0.333 msec.

#     4. Number of samples the user can choose. 

#     Typical values however range from 256 to 1024.  

#     5. Inputs: 

#     6. Outputs:
        
#     7. References: L. Vanhamme, A. van den Boogaart,

#     S. Van Huffel: Fast and accurate parameter estimation of noisy complex
#     exponentials with use of prior knowledge , Proceedings EUSIPCO-96,
#     Sept. 10-13 1996, Trieste, Italy.

#     L. Vanhamme, A. van den Boogaart, S. Van Huffel: Improved Method for
#     Accurate and Efficient Quantification of MRS Data with Use of Prior
#     Knowledge , January 1997, submitted for publication in "Journal of
#     Magnetic Resonance". (ESAT/SISTA report 97-02).

#     8. Known properties/peculiarities
        
#     9. Some MATLAB-code to retrieve the data

#     The user can of course change the number of data point used to
#     construct the data. He can als add noise with any desired standard
#     deviation as described in the .m file.

#     '''
#     #todo
#     url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/simulators/mrs.m'
#     dir_name = 'DaISy_data'
#     save_dir = cashed_download(url,dir_name,dir_placement=dir_placement,download_size=None,force_download=force_download,zipped=False)
#     file = os.path.join(save_dir,url.split('/')[-1][:-3]) 
#     matfile = loadmat(file)
#     print(matfile.keys())

#     # url = 'url'
#     # data = DaISy_download(url,dir_placement=dir_placement,force_download=force_download)


def internet_traffic(dir_placement=None,force_download=False,split_data=True):
    '''Contributed by:
    Katrien De Cock
    K.U.Leuven, ESAT-SISTA
    Kardinaal Mercierlaan 94
    3001 Heverlee
    decock@esat.kuleuven.ac.be

    Description:  one hour of internet traffic between the Lawrence Berkeley Laboratory and the rest of the world

    Sampling:

    Number: 99999

    Inputs:

    Output: number of packets per  time unit

    References:

    Katrien De Cock and Bart De Moor, Identification of the first order parameters of a circulant modulated Poisson process. Accepted for publication in the proceedings of the International Conference on Telecommunication (ICT '98)

    V. Paxson and S. Floyd, Wide-area traffic: The failure of Poisson modeling, IEEE/ACM Transactions on Networking, 1995


    Properties:

    Columns:
    Column 1: time-steps
    Column 2: output y


    Category: Time series
    '''
    url = 'ftp://ftp.esat.kuleuven.ac.be/pub/SISTA/data/timeseries/internet_traffic.dat.gz'
    dir_name = 'DaISy_data'
    save_dir = cashed_download(url,dir_name,dir_placement=dir_placement,download_size=None,force_download=force_download)
    file = os.path.join(save_dir,url.split('/')[-1][:-3]) 
    with open(file) as f:
        splitted = '\n'.join(f.read().split('\n')[:-5]) #weird shit
        with tempfile.TemporaryFile() as fp:
            fp.write(bytes(splitted,'utf-8'))
            fp.seek(0)
            data = np.loadtxt(fp)
    data = System_data(u=None,y=data[:,1])
    return data.train_test_split() if split_data else data
        

if __name__=='__main__':
    # uxyeye.data_sets.clear_cache()
    destill()
    destill_all() #returns list with all the noise levels
    glassfurnace()
    powerplant()
    evaporator()
    pHdata()
    # distill2() #step responce
    dryer2()
    exchanger()
    winding()
    cstr()
    steamgen()
    ballbeam()
    dryer()
    CD_player_arm()
    flutter()
    robot_arm()
    flexible_structure()
    foetal_ecg()
    #tongue() #not time data skipping
    erie()
    erie_all()
    thermic_res_wall()
    heating_system()
    #mrs() #is a script, todo
    internet_traffic()

all_sets = [destill, glassfurnace, powerplant, evaporator, pHdata, dryer2, exchanger, winding, cstr, steamgen, ballbeam, dryer, CD_player_arm, flutter, robot_arm, flexible_structure, foetal_ecg, erie, thermic_res_wall, heating_system, internet_traffic]
joined_sets = [erie_all,destill_all]