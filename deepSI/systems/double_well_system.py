


import deepSI
from deepSI.systems.System import System, System_deriv, System_data
import numpy as np
from gym.spaces import Box


class double_well_system(System_deriv): #discrate system single system
    """docstring for double_well_system

    V(x) = 1/2*min((x-a)**2,(x+a)**2)
    v' = -(x-a) if x>0 else (x+a) + u #+ resistance 
    x' = v
    Fmax < a
    Fmin > -a
    """
    def __init__(self, a=1, Fmax=0.25,Nresist=0.7):
        '''Noise, system setting and x0 settings'''
        self.a = a
        self.Fmax = Fmax
        self.gamma = 1/(Nresist*2*np.pi)
        dt = 2*np.pi/20 #20 points in the sin
        super(double_well_system, self).__init__(dt=dt,nx=2)
        self.action_space = Box(-float(-1),float(1),shape=tuple())

    def reset(self):
        self.x = [-self.a,0]
        return self.h(self.x) #return position

    def deriv(self,x,u): #will be converted by 
        u = np.clip(u,-1,1)*self.Fmax
        x,v = x
        dxdt = v
        dvdt = -(x-self.a) if x>0 else -(x+self.a)
        dvdt += u
        dvdt -= self.gamma*v
        return [dxdt,dvdt]

    def h(self,x):
        return x[0] #return position

class double_well_video_system(double_well_system): #discrate system single system
    """docstring for double_well_system

    V(x) = 1/2*min((x-a)**2,(x+a)**2)
    v' = -(x-a) if x>0 else (x+a) + u #+ resistance 
    x' = v
    Fmax < a
    Fmin > -a
    """
    def __init__(self, a=1, Fmax=0.25,Nresist=0.7):
        self.ny_vid, self.nx_vid = 30, 10
        super(double_well_video_system, self).__init__(a=a,Fmax=Fmax,Nresist=Nresist)
        self.observation_space = Box(0.,1.,shape=(20,100))
        

    def h(self,x):
        A = np.zeros((self.nx_vid,self.ny_vid))
        Y = np.linspace(-2.5,2.5,num=self.ny_vid)
        dy = (Y[1]-Y[0])
        X = np.linspace(-1,1,num=self.nx_vid)
        Y,X = np.meshgrid(Y,X)
        r = 0.75
        A = np.clip((r**2-X**2-(Y-x[0])**2)/r**2,0,1)
        return A #return position

if __name__=='__main__':
    import cv2

    import os
    from PIL import Image
    from matplotlib import pyplot as plt
    sys = double_well_video_system()
    train = sys.get_train_data()
    test = sys.get_test_data()
    exp = deepSI.System_data(u=2*np.sin(np.arange(0,100,sys.dt)))




    # exp.plot(show=True)

    sin_out = sys.apply_experiment(exp)


    def to_vid(sys_data,video_name = 'droplets-video.mp4'):
        
        nx,ny = sys_data.y.shape[1],sys_data.y.shape[2] #resolution of simulation
        nx_out,ny_out = nx*10,ny*10 #resolution of video produced
        height, width = nx_out, ny_out

        # height, width = 20, 100
        video = cv2.VideoWriter(video_name, 0, 60, (width,height))
        # video.write(to_img(D))
        # to_img = lambda x: (x.copy()[:,:,None]*255*np.ones((1,1,3))).astype(np.uint8)
        resize = lambda x: np.array(Image.fromarray(x).resize((ny_out,nx_out)))
        to_img = lambda x: resize((x.copy()[:,:,None]*255*np.ones((1,1,3))).astype(np.uint8))
        try: 
            for yi in sys_data.y:
                video.write(to_img(yi))
        finally:
            cv2.destroyAllWindows()
            video.release()


    #SS_encoder
    # fit_sys = deepSI.fit_systems.Torch_io(na=4,nb=2)
    to_vid(train,'train_real.mp4')
    



    if True:
        fit_sys = deepSI.fit_systems.SS_encoder(nx=5, na=12, nb=12)
        fit_sys.fit(train[1000:].flatten(),verbose=2,epochs=15*60, Loss_kwargs=dict(nf=50),sim_val=train[:1000].flatten(),sim_val_fun='RMS')
        fit_sys.save_system('encoder_visual_sys')
    else:
        fit_sys = deepSI.load_system('encoder_visual_sys')

    
    plt.plot(fit_sys.n_step_error(train.flatten()))
    plt.show()

    train_predict = fit_sys.apply_experiment(train.flatten())

    # to_vid(train,'train_real.mp4')
    to_vid(train_predict.reshape_as(train),'train_predict.mp4')

    


    # print(sin_out.y)
    # sin_out.plot(show=True)
    # fit_sys = deepSI.fit_systems.NN_ARX_multi(nf=60)
    # fit_sys.fit(train,verbose=2,epochs=200)
    # test_predict = fit_sys.simulation(test)
    # test.plot(show=False)
    # train.plot(show=False)
    # test_predict.plot(show=False)
    # plt.legend(['real',f'lin {test_predict.BFR(test)/100:.2%}'])
    # plt.show()

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
