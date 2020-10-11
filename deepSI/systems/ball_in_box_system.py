



import deepSI
from deepSI.systems.System import System, System_Deriv, System_data
import numpy as np
from gym.spaces import Box


class ball_in_box_system(System_Deriv): #discrate system single system
    """docstring for double_well_system

    V(x) = 1/2*min((x-a)**2,(x+a)**2)
    v' = -(x-a) if x>0 else (x+a) + u #+ resistance 
    x' = v
    Fmax < a
    Fmin > -a
    """
    def __init__(self, Fmax=0.25, Nresist=0.7):
        '''Noise, system setting and x0 settings'''
        self.Fmax = Fmax
        dt = 2*np.pi/20 #20 points in the sin
        self.gamma = Fmax*dt/0.1 # ux*dt/gamma = X=0.1
        super(ball_in_box_system, self).__init__(dt=dt,nx=2)
        self.action_space = Box(float(-1),float(1),shape=(2,))

    def reset(self):
        self.x = [0.5,0.5,0,0] #[x,y,vx,vy]
        return self.h(self.x) #return position

    def deriv(self,x,u): #will be converted by 
        ux,uy = np.clip(u,-1,1)*self.Fmax
        # print(u)
        x,y,vx,vy = x
        bar = 500
        dvxdt = (1/x**2-1/(1-x)**2)/200+ux-self.gamma*vx
        dvydt = (1/y**2-1/(1-y)**2)/200+uy-self.gamma*vy
        return [vx,vy,dvxdt,dvydt]

    def h(self,x):
        return x[0],x[1] #return position

class ball_in_box_video_system(ball_in_box_system): #discrate system single system
    """docstring for double_well_system

    V(x) = 1/2*min((x-a)**2,(x+a)**2)
    v' = -(x-a) if x>0 else (x+a) + u #+ resistance 
    x' = v
    Fmax < a
    Fmin > -a
    """
    def __init__(self, Fmax=0.25, Nresist=0.7):
        self.ny_vid, self.nx_vid = 25, 25
        super(ball_in_box_video_system, self).__init__(Fmax=Fmax,Nresist=Nresist)
        self.observation_space = Box(0.,1.,shape=(self.nx_vid,self.ny_vid))
        

    def h(self,x):
        # A = np.zeros((self.nx_vid,self.ny_vid))
        Y = np.linspace(0,1,num=self.ny_vid)
        X = np.linspace(0,1,num=self.nx_vid)
        Y,X = np.meshgrid(Y,X)
        r = 0.22
        A = np.clip((r**2-(X-x[1])**2-(Y-x[0])**2)/r**2,0,1)
        return A #return position

if __name__ == '__main__':
    sys = ball_in_box_video_system() 
    exp = System_data(u=[sys.action_space.sample() for i in range(1000)])
    print(sys.action_space.low)
    sys_data = sys.apply_experiment(exp)

    sys_data.to_video(file_name='test')
