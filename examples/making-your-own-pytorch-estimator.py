import deepSI
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
import torch

class NARX_basic(deepSI.fit_systems.System_torch):
    """docstring for NARX"""
    def __init__(self, na=20, nb=20):
        super(NARX_basic, self).__init__()
        self.na, self.nb = na, nb

        from deepSI.utils import simple_res_net, feed_forward_nn
        self.net = simple_res_net
        self.n_hidden_layers = 2
        self.n_nodes_per_layer = 64
        self.activation = nn.Tanh


    def init_nets(self, nu, ny):
        '''Defined in subclass and initializes networks and returns the parameters

        Parameters
        ----------
        nu : None, int or tuple
            The shape of the input u
        ny : None, int or tuple
            The shape of the output y

        Returns
        -------
        parameters : list
            List of the network parameters
        '''
        assert nu is None
        assert ny is None
        self.NARX_net = self.net(n_in=self.na + self.nb, n_out=1, n_nodes_per_layer=self.n_nodes_per_layer,\
                                     n_hidden_layers=self.n_hidden_layers, activation=self.activation)
        return list(self.NARX_net.parameters())

    def make_training_data(self, sys_data, **loss_kwargs):
        '''Defined in subclass which converts the normed sys_data into training data

        Parameters
        ----------
        sys_data : System_data or System_data_list
            Already normalized
        loss_kwargs : dict
            loss function settings passed into .fit
        '''
        assert sys_data.normed == True

        uy,Y = sys_data.to_IO_data(na=self.na, nb=self.nb)
        return uy, Y

    def loss(self, uy, Y, **loss_kwargs):
        '''Defined in subclass which take the batch data and calculates the loss based on loss_kwargs

        Parameters
        ----------
        training_data_batch : list
            batch of the training data returned by make_training_data and converted to torch arrays
        loss_kwargs : dict
            loss function settings passed into .fit
        '''
        Yhat = self.NARX_net(uy)[:,0]
        return torch.mean((Y-Yhat)**2)

    def init_state(self, sys_data):
        '''Initialize the internal state of the model using the start of sys_data

        Returns
        -------
        Output : an observation (e.g. floats)
            The observation/predicted state at time step k0
        k0 : int
            number of steps that should be skipped

        Notes
        -----
        Example: x0 = encoder(u[t-k0:k0],yhist[t-k0:k0]), and return h(x0), k0
        This function is often overwritten in child. As default it will return self.reset(), 0 
        '''
        k0 = max(self.na,self.nb)
        self.yhist = list(sys_data.y[k0-self.na:k0])
        self.uhist = list(sys_data.u[k0-self.nb:k0-1]) #how it is saved, len(yhist) = na, len(uhist) = nb-1
        #when taking an action uhist gets appended to create the current state
        return sys_data.y[k0-1], k0



    def init_state_multi(self, sys_data, nf=None, dilation=1):
        '''Similar to init_state but to initialize multiple states 
           (used in self.n_step_error and self.one_step_ahead)

            Parameters
            ----------
            sys_data : System_data
                Data used to initialize the state
            nf : int
                skip the nf last states
            dilation: int
                number of states between each state
           '''
        k0 = max(self.na,self.nb)
        self.yhist = np.array([sys_data.y[k0-self.na+i:k0+i] for i in range(0,len(sys_data)-k0-nf+1,dilation)]) #+1? #shape = (N,na)
        self.uhist = np.array([sys_data.u[k0-self.nb+i:k0+i-1] for i in range(0,len(sys_data)-k0-nf+1,dilation)]) #+1? #shape = 
        return self.yhist[:,-1], k0

    def step(self, action):
        '''Applies the action to the system and returns the new observation, 
        should always be overwritten in subclass'''
        self.uhist.append(action)
        uy = np.concatenate((np.array(self.uhist).flat,np.array(self.yhist).flat),axis=0) #might not be the quickest way
        yout = self.NARX_net(torch.as_tensor(uy,dtype=torch.float32)[None,:])[0,0].detach().numpy()
        self.yhist.append(yout)
        self.yhist.pop(0)
        self.uhist.pop(0)
        return yout

    def step_multi(self, actions):
        '''Applies the actions to the system and returns the new observations'''
        self.uhist = np.append(self.uhist, actions[:,None], axis=1)
        uy = np.concatenate([self.uhist.reshape(self.uhist.shape[0],-1),self.yhist.reshape(self.uhist.shape[0],-1)],axis=1) ######todo MIMO
        yout = self.NARX_net(torch.as_tensor(uy,dtype=torch.float32))[:,0].detach().numpy()
        self.yhist = np.append(self.yhist[:,1:],yout[:,None],axis=1)
        self.uhist = self.uhist[:,1:]
        return yout

if __name__ == '__main__':
    sys = NARX_basic()
    train, test = deepSI.datasets.Silverbox()
    sys.fit(train, sim_val=test[:1000])
    test_sim = sys.apply_experiment(test)
    print('NRMS:',test_sim.NRMS(test))
    test.plot()
    (test-test_sim).plot(show=True)
    plt.plot(sys.n_step_error(test))
    plt.show()
