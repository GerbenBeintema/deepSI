
from deepSI.fit_systems.fit_system import System_fittable, System_torch
import torch
from torch import nn

class SS_encoder(System_torch):
    """docstring for SS_encoder"""
    def __init__(self, nx=10, na=20, nb=20):
        super(SS_encoder, self).__init__()
        self.nx, self.na, self.nb = nx, na, nb
        self.k0 = max(self.na,self.nb)
        
        from deepSI.utils import simple_res_net, feed_forward_nn
        self.net = simple_res_net
        self.n_hidden_layers = 2
        self.n_nodes_per_layer = 64
        self.activation = nn.Tanh

    ########## How to fit #############
    def make_training_data(self, sys_data, **Loss_kwargs):
        assert sys_data.normed == True
        nf = Loss_kwargs.get('nf',25)
        dilation = Loss_kwargs.get('dilation',1)
        return sys_data.to_encoder_data(na=self.na,nb=self.nb,nf=nf,dilation=dilation,force_multi_u=True,force_multi_y=True) #returns np.array(hist),np.array(ufuture),np.array(yfuture)

    def init_nets(self, nu, ny): # a bit weird
        ny = ny if ny is not None else 1
        nu = nu if nu is not None else 1
        self.encoder = self.net(n_in=self.nb*nu+self.na*ny, n_out=self.nx, n_nodes_per_layer=self.n_nodes_per_layer, n_hidden_layers=self.n_hidden_layers, activation=self.activation)
        self.fn =      self.net(n_in=self.nx+nu,            n_out=self.nx, n_nodes_per_layer=self.n_nodes_per_layer, n_hidden_layers=self.n_hidden_layers, activation=self.activation)
        self.hn =      self.net(n_in=self.nx,               n_out=ny,      n_nodes_per_layer=self.n_nodes_per_layer, n_hidden_layers=self.n_hidden_layers, activation=self.activation)
        return list(self.encoder.parameters()) + list(self.fn.parameters()) + list(self.hn.parameters())

    def loss(self, hist, ufuture, yfuture, **Loss_kwargs):
        x = self.encoder(hist)
        y_predict = []
        for u in torch.transpose(ufuture,0,1):
            y_predict.append(self.hn(x)) #output prediction
            fn_in = torch.cat((x,u),dim=1)
            x = self.fn(fn_in)
        return torch.mean((torch.stack(y_predict,dim=1)-yfuture)**2)

    ########## How to use ##############
    def init_state(self,sys_data): #put nf here for n-step error?
        hist = torch.tensor(sys_data.to_encoder_data(na=self.na,nb=self.nb,nf=len(sys_data)-max(self.na,self.nb))[0][:1],dtype=torch.float32) #(1,)
        with torch.no_grad():
            self.state = self.encoder(hist) #detach here?
        y_predict = self.hn(self.state).detach().numpy()[0,:]
        return (y_predict[0] if self.ny is None else y_predict), max(self.na,self.nb)

    def init_state_multi(self,sys_data,nf=100,dilation=1):
        hist = torch.tensor(sys_data.to_encoder_data(na=self.na,nb=self.nb,nf=nf,dilation=dilation)[0],dtype=torch.float32) #(1,)
        with torch.no_grad():
            self.state = self.encoder(hist)
        y_predict = self.hn(self.state).detach().numpy()
        return (y_predict[:,0] if self.ny is None else y_predict), max(self.na,self.nb)

    def reset(self): #to be able to use encoder network as a data generator
        self.state = torch.randn(1,self.nx)
        y_predict = self.hn(self.state).detach().numpy()[0,:]
        return (y_predict[0] if self.ny is None else y_predict)

    def step(self,action):
        action = torch.tensor(action,dtype=torch.float32) #number
        action = action[None,None] if self.nu is None else action[None,:]
        with torch.no_grad():
            self.state = self.fn(torch.cat((self.state,action),axis=1))
        y_predict = self.hn(self.state).detach().numpy()[0,:]
        return (y_predict[0] if self.ny is None else y_predict)

    def step_multi(self,action):
        action = torch.tensor(action,dtype=torch.float32) #array
        action = action[:,None] if self.nu is None else action
        with torch.no_grad():
            self.state = self.fn(torch.cat((self.state,action),axis=1))
        y_predict = self.hn(self.state).detach().numpy()
        return (y_predict[:,0] if self.ny is None else y_predict)

class SS_encoder_rnn(System_torch):
    """docstring for SS_encoder_rnn"""
    def __init__(self, hidden_size=10, num_layers=2, na=20, nb=20):
        super(SS_encoder_rnn, self).__init__(None,None)
        self.na = na
        self.nb = nb
        from deepSI.utils import simple_res_net, feed_forward_nn
        self.net = simple_res_net
        self.n_hidden_layers = 2
        self.n_nodes_per_layer = 64
        self.activation = nn.Tanh

        #RNN parameters
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def make_training_data(self, sys_data, **Loss_kwargs):
        assert sys_data.normed == True
        nf = Loss_kwargs.get('nf',25)
        return sys_data.to_encoder_data(na=self.na,nb=self.nb,nf=nf) #returns np.array(hist),np.array(ufuture),np.array(yfuture)

    def init_nets(self, nu, ny): # a bit weird
        # print(nu,ny)
        assert ny==None and nu==None
        ny = 1
        nu = 1

        self.rnn = nn.RNN(input_size=nu,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True) #batch_first yes
        # output, h_n = self.RNN(input,h_0) #input = (batch, seq_len, input_size), 
        #h_0 = (num_layers, batch, hidden_size)
        #outputs = (batch, seq_len, hidden_size) #last layer

        #encoder: self.nb + self.na -> h_0 = (num_layers, batch, hidden_size)
        #hn: (batch*seq_len, hidden_size) -> (batch*seq_len, ny)
        self.encoder = self.net(n_in=self.nb+self.na, n_out=self.hidden_size*self.num_layers, n_nodes_per_layer=self.n_nodes_per_layer, n_hidden_layers=self.n_hidden_layers, activation=self.activation)
        self.hn =      self.net(n_in=self.hidden_size,n_out=ny,      n_nodes_per_layer=self.n_nodes_per_layer, n_hidden_layers=self.n_hidden_layers, activation=self.activation)
        return list(self.encoder.parameters()) + list(self.rnn.parameters()) + list(self.hn.parameters())

    def loss(self, hist, ufuture, yfuture, **Loss_kwargs):
        x = self.encoder(hist) # (s, nhist = nb + na) -> (s, hidden_size*num_layers)
        h_0 = x.view(-1, self.num_layers, self.hidden_size).permute(1,0,2)   # to (num_layers, s, hidden_size)

        # ufuture (batch, seq_len)
        output, h_n = self.rnn(ufuture[:,:-1,None], h_0) #do not use the last u
        #print(output.shape) #has shape (s, seq_len-1, hidden_size)
        output = torch.cat((h_0[-1][:,None,:],output),dim=1)
        #outputs = (batch, seq_len, hidden_size) -> (batch*seq_len, hidden_size)
        h_in = output.reshape(-1,self.hidden_size)
        y_predict = self.hn(h_in).view(output.shape[0], output.shape[1])
        return torch.mean((y_predict-yfuture)**2)

    def init_state(self,sys_data): #put nf here for n-step error?
        hist = torch.tensor(sys_data.to_encoder_data(na=self.na,nb=self.nb,nf=len(sys_data)-max(self.na,self.nb))[0][:1],dtype=torch.float32) #(1,)
        self.state = self.encoder(hist).view(-1, self.num_layers, self.hidden_size).permute(1,0,2)
        return self.hn(self.state[-1,:,:])[0,0].item(), max(self.na,self.nb) #some error is being made here

    def init_state_multi(self,sys_data,nf=100,dilation=1):
        hist = torch.tensor(sys_data.to_encoder_data(na=self.na,nb=self.nb,nf=nf,dilation=dilation)[0],dtype=torch.float32) #(1,)
        self.state = self.encoder(hist).view(-1, self.num_layers, self.hidden_size).permute(1,0,2)
        return self.hn(self.state[-1,:,:])[:,0].detach().numpy(), max(self.na,self.nb)

    def step(self,action):
        action = torch.tensor(action,dtype=torch.float32) #number
        output, self.state = self.rnn(action[None,None,None], self.state)
        return self.hn(output[:,0,:])[0,0].item()

    def step_multi(self,action):
        action = torch.tensor(action,dtype=torch.float32) #array
        output, self.state = self.rnn(action[:,None,None], self.state)
        return self.hn(output[:,0,:])[:,0].detach().numpy()

if __name__ == '__main__':
    sys = SS_encoder()
    from deepSI.datasets.sista_database import powerplant
    from deepSI.datasets import Silverbox
    train, test = Silverbox()#powerplant()
    # train, test = train[:150], test[:50]
    print(train, test)
    # sys.fit(train, sim_val=test,epochs=50)
    import deepSI
    test2 = deepSI.system_data.System_data_list([test,test])
    sys.fit_val_multiprocess(train, sim_val=test2,epochs=50)

    # fit_val_multiprocess
    train_predict = sys.apply_experiment(train)
    train.plot()
    train_predict.plot(show=True)
    from matplotlib import pyplot as plt
    plt.plot(sys.n_step_error(train,nf=20))
    plt.show()
