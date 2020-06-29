import torch
from torch import nn, optim


class feed_forward_nn(nn.Module): #for encoding
    def __init__(self,n_in=6,n_out=5,n_nodes_per_layer=64,n_hidden_layers=2,activation=nn.Tanh):
        super(feed_forward_nn,self).__init__()
        seq = [nn.Linear(n_in,n_nodes_per_layer),activation()]
        assert n_hidden_layers>0
        for i in range(n_hidden_layers-1):
            seq.append(nn.Linear(n_nodes_per_layer,n_nodes_per_layer))
            seq.append(activation())
        seq.append(nn.Linear(n_nodes_per_layer,n_out))
        self.net = nn.Sequential(*seq)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, val=0) #bias
    def forward(self,X):
        return self.net(X)        


class simple_res_net(nn.Module):
    def __init__(self,n_in=6,n_out=5,n_nodes_per_layer=64,n_hidden_layers=2,activation=nn.Tanh):
        #linear + non-linear part 
        super(simple_res_net,self).__init__()
        self.net_lin = nn.Linear(n_in,n_out)
        self.net_non_lin = feed_forward_nn(n_in,n_out,n_nodes_per_layer=n_nodes_per_layer,n_hidden_layers=n_hidden_layers,activation=activation)
    def forward(self,x):
        return self.net_lin(x) + self.net_non_lin(x)
        