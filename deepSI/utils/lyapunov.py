import torch
import numpy as np
from torch.autograd.functional import jacobian
from matplotlib import pyplot as plt

def get_lyapunov_exponent(sys, test, nsteps = 15, n_samp=100, verbose=1 ):

    test_p = sys.apply_experiment(test, save_state=True)

    xt = torch.tensor(test_p.x,dtype=torch.float32) #this is not normalized
    ut = torch.tensor(sys.norm.transform(test_p).u,dtype=torch.float32)

    def f(x,U):
        X = [x]
        for u in U:
            xu = torch.cat([x[None],u[None,None]],dim=1)
            x = sys.fn(xu)[0]
            X.append(x)
        return torch.stack(X,dim=0)

    eigs = []
    #t from n_cheat to len()
    for _ in range(n_samp):
        t = np.random.randint(test_p.cheat_n,high=len(test_p)-nsteps)
        out = jacobian(lambda x: f(x,ut[t:t+nsteps]), xt[t]).numpy()
        eigs.append(np.max(np.abs(np.linalg.eigvals(out)),axis=1))
    eigs=np.array(eigs)

    x = np.array([np.arange(eigs.shape[1])]*len(eigs)).flatten()
    y = np.log(eigs).flat

    a, _, _, _ = np.linalg.lstsq(x[:,None], y, rcond=None)
    lambdt = a[0]
    if verbose:
        plt.plot(x,y,'.b')
        plt.plot(x,lambdt*x,'o--')
        plt.xlabel('n steps')
        plt.ylabel('Largest eigen value')
        plt.show()

    return lambdt #returns lambda*dt = -dt/tau