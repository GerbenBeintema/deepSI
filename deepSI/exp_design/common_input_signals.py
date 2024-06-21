from scipy.fftpack import *
import numpy as np

crest_factor = lambda uk: np.max(np.abs(uk))/np.sqrt(np.mean(uk**2))
duplicate = lambda uk,n: np.concatenate([uk]*n)

def multisine(N_points_per_period, N_periods=1, pmin=1, pmax=21, prule=lambda p: p%2==1 and p%6!=1, par=None, 
            n_crest_factor_optim=1, seed=None):
    '''A multi-sine geneator with only odd frequences and random phases. 

    Paramters
    ---------
    N_points_per_period : int
    N_periods : int
    pmin : int
        The lowest number of sin periods allowed in the signal
    pmax : int
        The hightest number of sin periods allowed in the signal
    prule : lambda function
        A function which is true if the p should be allowed. Often used to only select odd frequences
        By default it will allow p%2==1 and p%6!=1 as: [3, 5, 9, 11, 15, 17, 21, 23, 27, 29, 33, 35, 39, 41, 45...]
    par : list of ints like
        Manual list of sin periods in the signal (note: overwrites prule)
    n_crest_factor_optim : int
        n random trails to mimize the crest factor (max(y)/std(y))
    seed : int, None or RandomState
        the random seed used for the generation
    

    fmax = pmax/(N_points_per_period*sample time)
    '''
    if isinstance(seed,int) or seed is None:
        rng = np.random.RandomState(seed)
    else:
        rng = seed
    assert isinstance(rng, np.random.mtrand.RandomState)

    assert pmax<N_points_per_period//2
    #crest factor optim:
    if n_crest_factor_optim>1:
        ybest = None
        crest_best = float('inf')
        for i in range(n_crest_factor_optim):
            seedi = None if seed is None else seed + i
            uk = multisine(N_points_per_period, N_periods=1, pmax=pmax, pmin=pmin, prule=prule, n_crest_factor_optim=1, seed=seedi)
            crest = crest_factor(uk)
            if crest<crest_best:
                ybest = uk
                crest_best = crest
        return duplicate(ybest, N_periods)

    N = N_points_per_period

    uf = np.zeros((N,),dtype=complex)
    for p in range(pmin,pmax) if par==None else par:
        if par==None and not prule(p):
            continue
        uf[p] = np.exp(1j*rng.uniform(0,np.pi*2))
        uf[N-p] = np.conjugate(uf[p])

    uk = np.real(ifft(uf/2)*N)
    uk /= np.std(uk)

    return duplicate(uk, N_periods)

def filtered_signal(N_points_per_period, N_periods=1, fmax=0.1, q=1, transient_periods=5, rng=None):
    '''Generate a signal from filtered uniform noise where u**(1/q) is returned'''
    from scipy import signal
    if isinstance(rng, int):
        rng = np.random.RandomState(rng)
    u0 = np.random.normal(size=N_points_per_period) if rng is None else rng.normal(size=N_points_per_period)
    u0 = duplicate(u0,N_periods+transient_periods)
    if fmax>=1: 
        u1 = u0
    else:
        u1 = signal.lfilter(*signal.butter(6,fmax),u0) #sixth order Butterworth filter

    u1 = u1[N_points_per_period*transient_periods:]
    u1 /= max(np.max(u1),-np.min(u1)) #standardize
    if q==1:
        return u1
    else:
        return abs(u1)**(1/q)*np.sign(u1) if np.isfinite(q) else np.sign(u1)