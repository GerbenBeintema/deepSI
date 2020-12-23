from deepSI.systems.System import System, System_ss, System_data
from deepSI.fit_systems import System_fittable
import deepSI
import numpy as np
from tqdm import tqdm as tqdm
from warnings import warn
import math
import scipy as sc
from numpy.linalg import pinv


#adapted from https://github.com/CPCLAB-UNIPI/SIPPY/blob/master/sippy/OLSims_methods.py

def check_types(threshold, max_order, fixed_order, f, p=20):
    if threshold < 0. or threshold >= 1.:
        print("Error! The threshold value must be >=0. and <1.")
        return False
    if (np.isnan(max_order)) == False:
        if type(max_order) != int:
            print("Error! The max_order value must be integer")
            return False
    if (np.isnan(fixed_order)) == False:
        if type(fixed_order) != int:
            print("Error! The fixed_order value must be integer")
            return False
    if type(f) != int:
        print("Error! The future horizon (f) must be integer")
        return False
    if type(p) != int:
        print("Error! The past horizon (p) must be integer")
        return False
    return True


def check_inputs(threshold, max_order, fixed_order, f):
    if (math.isnan(fixed_order)) == False:
        threshold = 0.0
        max_order = fixed_order
    if f < max_order:
        print('Warning! The horizon must be larger than the model order, max_order setted as f')
    if (max_order < f) == False:
        max_order = f
    return threshold, max_order

def rescale(y):
    ystd = np.std(y)
    y_scaled = y/ystd
    return ystd, y_scaled

def ordinate_sequence(y, f, p):
    [l, L] = y.shape
    N = L - p - f + 1
    Yp = np.zeros((l * f, N))
    Yf = np.zeros((l * f, N))
    for i in range(1, f + 1):
        Yf[l * (i - 1):l * i] = y[:, p + i - 1:L - f + i]
        Yp[l * (i - 1):l * i] = y[:, i - 1:L - f - p + i]
    return Yf, Yp

def PI_PIort(X):
    pinvXT = pinv(X.T)
    PI = np.dot(X.T, pinvXT) #memeory error!
    PIort = np.identity((PI[:, 0].size)) - PI
    return PI, PIort

def reducingOrder(U_n, S_n, V_n, threshold=0.1, max_order=10):
    s0 = S_n[0]
    index = S_n.size
    for i in range(S_n.size):
        if S_n[i] < threshold * s0 or i >= max_order:
            index = i
            break
    return U_n[:, 0:index], S_n[0:index], V_n[0:index, :] if V_n is not None else None

def SVD_weighted(y, u, f, l, weights='N4SID'):
    Yf, Yp = ordinate_sequence(y, f, f)
    Uf, Up = ordinate_sequence(u, f, f)
    Zp = impile(Up, Yp)
    O_i = (Yf - (Yf@Uf.T)@pinv(Uf.T))@pinv(Zp - (Zp@Uf.T)@pinv(Uf.T))@Zp
    if weights == 'MOESP':
        W1 = None
        U_n, S_n, V_n = np.linalg.svd(O_i - (O_i@Uf.T)@pinv(Uf.T),full_matrices=False)
    elif weights == 'CVA':
        #todo, makes this memory effective if it is possible to due so.
        _, PIort_Uf = PI_PIort(Uf)#Will give a memory error for large metricies
        W1 = np.linalg.inv(
            sc.linalg.sqrtm(np.dot(np.dot(Yf, PIort_Uf), np.dot(Yf, PIort_Uf).T)).real)
        W2 = 1. * PIort_Uf
        U_n, S_n, V_n = np.linalg.svd(np.dot(np.dot(W1, O_i), W2),full_matrices=False)
    elif weights == 'N4SID':
        U_n, S_n, V_n = np.linalg.svd(O_i,full_matrices=False)
        W1 = None
    return U_n, S_n, V_n, W1, O_i

def impile(M1, M2):
    M = np.zeros((M1[:, 0].size + M2[:, 0].size, M1[0, :].size))
    M[0:M1[:, 0].size] = M1
    M[M1[:, 0].size::] = M2
    return M

def algorithm_1(y, u, l, m, f, N, U_n, S_n, V_n, W1, O_i, threshold, max_order, D_required):
    U_n, S_n, V_n = reducingOrder(U_n, S_n, V_n, threshold, max_order)
    # V_n = V_n.T #V_n not used in this function
    n = S_n.size
    S_n = np.diag(S_n)
    if W1==None: #W1 is identity
        Ob = np.dot(U_n, sc.linalg.sqrtm(S_n))
    else:
        Ob = np.dot(np.linalg.inv(W1), np.dot(U_n, sc.linalg.sqrtm(S_n)))
    X_fd = np.dot(np.linalg.pinv(Ob), O_i)
    Sxterm = impile(X_fd[:, 1:N], y[:, f:f + N - 1])
    Dxterm = impile(X_fd[:, 0:N - 1], u[:, f:f + N - 1])
    if D_required == True:
        M = np.dot(Sxterm, np.linalg.pinv(Dxterm))
    else:
        M = np.zeros((n + l, n + m))
        M[0:n, :] = np.dot(Sxterm[0:n], np.linalg.pinv(Dxterm))
        M[n::, 0:n] = np.dot(Sxterm[n::], np.linalg.pinv(Dxterm[0:n, :]))
    residuals = Sxterm - np.dot(M, Dxterm)
    return Ob, X_fd, M, n, residuals

def forcing_A_stability(M, n, Ob, l, X_fd, N, u, f):
    Forced_A = False
    if np.max(np.abs(np.linalg.eigvals(M[0:n, 0:n]))) >= 1.:
        Forced_A = True
        print("Forcing A stability")
        M[0:n, 0:n] = np.dot(np.linalg.pinv(Ob), impile(Ob[l::, :], np.zeros((l, n))))
        M[0:n, n::] = np.dot(X_fd[:, 1:N] - np.dot(M[0:n, 0:n], X_fd[:, 0:N - 1]),
                             np.linalg.pinv(u[:, f:f + N - 1]))
    res = X_fd[:, 1:N] - np.dot(M[0:n, 0:n], X_fd[:, 0:N - 1]) - np.dot(M[0:n, n::],
                                                                        u[:, f:f + N - 1])
    return M, res, Forced_A


def extracting_matrices(M, n):
    A = M[0:n, 0:n]
    B = M[0:n, n::]
    C = M[n::, 0:n]
    D = M[n::, n::]
    return A, B, C, D

def K_calc(A, C, Q, R, S):
    n_A = A[0, :].size
    try:
        P, L, G = cnt.dare(A.T, C.T, Q, R, S, np.identity(n_A))
        K = np.dot(np.dot(A, P), C.T) + S
        K = np.dot(K, np.linalg.inv(np.dot(np.dot(C, P), C.T) + R))
        Calculated = True
    except:
        K = []
        # print("Kalman filter cannot be calculated")
        Calculated = False
    return K, Calculated


def OLSims(y, u, f, weights='N4SID', threshold=0.1, max_order=np.NaN, fixed_order=np.NaN,
           D_required=False, A_stability=False):
    ''' y array 
        u array
        SS_f = f = 20, #future?
        id_method = weights = 'N4SID' or 'MOESP' or 'CVA'
        SS_threshold = threshold = 0.1
        SS_max_order = np.NaN order of the model
        SS_fixed_order = (np.NaN) or 2
        SS_D_required = False
        SS_A_stability = False'''


    y = 1. * np.atleast_2d(y)
    u = 1. * np.atleast_2d(u)
    l, L = y.shape
    m = u[:, 0].size
    if check_types(threshold, max_order, fixed_order, f) == False:
        return np.array([[0.0]]), np.array([[0.0]]), np.array([[0.0]]), np.array(
                [[0.0]]), np.inf, [], [], [], []
    else:
        threshold, max_order = check_inputs(threshold, max_order, fixed_order, f)
        N = L - 2 * f + 1
        Ustd = np.zeros(m)
        Ystd = np.zeros(l)
        for j in range(m):
            Ustd[j], u[j] = rescale(u[j])
        for j in range(l):
            Ystd[j], y[j] = rescale(y[j])
        # print('SVD_weighted')
        U_n, S_n, V_n, W1, O_i = SVD_weighted(y, u, f, l, weights) #here is a memory error
        Ob, X_fd, M, n, residuals = algorithm_1(y, u, l, m, f, N, U_n, S_n, V_n, W1, O_i, threshold,
                                                max_order, D_required) 
        if A_stability == True:
            M, residuals[0:n, :], useless = forcing_A_stability(M, n, Ob, l, X_fd, N, u, f)
        A, B, C, D = extracting_matrices(M, n)
        Covariances = np.dot(residuals, residuals.T)/(N - 1)
        Q = Covariances[0:n, 0:n]
        R = Covariances[n::, n::]
        S = Covariances[0:n, n::]
        X_states, Y_estimate = SS_lsim_process_form(A, B, C, D, u)
        Vn = np.trace(np.dot((y - Y_estimate), (y - Y_estimate).T))/ (2 * L)
        K, K_calculated = K_calc(A, C, Q, R, S)
        for j in range(m):
            B[:, j] = B[:, j]/Ustd[j]
            D[:, j] = D[:, j]/Ustd[j]
        for j in range(l):
            C[j, :] = C[j, :] * Ystd[j]
            D[j, :] = D[j, :] * Ystd[j]
            if K_calculated == True:
                K[:, j] = K[:, j]/Ystd[j]
        return A, B, C, D, Vn, Q, R, S, K

class SS_model(object):
    def __init__(self, A, B, C, D, K, Q, R, S, ts, Vn):
        self.n = A[:, 0].size
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Vn = Vn
        self.Q = Q
        self.R = R
        self.S = S
        self.K = K
        # self.G = cnt.ss(A, B, C, D, ts)
        self.ts = ts
        self.x0 = np.zeros((A[:, 0].size, 1))
        try:
            A_K = A - np.dot(K, C)
            B_K = B - np.dot(K, D)
        except:
            A_K = []
            B_K = []
        self.A_K = A_K
        self.B_K = B_K

def SS_lsim_process_form(A, B, C, D, u, x0=None):
    m, L = u.shape
    l, n = C.shape
    y = np.zeros((l, L))
    x = np.zeros((n, L))
    if x0 is not None:
        x[:, 0] = x0[:, 0]
    y[:, 0] = np.dot(C, x[:, 0]) + np.dot(D, u[:, 0])
    for i in range(1, L):
        x[:, i] = np.dot(A, x[:, i - 1]) + np.dot(B, u[:, i - 1])
        y[:, i] = np.dot(C, x[:, i]) + np.dot(D, u[:, i])
    return x, y

class SS_linear(System_ss, System_fittable):
    def __init__(self,seed=None,A=None,B=None,C=None,D=None,nx=2):
        self.ny = None
        self.nu = None
        self.A, self.B, self.C, self.D = A, B, C, D
        if A is not None:
            self.fitted = True
            nx = A.shape[0]
            self.ny = C.shape[0]
            self.nu = B.shape[1]
        super(SS_linear, self).__init__(nx=nx)

    def _fit(self,sys_data,SS_A_stability=False,SS_f=20):
        assert isinstance(sys_data,System_data), 'todo for multiple data sets'


        y = sys_data.y.T if sys_data.y.ndim==2 else sys_data.y.T[None,:] #work with (features,time)
        u = sys_data.u.T if sys_data.u.ndim==2 else sys_data.u.T[None,:] #work with (features,time)
        self.ny, self.nu = y.shape[0], u.shape[0]

        # SS_f = 20 #future steps?
        SS_fixed_order = self.nx
        id_method = 'N4SID' #'N4SID' or 'MOESP' or 'CVA'
        SS_threshold = 0.1
        SS_max_order=np.NaN
        SS_D_required=False
        tsample = 1.

        #fit
        A, B, C, D, Vn, Q, R, S, K = OLSims(y, u, SS_f, id_method, SS_threshold, \
                                            SS_max_order, SS_fixed_order, \
                                            SS_D_required, SS_A_stability)
        self.model = SS_model(A, B, C, D, K, Q, R, S, tsample, Vn) #useless?, wait....
        self.A, self.B, self.C, self.D = A, B, C, D
        # sys_data_sim = self.apply_experiment(self.norm.inverse_transform(sys_data)) #multiple sys_data todo
        X = []
        x = np.zeros((self.nx,))
        for u in sys_data.u:
            X.append(x)
            x = self.f(x,u)
        assert np.any(np.isnan(X))==False, 'x exploded, consider lowering nx increasing SS_f or SS_A_stability=True'
        T = np.diag(1/np.std(X,axis=0))
        Tinv = np.linalg.inv(T)
        self.A = T@self.A@Tinv
        self.B = T@self.B
        self.C = self.C@Tinv
        self._nx = A.shape[0]
        self.ny = C.shape[0]
        self.nu = B.shape[1]


    def f(self,x,u):
        u = np.array(u) 
        if u.ndim==0:
            u = u[None]
        return np.dot(self.A, x) + np.dot(self.B, u)

    def h(self,x):
        # u = np.array(u) 
        # if u.ndim==0:
        #     u = u[None]
        yhat = np.dot(self.C, x) #+ np.dot(self.D, u)
        return yhat[0] if self.ny==1 or self.ny==None else yhat



if __name__=='__main__':
    A = np.array([[0.89, 0.], [0., 0.45]]) #(2,2) x<-x
    B = np.array([[0.3], [2.5]]) #(2,1) x<-u
    C = np.array([[0.7, 1.]]) #(1,2) y<-u
    D = np.array([[0.0]]) #(1,1) 
    sys = SS_linear(A=A,B=B,C=C,D=D)
    exp = System_data(u=np.random.normal(size=1000)[:,None])
    sys_data = sys.apply_experiment(exp)
    # sys_data.plot(show=True)


    sys_fit = SS_linear()
    sys_fit.fit(sys_data)
    sys_data_predict = sys_fit.apply_experiment(sys_data)
    print(sys_data_predict.NRMS(sys_data))
    # sys_data_fitted = sys_fit.simulation(sys_data)
    # print(sys_fit.A,sys_fit.B,sys_fit.C,sys_fit.D)
    # sys_data.plot(show=False)
    # sys_data_fitted.plot()
    # print(sys_data_fitted.NMSE(sys_data))

    sys = deepSI.systems.nonlin_Ibased_normals_system()
    train_data = sys.get_train_data()
    sys_data = sys.get_test_data()

    best_score, best_sys, best_sys_dict, best_fit_dict = deepSI.fit_systems.grid_search(SS_linear, train_data, sys_dict_choices=dict(nx=[3,4,5,6]), fit_dict_choices=dict(SS_A_stability=[True,False],SS_f=[3,4,5,8,10]), sim_val=sys_data, RMS=False, verbose=2)

    print(best_score, best_sys, best_sys_dict, best_fit_dict)
