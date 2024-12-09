
import nonlinear_benchmarks as nlb
import numpy as np
import torch

C = lambda x: torch.as_tensor(x, dtype=torch.float32) if x is not None else None
class IO_normalization_f(torch.nn.Module):
    def __init__(self, fun, umean, ustd):
        super().__init__()
        self.fun, self.umean, self.ustd = fun, C(umean), C(ustd)    
    def forward(self, x, u):
        return self.fun(x, (u-self.umean)/self.ustd)

class IO_normalization_f_CT(torch.nn.Module):
    def __init__(self, fun, umean, ustd, tau):
        super().__init__()
        self.fun, self.umean, self.ustd, self.tau = fun, C(umean), C(ustd), C(tau)
    def forward(self, x, u):
        return self.fun(x, (u-self.umean)/self.ustd)/self.tau

class IO_normalization_h(torch.nn.Module):
    def __init__(self, fun, umean, ustd, ymean, ystd):
        super().__init__()
        self.fun, self.umean, self.ustd, self.ymean, self.ystd = fun, C(umean), C(ustd), C(ymean), C(ystd)
    def forward(self, x, u=None):
        if u is None:
            y_normed = self.fun(x)
        else:
            y_normed = self.fun(x, (u-self.umean)/self.ustd)
        return y_normed*self.ystd + self.ymean

class IO_normalization_encoder(torch.nn.Module):
    def __init__(self, fun, umean, ustd, ymean, ystd):
        super().__init__()
        self.fun, self.umean, self.ustd, self.ymean, self.ystd = fun, C(umean), C(ustd), C(ymean), C(ystd)
    def forward(self, upast, ypast):
        return self.fun((upast-self.umean)/self.ustd, (ypast-self.ymean)/self.ystd)

class Norm:
    def __init__(self, umean, ustd, ymean, ystd, sampling_time=1):
        self.umean, self.ustd, self.ymean, self.ystd = C(umean), C(ustd), C(ymean), C(ystd)
        self.sampling_time = C(sampling_time)
    
    def f(self, fun):
        return IO_normalization_f(fun, self.umean, self.ustd)
    def h(self, fun):
        return IO_normalization_h(fun, self.umean, self.ustd, self.ymean, self.ystd)
    def encoder(self, fun):
        return IO_normalization_encoder(fun, self.umean, self.ustd, self.ymean, self.ystd)
    def f_CT(self, fun, tau):
        return IO_normalization_f_CT(fun, self.umean, self.ustd, tau)

    def transform(self, dataset : nlb.Input_output_data | list):
        if isinstance(dataset, (list, tuple)):
            return [self.transform(d) for d in dataset]
        u = (dataset.u - self.umean.numpy())/self.ustd.numpy()
        y = (dataset.y - self.ymean.numpy())/self.ystd.numpy()
        sampling_time = None if dataset.sampling_time is None else dataset.sampling_time/self.sampling_time.item()
        return nlb.Input_output_data(u, y, sampling_time=sampling_time, name=f'{dataset.name}-normed', \
                                     state_initialization_window_length=dataset.state_initialization_window_length)

    def __repr__(self):
        return (f"Norm(umean={self.umean.numpy()}, ustd={self.ustd.numpy()}, "
                f"ymean={self.ymean.numpy()}, ystd={self.ystd.numpy()}, "
                f"sampling_time={self.sampling_time.numpy()})")

def get_nu_ny_and_auto_norm(data: nlb.Input_output_data | list):
    if not isinstance(data, (tuple, list)):
        data = [data]
    u = np.concatenate([d.u for d in data],axis=0)
    y = np.concatenate([d.y for d in data],axis=0)
    assert u.ndim<=2 and y.ndim<=2, f'auto norm only defined for scalar or vector outputs y and input u {y.shape=} {u.shape=}'
    sampling_time = data[0].sampling_time
    assert all(sampling_time==d.sampling_time for d in data), f"the given datasets don't have all the sample sampling_time set {[d.sampling_time for d in data]=}"
    umean, ustd = u.mean(0), u.std(0)
    ymean, ystd = y.mean(0), y.std(0)
    norm = Norm(umean, ustd, ymean, ystd, sampling_time)
    nu = 'scalar' if u.ndim==1 else u.shape[1]
    ny = 'scalar' if y.ndim==1 else y.shape[1]
    return nu, ny, norm





