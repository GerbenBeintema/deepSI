## deepSI

deepSI provides a lightweight pytorch based framework for data-driven learning of dynamical systems (i.e. system identification). It contains a large forcus on the SUBNET method which is able to robustly model many systems.

### ⚠️ deepSI has been refractored without backward compatibility (5 December 2024) ⚠️

If you need to install the legacy version you can do this with 
```bash
pip install git+https://github.com/GerbenBeintema/deepSI@legacy
```
and using the legacy documentation provided in [https://github.com/GerbenBeintema/deepSI/tree/legacy](https://github.com/GerbenBeintema/deepSI/tree/legacy).

## Example usage

```python
# Imports
import numpy as np
import deepSI as dsi

# Generate or load data 
np.random.seed(0)
ulist = np.random.randn(10_000) # Input sequence
x = [0, 0] # Initial state
ylist = [] # Output sequence
for uk in ulist:
    ylist.append(x[1]*x[0]*0.1 + x[0] + np.random.randn()*1e-3)  # Compute output
    x = x[0]/(1.2+x[1]**2) + x[1]*0.4, \
        x[1]/(1.2+x[0]**2) + x[0]*0.4 + uk*(1+x[0]**2/10) # Advance state

# Put the input and output sequence in the Input_output_data format
data = dsi.Input_output_data(u=ulist, y=np.array(ylist)) 

# Split dataset
train, val, test  = data[:8000], data[8000:9000], data[9000:]

# Create model
nu, ny, norm = dsi.get_nu_ny_and_auto_norm(data) # Characterize data
model = dsi.SUBNET(nu, ny, norm, nx=2, nb=20, na=20) # Creates encoder, f and h as MLP

# Train model on data using Adam
train_dict = dsi.fit(model, train, val, n_its=10_000, T=20, batch_size=256, val_freq=100)

# Simulate model on the test input sequence (using the encoder to initialize the state)
test_p = model.simulate(test)

# Visualize simulation of the model
from matplotlib import pyplot as plt
plt.figure(figsize=(7,3))
plt.plot(test.y, label='Real Data')
plt.plot(test_p.y, label=f'Model Sim. (NRMS = {((test.y-test_p.y)**2).mean()**0.5/test.y.std():.2%})', linestyle='--')
plt.title('Comparison of Real Data and Model Simulation', fontsize=14, fontweight='bold')
plt.legend(); plt.xlabel('Time Index'); plt.ylabel('y'); plt.grid(); plt.tight_layout(pad=0.5)
plt.show()
```

![dsi SUBNET result on example](examples/docs/NL-example.jpg)

## Installation

```bash
conda install -c anaconda git
pip install git+https://github.com/GerbenBeintema/deepSI@main
```

## Features

* A number of popular SUBNET model structures
  * SUBNET encoder structue (`deepSI.models.SUBNET`). Featuring in: [\[1\]](https://proceedings.mlr.press/v144/beintema21a), [\[2\]](https://www.sciencedirect.com/science/article/pii/S2405896321012167), [\[3\]](https://www.sciencedirect.com/science/article/pii/S2405896321012180), [\[4\]](https://arxiv.org/abs/2303.17305), [\[5\]](https://arxiv.org/abs/2304.02119)
  * Continuous time SUBNET encoder structure (`deepSI.models.SUBNET_CT`). Featuring in: [\[1\]](https://arxiv.org/abs/2204.09405), [\[2\]](https://www.sciencedirect.com/science/article/pii/S2405896324013223), [\[3\]](https://www.sciencedirect.com/science/article/pii/S240589632401317X)
  * Base class for fully custom SUBNET structures with shared parameters between `f`, `h` or `encoder`. (`deepSI.models.Custom_SUBNET`) as used in:
  * CNN SUBNET (`CNN_SUBNET`). Featuring in: [\[1\]](https://research.tue.nl/files/318935789/20240321_Beintema_hf.pdf) Chapter 4, [\[2\]](https://www.sciencedirect.com/science/article/pii/S2405896321012167)
  * LPV SUBNET (`SUBNET_LPV` and `SUBNET_LPV_ext_scheduled`). Featuring in: [\[1\]](https://arxiv.org/abs/2204.04060)
  * port HNN SUBNET (`pHNN_SUBNET`). Featuring in: [\[1\]](https://arxiv.org/abs/2305.01338)
  * Koopman SUBNET (`Koopman_SUBNET`). Featuring in: [\[1\]](https://ieeexplore.ieee.org/abstract/document/9682946)
* Connection to [`nonlinear_benchmarks`](https://github.com/GerbenBeintema/nonlinear_benchmarks) to easily load and evaluate on benchmarks.
* Low amount of code such that it can be easily forked and edited to add missing features.

## Futher documentation

Check out [`examples/1. Overview deepSI.ipynb`](examples/1.%20Overview%20deepSI.ipynb).

## Contributing

deepSI is in ongoing development and anyone can contribute to any part of module.

## todo list and known issues

* Expand demonstration notebook with pHNN examples
* Issue where discrete time is printed in Input_output_data with torch.Tensors, and np.arrays sample time.
* General documentation 
* known issue: CT SUBNET and DT SUBNET does not produce the correct initial when the sampling time is altered. (the encoder assumes that the sampling time does not change)
* pypi data upload such that it can be easily installed with `pip install deepSI`
* Improve speed with copy if enough memory is available. Also pre-transfer to GPU and maybe asyncroness getting of arrays.
* Known issue: Using the compile function in `fit` will sometimes result in a memory leak
