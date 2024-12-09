# This document is used to generate the example that is seen in the README.md. 

import deepSI_lite as dsi
import numpy as np

# Generate data 
np.random.seed(0)
ulist = np.random.randn(10_000) #input sequence
x = [0, 0] #initial state
ylist = [] #output sequence
for uk in ulist:
    ylist.append(x[1]*x[0]*0.1 + x[0] + np.random.randn()*1e-3)  #compute output
    x = x[0]/(1.2+x[1]**2) + x[1]*0.4, \
        x[1]/(1.2+x[0]**2) + x[0]*0.4 + uk*(1+x[0]**2/10) #advance state

# Put the inputs and outputs in a Input_output_data format
data = dsi.Input_output_data(u=ulist, y=np.array(ylist)) 

# Split dataset
train, val, test  = data[:8000], data[8000:9000], data[9000:]

# Create model
nu, ny, norm = dsi.get_nu_ny_and_auto_norm(data) # Characterize data
model = dsi.SUBNET(nu, ny, norm, nx=2, nb=20, na=20) # Creates encoder, f and h as MLP

# Train model on data
if False:
    train_dict = dsi.fit(model, train, val, n_its=10_000, T=20, batch_size=256, val_freq=100) #Adam
else:    
    import cloudpickle
    folder = dsi.fitting.get_checkpoint_dir()
    train_dict = cloudpickle.load(open(folder + '/SUBNET-GENrrw.pth', 'rb'))
    model = train_dict['best_model']

# Simulate model on the test input sequence
test_p = model.simulate(test)

import matplotlib.pyplot as plt

# Improved plot settings
plt.figure(figsize=(10, 4), dpi=120)
plt.plot(test.y, label='Real Data', color='dodgerblue', linewidth=1.5)
plt.plot(test_p.y, label=f'Model (NRMSE = {((test.y - test_p.y)**2).mean()**0.5 / test.y.std():.2%})', color='darkorange', linestyle='--', linewidth=1.5)

# Add labels, legend, and grid
plt.xlabel('Time Index', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Comparison of Real Data and Model Simulation', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='upper right')
plt.grid(visible=True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout(pad=0.8)
plt.savefig('NL-example.jpg')
plt.show()
