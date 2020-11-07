#!/usr/bin/env python
# coding: utf-8

# In[1]:


import uxyeye
import deepSI
from matplotlib import pyplot as plt


# In[2]:


# sys_data_deepSI = 
sys_data_deepSI = deepSI.datasets.WienerHammerBenchMark(split_data=False)#sys_data_deepSI[:134020], sys_data_deepSI[134020:]
sys_data_uxyeye = uxyeye.data_sets.WienerHammerBenchMark(split_data=False)


# In[3]:


sys_train_deepSI,sys_test_deepSI = sys_data_deepSI[:134020], sys_data_deepSI[134020:]
sys_train_uxyeye,sys_test_uxyeye = sys_data_uxyeye[:134020], sys_data_uxyeye[134020:]


# In[4]:


sys_uxyeye = uxyeye.fit_systems.statespace_encoder_system_base(na=50,nb=50,nx=8,n_nodes_per_layer=60,n_hidden_layers=2,net_base=uxyeye.utils.torch_nets.modified_Lin_network)


# In[5]:


sys_deepSI = deepSI.fit_systems.Fit_system.System_encoder(nx=8,na=50,nb=50)


# In[12]:


sys_uxyeye.fit(sys_train_uxyeye,verbose=2,epochs=100,nf=30,sim_val=sys_test_uxyeye[:5000],batch_size=1024)


# In[6]:


# sys_uxyeye.save_system(dir_placement='./',name='encoder_uxyeye_best')
# sys_uxyeye.load_system(dir_placement=sys_uxyeye.checkpoint_dir,name='_last')
# sys_uxyeye.save_system(dir_placement='./',name='encoder_uxyeye_last')
sys_uxyeye.load_system(dir_placement='./',name='encoder_uxyeye_best')


# In[7]:


#SAVE!
#OPTIMIZER


# In[8]:


sys_deepSI.fit(sys_train_deepSI, epochs=0, verbose=1, batch_size=1024, Loss_kwargs=dict(nf=30), sim_val=sys_test_deepSI[:5000])


# In[9]:


from torch import optim

sys_deepSI.optimizer = optim.Adam(sys_deepSI.paremters)


# In[10]:


sys_deepSI.fit(sys_train_deepSI, epochs=100, verbose=1, batch_size=1024, Loss_kwargs=dict(nf=30), sim_val=sys_test_deepSI[:5000])


# In[11]:


sys_deepSI.checkpoint_load_system()


# In[19]:


plt.plot(sys_deepSI.Loss_val[2:])
plt.plot(sys_uxyeye.Loss_val)
plt.legend(['deepSI','uxyeye'])
plt.xlabel('epoch')
plt.ylabel('Simulation loss test')
plt.show()


# In[35]:


sys_data_norm = sys_deepSI.norm.transform(sys_test_deepSI)
obs, k0 = sys_deepSI.init_state_multi(sys_data_norm, nf=len(sys_data_norm)-50)
_,_,ufuture,yfuture = sys_data_norm.to_hist_future_data(na=k0,nb=k0,nf=len(sys_data_norm)-50)
predict = []
real = []
for unow,ynow in zip(np.swapaxes(ufuture,0,1),np.swapaxes(yfuture,0,1)):
    predict.append(obs)
    real.append(ynow)
#     Losses.append(np.mean((ynow-obs)**2)**0.5)
    obs = sys_deepSI.step_multi(unow)
# return np.array(Losses)
predict = np.array(predict)[:,0]
real = np.array(real)[:,0]


# In[42]:


plt.plot(predict)
plt.plot(real)
plt.show()

np.mean((predict-real)**2)**0.5/np.std(real)


# In[52]:


Ypredict = []
Yreal = []
sys_data_norm = sys_deepSI.norm.transform(sys_test_deepSI)

U = sys_data_norm.u
obs, k0 = sys_deepSI.init_state(sys_data_norm) #is reset if init_state is not defined #normed obs
# Y.extend(sys_data_norm.y[:k0])


for action,yreal in zip(U[k0:],sys_data_norm.y[k0:]):
    Ypredict.append(obs)
    Yreal.append(yreal)
    obs = sys_deepSI.step(action)
Ypredict = np.array(Ypredict)
Yreal = np.array(Yreal)


# In[53]:


U[k0:],ufuture[0],yfuture[0],Yreal #u and y are the same


# In[54]:


Ypredict,predict #found offset present


# In[55]:


plt.plot(Ypredict)
plt.plot(Yreal)
plt.show()

np.mean((Ypredict-Yreal)**2)**0.5/np.std(Yreal)


# In[16]:


import numpy as np
sys_test_deepSI.plot()
np.mean((sys_test_deepSI_predict-sys_test_deepSI).y**2)**0.5/np.std(sys_test_deepSI.y)*100


# In[21]:


sys_test_deepSI_predict


# In[20]:


sys_test_deepSI


# In[19]:


12.884041966245285*sys_deepSI.norm.ystd


# In[12]:


sys_test_deepSI_predict = sys_deepSI.apply_experiment(sys_test_deepSI)


# In[23]:


sys_test_deepSI_predict.y


# In[24]:


sys_test_deepSI.y

