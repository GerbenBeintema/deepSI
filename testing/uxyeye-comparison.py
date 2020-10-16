import uxyeye

train,test = uxyeye.data_sets.WienerHammerBenchMark()

sys = uxyeye.fit_systems.statespace_encoder_system_base(nx=8,na=20,nb=20)
