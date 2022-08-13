import numpy as np
import matplotlib.pyplot as plt

from kf import KF

plt.ion()
plt.figure()

real_x = 50.0
meas_variance = 0.001 ** 2
real_v = 0.0
meas_value = 0
old_meas_value = 0
diffValue = 0


kf = KF(initial_x=50.0, initial_v=0.0, accel_variance=0.1)

DT =0.02
NUM_STEPS = 5000
MEAS_EVERY_STEPS = 1



mus = []
covs = []
real_xs = []
real_vs = []
real_meas_value=[]
real_time = []


for step in range(NUM_STEPS):
    if step <750:
        real_v = 0.0
    else:
        real_v = 0.5

    if step >2500:
        real_v = 0.0
          
    covs.append(kf.cov)
    mus.append(kf.mean)

    real_x = real_x + DT * real_v
 
 

    kf.predict(dt=DT)
    meas_value=real_x + np.random.randn() * np.sqrt(meas_variance)
    if step == 1100:
        meas_value =100
        
    diffValue = abs(meas_value -old_meas_value)

    if  diffValue < 10:
       if step != 0 and step % MEAS_EVERY_STEPS == 0:
             kf.update(meas_value, meas_variance=meas_variance*1)
       
    real_xs.append(real_x)
    real_vs.append(real_v)
    real_meas_value.append(meas_value)
    real_time.append(DT*step)
    old_meas_value = meas_value



plt.rcParams['figure.figsize']= (4,4)
plt.rcParams['figure.dpi'] =500
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.subplots_adjust(hspace=0.5)
ax1.set_title("Kalman ROCIF masurement")
ax1.set_xlabel('time')
ax1.plot(real_time,real_meas_value, linewidth=0.5)

ax1.set_xlim(0, NUM_STEPS/50)
#ax1.set_ylim(49.9,50.1)
#ax1.set_ylim(49,70)
ax1.plot(real_time,[mu[0] - 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus,covs)], 'r:', linewidth=0.5)
ax1.plot(real_time,[mu[0] + 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus,covs)], 'r:', linewidth=0.5)
ax1.plot(real_time,[mu[0] for mu in mus], 'k', linewidth=0.6)


ax1.grid(color='b', alpha = 0.3 , linestyle='-.', linewidth=0.5)

#ax2.set_ylim(-0.5,1)
ax2.set_xlim(0, NUM_STEPS/50)
#ax2.set_xlim(14, 17)
ax2.plot(real_time,real_vs,'b:', linewidth=0.5)
ax2.grid(color='b', alpha = 0.3 , linestyle='-.', linewidth=0.5)

ax2.plot(real_time,[mu[1] - 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus,covs)], 'r:', linewidth=0.5)
ax2.plot(real_time,[mu[1] + 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus,covs)], 'r:', linewidth=0.5)
ax2.plot(real_time,[mu[1] for mu in mus], 'k', linewidth=0.6)
ax2.set_xlabel('ROCOF')

#plt.ginput(1)


