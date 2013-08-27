import numpy as np
import matplotlib.pyplot as plt

pstc = .05   # post synaptic time constant
# dt = .00001   # simulation time step
T = 2      # total simulation time

plt.figure("state_timeseries", figsize=(16,13))
plt.figure("state_histogram_uniform", figsize=(16,13))
plt.figure("state_histogram_poisson", figsize=(16,13))

rates = np.logspace(0,4,9)
for idx, f in enumerate(rates):
    print f
    # f = 1.     # input firing rate
    dt = 1./(f*100.)
    Tf_uniform = 1./f   # firing rate period
    Tf_poisson = np.random.exponential(1./f) # interspike interval period
    elapsed_uniform = Tf_uniform # elapsed time since last spike
    elapsed_poisson = Tf_poisson # elapsed time since last spike
    
    steps = int(T/dt)
    decay = np.exp(-dt/pstc)
    
    t = 0.
    s_uniform = 0.
    s_poisson = 0.
    time = []
    state_poisson = []
    state_uniform = []
    
    spike_times_poisson = []
    spike_times_uniform = []
    for i in range(steps):
        if elapsed_poisson >= Tf_poisson:
            s_poisson = s_poisson * decay + 1./pstc
            elapsed_poisson = elapsed_poisson - Tf_poisson + dt
            Tf_poisson = np.random.exponential(1./f)
            spike_times_poisson.append(t)
        else:
            s_poisson = s_poisson * decay 
            elapsed_poisson += dt
        state_poisson.append(s_poisson)
    
        if elapsed_uniform >= Tf_uniform:
            s_uniform = s_uniform * decay + 1./pstc
            elapsed_uniform = elapsed_uniform - Tf_uniform + dt
            spike_times_uniform.append(t)
        else:
            s_uniform = s_uniform * decay 
            elapsed_uniform += dt
        state_uniform.append(s_uniform)
    
        time.append(t)
        t += dt
        
    time = np.array(time)
    state_poisson = np.array(state_poisson)
    state_uniform = np.array(state_uniform)

	# plot timeseries
    plt.figure("state_timeseries")
    plt.subplot(3,3,idx+1)
    plt.plot(time,state_poisson/f, 'r')
    plt.plot(time,state_uniform/f, 'b')
    plt.axvline(1*pstc, c='k', ls=':')
    plt.axvline(2*pstc, c='k', ls=':')
    plt.axvline(3*pstc, c='k', ls=':')
    plt.axvline(4*pstc, c='k', ls=':')
    plt.axvline(5*pstc, c='k', ls=':')
    plt.axhline((1-np.exp(-1)), c='k', ls=':')
    plt.axhline((1-np.exp(-2)), c='k', ls=':')
    plt.axhline((1-np.exp(-3)), c='k', ls=':')
    plt.axhline((1-np.exp(-4)), c='k', ls=':')
    plt.axhline((1-np.exp(-5)), c='k', ls=':')
    plt.title('Firing Rate %1.0fHz'%f)

	# plot histogram of states
    plt.figure("state_histogram_uniform")
    plt.subplot(3,3,idx+1)
    plt.hist(state_uniform[time>5*pstc] / f, bins=108, normed=True, fc='blue', ec='none')
    plt.title('Firing Rate %1.0fHz'%f)
    plt.figure("state_histogram_poisson")
    plt.subplot(3,3,idx+1)
    plt.hist(state_poisson[time>5*pstc] / f, bins=108, normed=True, fc='red', ec='none')
    plt.title('Firing Rate %1.0fHz'%f)

plt.figure("state_timeseries")
plt.subplot(338)
plt.xlabel('Time (s)', fontsize=16)
plt.subplot(334)
plt.ylabel('Estimated Firing Rate (Hz), Normalized', fontsize=16)
plt.tight_layout()
plt.savefig('synapse_sim', dpi=240)

plt.figure("state_histogram_uniform")
plt.subplot(338)
plt.xlabel('state / firing rate after 5 tau_syn', fontsize=16)
plt.tight_layout()
plt.savefig('synapse_sim_hist_uniform', dpi=240)

plt.figure("state_histogram_poisson")
plt.subplot(338)
plt.xlabel('state / firing rate after 5 tau_syn', fontsize=16)
plt.tight_layout()
plt.savefig('synapse_sim_hist_poisson', dpi=240)

# print np.mean(state[time>5])
# print 1./np.mean(np.diff(np.array(spike_times)))
