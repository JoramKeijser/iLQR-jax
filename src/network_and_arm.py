import jax
from jax import numpy as np
from jax.lax import scan
from jax import vmap, jit

import pickle

import arm_model

# Network dynamics
with open("../data/network_s972356.pickle", 'rb') as handle:
    data = pickle.load(handle)
params = data['params']
C = np.asarray(params['C'])
W = np.asarray(data['W'])
hbar = np.asarray(data['hbar'])
phi = lambda x: x 

def continuous_network_dynamics(x, inputs):
    tau = 150
    return (-x + W.dot(phi(x)) + inputs + hbar) / tau

def discrete_network_dynamics(x, inputs):
    # x: (neurons + 2, ), first two dims are readouts
    dt = 1.0
    y, h = x[:2], x[2:]
    h = h + dt*continuous_network_dynamics(h, inputs)
    y = h.dot(C)
    x_new = np.concatenate((y, h))
    return x_new, x_new


# Combine them
def discrete_dynamics(x, inputs):
    """
    x: [y, h, q] of size 2+N+4 = N+6
    inputs: size (N, )
    """
    N = inputs.shape[0]
    network_states = discrete_network_dynamics(x[:N+2], inputs)[0]
    y, h = network_states[:2], network_states[2:]
    arm_states = arm_model.discrete_dynamics(x[N+2:], y)[0]
    x_new = np.concatenate((network_states, arm_states))
    return x_new, x_new 

def rollout(x0, u_trj):
    """
    x0: init states [y0, h0, q0], size (N+6, )
    u_trj: network inputs, size (N, )
    """
    N = u_trj.shape[1]
    _, x_trj = scan(discrete_dynamics, x0, u_trj)
    y, h, q = x_trj[:,:2], x_trj[:,2:N+2], x_trj[:,N+2:]
    return y, h, q

rollout_jit = jit(rollout)
rollout_batch = jit(vmap(rollout), (0,0))