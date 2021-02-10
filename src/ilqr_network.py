import jax
from jax import numpy as np
from jax.lax import scan
from jax import vmap, jit
from jax import jacfwd, jacrev
import numpy as onp

import time
import pickle



## Cost functions ## 

def cost_stage(x, u, target, lmbda):
    """
    \|state - target\|^2 + lmbda \|u\|^2
    x: (n_states,)
    u: (n_controls, )
    target: (n_states, ) -- To Do
    lmbda: float > 0, penalty on cost
    """
    state_cost = np.sum((x[:2] - target)**2)
    control_cost = np.sum(u**2)
    return state_cost + lmbda * control_cost

def cost_final(x, target):
    """
    \|state - target\|^2 
    """
    return np.sum((x[:2] - target)**2)
    
# Computes cost over trajectory of ln. (time steps, n_states)
cost_stage_trj = vmap(cost_stage, in_axes=(0,0,0,None))
# Cost of multiple trajectories: (batch size, time steps, n_states)
cost_stage_batch = vmap(cost_stage_trj, in_axes=(0,0,0,None))

def cost_trj(x_trj, u_trj, target_trj, lmbda):
    """
    \sum_t \|state - target\|^2 + lmbda \|u\|^2
    """
    c_stage = cost_stage_trj(x_trj[:-1], u_trj[:-1], target_trj[:-1], lmbda) 
    c_final = cost_final(x_trj[-1], target_trj[-1])
    return c_stage + c_final

cost_trj_batch = vmap(cost_trj, (0,0,0,None))


### Derivatives ### 

def cost_stage_grads(x, u, target, lmbda):
    """
    x: (n_states, )
    u: (n_controls,)
    target: (n_states, )
    lmbda: penalty on controls 
    """
    
    dL = jacrev(cost_stage, (0,1)) #l_x, l_u
    d2L = jacfwd(dL, (0,1)) # l_xx etc
    
    l_x, l_u = dL(x, u, target, lmbda)
    d2Ldx, d2Ldu = d2L(x, u, target, lmbda)
    l_xx, l_xu = d2Ldx
    l_ux, l_uu = d2Ldu
    
    return l_x, l_u, l_xx, l_ux, l_uu

# Accepts (batch size, n_states) etc.
cost_stage_grads_batch = vmap(cost_stage_grads, in_axes=(0,0,0,None)) 

def cost_final_grads(x, target):
    """
    x: (n_states, )
    target: (n_states, )
    """
    dL = jacrev(cost_final) #l_x, l_u
    d2L = jacfwd(dL) # l_xx etc
    
    l_x = dL(x, target)
    l_xx = d2L(x, target)
    
    return l_x, l_xx

cost_final_grad_batch = vmap(cost_final_grads, in_axes=(0,0))

# Dynamics
# Data
with open("../data/network_s972356.pickle", 'rb') as handle:
    data = pickle.load(handle)
params = data['params']
C = np.asarray(params['C'])
W = np.asarray(data['W'])
hbar = np.asarray(data['hbar'])

def continuous_dynamics(x, inputs):
    phi = lambda x: x #relu(x)
    tau = 150
    return (-x + W.dot(phi(x)) + inputs + hbar) / tau

def discrete_dynamics(x, inputs):
    # x: (neurons + 2, ), first two dims are readouts
    dt = 1.0
    y, h = x[:2], x[2:]
    h = h + dt*continuous_dynamics(h, inputs)
    y = h.dot(C)
    x_new = np.concatenate((y, h))
    return x_new, x_new

def rollout(h0, inputs):
    """
    Require:
        h0: network states (N, )
        inputs: (time steps, N)
    Return:
        y: readout (time steps, 2)
        h: rates (time steps, N)
    """
    x0 = np.concatenate((np.zeros((2, )), h0))
    _, x = scan(discrete_dynamics, x0, inputs)
    y, h = x[:,:2], x[:,2:]
    return y, h

rollout_jit = jit(rollout)
rollout_batch = jit(vmap(rollout, (0, 0)))

def dynamics_grads(x, u):
    """
    f: discrete dynamics x[t+1] = f(x[t], u[t])
    """
    def f(x,u):
        # Grab first output 
        return discrete_dynamics(x,u)[0]
    
    f_x, f_u = jacfwd(f, (0,1))(x,u)
    return f_x, f_u
  
dynamics_grads_batch = vmap(dynamics_grads, (None,0,0))


### Helpers for LQR approximation ### 

def Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
    """
    Assemble coefficients for quadratic approximation of value fn
    """
    Q_x = l_x.T + V_x.T @ f_x 
    Q_u = l_u.T + V_x.T @ f_u 
    Q_xx = l_xx + f_x.T @ V_xx @ f_x #
    Q_ux = l_ux + f_u.T @ V_xx @ f_x #
    Q_uu = l_uu + f_u.T @ V_xx @ f_u #
    return Q_x, Q_u, Q_xx, Q_ux, Q_uu


def gains(Q_uu, Q_u, Q_ux):
    """
    Feedback control law u* = k + Kx*
    """
    Q_uu_inv = np.linalg.inv(Q_uu)
    k = np.zeros(Q_u.shape) - Q_uu_inv @ Q_u
    K = np.zeros(Q_ux.shape) - Q_uu_inv @ Q_ux
    return k, K

def V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):
    """
    Quadratic approximation of value function
    """
    V_x  = Q_x.T + Q_u.T @ K + k @ Q_ux + k.T @ Q_uu @ K
    V_xx = Q_xx + K.T @ Q_ux + Q_ux.T @ K + K.T @ Q_uu @ K
    return V_x, V_xx

def expected_cost_reduction(Q_u, Q_uu, k):
    """
    Assuming approximations are true
    """
    return -Q_u.T.dot(k) - 0.5 * k.T.dot(Q_uu.dot(k))


# Forward pass
def discrete_dynamics_affine(xs, inputs):
    """
    Wrapper around arm dynamics fun that pre-computes
    control law
    """
    ut, xt_new = xs[:200], xs[200:]
    xt, ut, kt, Kt = inputs
    ut_new = ut + kt + Kt@(xt_new - xt)
    xt_new2 = discrete_dynamics(xt_new, ut_new)[0]
    res = np.concatenate((ut_new, xt_new2))
    return res, res 


def forward_pass_scan(x_trj, u_trj, k_trj, K_trj):
    """
    Simulate the system using control law around (x_trj, u_trj)
    defined by k_trj, K_trj
    """
    inputs = (x_trj, u_trj, k_trj, K_trj)
    init = np.concatenate((np.zeros_like(u_trj[0]), x_trj[0]))
    states  = scan(discrete_dynamics_affine, init, (x_trj, u_trj, k_trj, K_trj))[1]
    u_trj_new, x_trj_new = states[:,:200], states[:-1,200:]
    #print(x_trj[0].shape, x_trj_new.shape)
    x_trj_new = np.concatenate((x_trj[0][None], x_trj_new), axis=0)
    return u_trj_new, x_trj_new

forward_pass_jit = jit(forward_pass_scan)
# Batch over x, u, and feedback
forward_pass_batch = jit(vmap(forward_pass_scan, (0,0,0,0))) 


# Backward pass

def step_back_scan(state, inputs, regu, lmbda):
    """
    One step of Bellman iteration, backward in time
    """
    x_t, u_t, target_t = inputs
    k, K, V_x, V_xx = state
    l_x, l_u, l_xx, l_ux, l_uu = cost_stage_grads(x_t, u_t, target_t, lmbda)
    f_x, f_u = dynamics_grads(x_t, u_t)
    Q_x, Q_u, Q_xx, Q_ux, Q_uu = Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx)
    Q_uu_regu = Q_uu + np.eye(Q_uu.shape[0])*regu
    k, K = gains(Q_uu_regu, Q_u, Q_ux)
    V_x, V_xx = V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
    new_state = (k, K, V_x, V_xx)
    return new_state, new_state

def backward_pass_scan(x_trj, u_trj, target_trj, regu, lmbda):
    """
    Bellman iteration over entire trajectory
    """
    n_x, n_u = x_trj.shape[1], u_trj.shape[1]
    k, K = np.zeros((n_u, )), np.zeros((n_u, n_x))
    l_final_x, l_final_xx = cost_final_grads(x_trj[-1], target_trj[-1])
    V_x = l_final_x
    V_xx = l_final_xx
    # Wrap initial state and inputs for use in scan
    init = (k, K, V_x, V_xx)
    xs = (x_trj, u_trj, target_trj)
    # Loop --- backward in time
    step_fn = lambda state, inputs: step_back_scan(state, inputs, regu, lmbda)
    _, state = scan(step_fn, init, xs, reverse=True)
    k_trj, K_trj, _, _ = state
    return k_trj, K_trj

backward_pass_jit = jit(backward_pass_scan)



def run_ilqr(x0, target_trj, u_trj = None, max_iter=10, regu_init=10, lmbda=1e-1):
    # Main loop
    # First forward rollout
    if u_trj is None:
        N = target_trj.shape[0]
        n_u = 200
        u_trj = onp.random.normal(size=(N, n_u)) * 0.0001
    y_trj, h_trj  = rollout(x0, u_trj)
    x_trj = np.concatenate((y_trj, h_trj),1)
    total_cost = cost_trj(x_trj, u_trj, target_trj, lmbda).sum()
    regu = regu_init
    
    cost_trace = [total_cost]
    
    # Run main loop
    for it in range(max_iter):
        # Backward and forward pass
        k_trj, K_trj = backward_pass_jit(x_trj, u_trj, target_trj, regu, lmbda)
        u_trj_new, x_trj_new = forward_pass_jit(x_trj, u_trj, k_trj, K_trj)
        # Evaluate new trajectory
        total_cost = cost_trj(x_trj_new, u_trj_new, target_trj, lmbda).sum()
        t1 = time.time()
        
        # 
        cost_redu = cost_trace[-1] - total_cost
        cost_trace.append(total_cost)
        
        #if it%1 == 0:
        #    print(it, total_cost, cost_redu)
    return x_trj_new, u_trj_new, np.array(cost_trace)

# To do: use scan and jit?. At least vmap the backward passes etc.
run_ilqr_batch = vmap(run_ilqr, (0, 0, 0, None, None, None))
