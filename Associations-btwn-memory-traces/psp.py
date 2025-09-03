# psp.py
"""
Double-exponential PSP kernel with τ_r=2 ms, τ_f=20 ms, cut-off T_ε=100 ms, peak=1 (eq. 1).
We implement two leaky traces r(t), f(t) per *postsynaptic population* (excitatory-input and inhibitory-input streams):
    r_{t+1} = r_t * exp(-dt/τ_r) + s_in(t)
    f_{t+1} = f_t * exp(-dt/τ_f) + s_in(t)
and y(t) = K * (f_t - r_t).
This matches the impulse response e^{-t/τ_f} - e^{-t/τ_r} with unit peak after scaling K.
"""
import numpy as np
from config import CFG

def kernel_scale(tau_r, tau_f):
    # Peak time where d/dt (e^{-t/τf}-e^{-t/τr}) = 0  ⇒ t* = (τr τf / (τf-τr)) ln(τf/τr)
    tp = (tau_r*tau_f/(tau_f - tau_r)) * np.log(tau_f/tau_r)
    peak = np.exp(-tp/tau_f) - np.exp(-tp/tau_r)
    return 1.0/peak

K_SCALE = kernel_scale(CFG.tau_r_ms, CFG.tau_f_ms)

class PSPState:
    def __init__(self, n_post):
        self.r = np.zeros(n_post, dtype=np.float32)
        self.f = np.zeros(n_post, dtype=np.float32)
        self.dec_r = np.exp(-1.0/CFG.tau_r_ms)
        self.dec_f = np.exp(-1.0/CFG.tau_f_ms)
    def step(self, s_in):
        self.r = self.r * self.dec_r + s_in
        self.f = self.f * self.dec_f + s_in
        return K_SCALE * (self.f - self.r)  # normalized to peak 1
