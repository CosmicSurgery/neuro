# neurons.py
"""
Neuron membrane update, stochastic spiking, and refractory (Section 2.1; eqs. 2–3).
- u_i(t) = Σ_j w_inp y_j + Σ_k w_EE z_k − Σ_l w_IE h_l + E_exc,i + E_exc,generic  (eq. 3)
- r_i(t) = r0 * exp(0.25 * u_i(t))  (eq. 2)
- Refractory: Gamma(k=2) with mean 10 ms (E) / 3 ms (I)

Excitabilities:
E_exc,i ~ LogNormal(μ=2.64, σ=0.23) shifted by −600 (text below eq. 3).
E_exc,generic = γ_exc * 300; I_exc,generic = 450 (eq. 5 and paragraph below).
"""
import numpy as np
from config import CFG

rng = np.random.default_rng(CFG.seed+1)

def gamma_refrac(mean_ms, size):
    k = 2.0
    theta = mean_ms / k
    return np.maximum(1, np.rint(rng.gamma(k, theta, size=size)).astype(int))

class NeuronPop:
    def __init__(self, nE, nI):
        self.nE, self.nI = nE, nI
        # Individual excitatory excitabilities (log-normal, shifted)
        ln = rng.normal(CFG.exc_log_mu, CFG.exc_log_sigma, size=nE)
        self.E_exc_i = (np.exp(ln) + CFG.exc_log_shift).astype(np.float32)
        self.E_exc_generic = (CFG.gamma_exc * CFG.E_exc_generic)
        self.I_exc_generic = CFG.I_exc_generic

        # States
        self.uE = np.zeros(nE, dtype=np.float32)
        self.uI = np.zeros(nI, dtype=np.float32)
        self.refracE = np.zeros(nE, dtype=int)
        self.refracI = np.zeros(nI, dtype=int)

        # Refractory sampling helpers
        self.refrac_draw_E = lambda n: gamma_refrac(CFG.refrac_mean_E_ms, n)
        self.refrac_draw_I = lambda n: gamma_refrac(CFG.refrac_mean_I_ms, n)

    def rates_from_u(self, u):
        # compute in float64; clamp exponent to ~80 (safe for float32) and to -50 (to avoid underflow)
        a = CFG.rate_exp_coeff * u.astype(np.float64)
        a = np.clip(a, -50.0, 80.0)
        r = CFG.r0_hz * np.exp(a)  # float64
        return r  # keep as float64 here


    def step(self, y_inp_toE, zE_toE, hI_toE, zE_toI, hI_toI):
        # E membrane (eq. 3): sum of PSPs ± offsets
        self.uE = (y_inp_toE + zE_toE - hI_toE
                   + self.E_exc_i + self.E_exc_generic).astype(np.float32)
        # I membrane (analogous, without external inputs)
        self.uI = (zE_toI - hI_toI + self.I_exc_generic).astype(np.float32)

        # Spiking probability with dt=1 ms; safe at high rates
        pE = 1.0 - np.exp(-self.rates_from_u(self.uE) * 0.001)
        pI = 1.0 - np.exp(-self.rates_from_u(self.uI) * 0.001)
        
        spikesE = (rng.random(self.nE) < pE) & (self.refracE == 0)
        spikesI = (rng.random(self.nI) < pI) & (self.refracI == 0)


        # Set refractory timers for spikers; decrement others
        self.refracE[spikesE] = self.refrac_draw_E(spikesE.sum())
        self.refracI[spikesI] = self.refrac_draw_I(spikesI.sum())
        self.refracE[self.refracE > 0] -= 1
        self.refracI[self.refracI > 0] -= 1

        return spikesE.astype(np.uint8), spikesI.astype(np.uint8)
