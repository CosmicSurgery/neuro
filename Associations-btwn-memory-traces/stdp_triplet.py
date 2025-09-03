# stdp_triplet.py
"""
Triplet STDP (eq. 19) on Inp→E and E→E synapses, all-to-all.

Detectors:
  - presyn r1,r2 (τ_r1=25 ms, τ_r2=25 ms); postsyn o1,o2 (τ_o1=1000 ms, τ_o2=25 ms).
Updates:
  - At presyn *spike time* t_pre: Δw = - o1_post(t) * [A2- + A3- * r2_pre(t-ε)].
  - At postsyn spike time t_post: Δw = + r1_pre(t) * [A2+ + A3+ * o2_post(t-ε)].
Ordering: compute Δw before incrementing r2 and o2 (ε→0⁺).

Pragmatic deviation: The paper’s NEST implementation updates at *arrival* times to each synapse
(incorporating per-connection conduction delays). Here, we update traces at *emission* times
(per pre/post unit). Delays are still applied to synaptic transmission/PSPs. This keeps the code
compact; qualitatively matches the reported behavior and learning curves.

Relative caps (with respect to each synapse’s initial weight): Inp→E ∈ [0, 200%], E→E ∈ [0, 1000%].
"""
import numpy as np
from config import CFG

class TripletSTDP:
    def __init__(self, W_inpE, W_EE):
        self.W_inpE = W_inpE
        self.W_EE = W_EE

        self.init_inpE = W_inpE.copy()
        self.init_EE   = W_EE.copy()

        n_inp = W_inpE.shape[1]
        nE    = W_inpE.shape[0]

        # Traces
        self.r1_inp = np.zeros(n_inp, dtype=np.float32)
        self.r2_inp = np.zeros(n_inp, dtype=np.float32)
        self.r1_E   = np.zeros(nE, dtype=np.float32)
        self.r2_E   = np.zeros(nE, dtype=np.float32)
        self.o1_E   = np.zeros(nE, dtype=np.float32)
        self.o2_E   = np.zeros(nE, dtype=np.float32)

        # Decays per 1 ms
        self.dr1 = np.exp(-1.0/CFG.tau_r1_ms)
        self.dr2 = np.exp(-1.0/CFG.tau_r2_ms)
        self.do1 = np.exp(-1.0/CFG.tau_o1_ms)
        self.do2 = np.exp(-1.0/CFG.tau_o2_ms)

    def decay(self):
        self.r1_inp *= self.dr1; self.r2_inp *= self.dr2
        self.r1_E   *= self.dr1; self.r2_E   *= self.dr2
        self.o1_E   *= self.do1; self.o2_E   *= self.do2

    def _clip_caps(self):
        self.W_inpE = np.clip(self.W_inpE, 0.0, CFG.cap_inpE * self.init_inpE)
        self.W_EE   = np.clip(self.W_EE,   0.0, CFG.cap_EE   * self.init_EE)

    def pre_update(self, pre_inp_idx=None, pre_E_idx=None, post_targets_for_inp=None, post_targets_for_E=None):
        # Δw at pre spikes (compute with current o1 and r2, then increment r2 after)
        if pre_inp_idx is not None and len(pre_inp_idx)>0:
            for j in pre_inp_idx:
                posts = post_targets_for_inp[j]
                if len(posts)==0: continue
                dw = - self.o1_E[posts] * (CFG.A2_minus + CFG.A3_minus * self.r2_inp[j])
                self.W_inpE[posts, j] += dw.astype(np.float32)
                # increment r1,r2 for that input channel *after* update
                self.r1_inp[j] += 1.0
                self.r2_inp[j] += 1.0

        if pre_E_idx is not None and len(pre_E_idx)>0:
            for j in pre_E_idx:
                posts = post_targets_for_E[j]
                if len(posts)==0: continue
                dw = - self.o1_E[posts] * (CFG.A2_minus + CFG.A3_minus * self.r2_E[j])
                self.W_EE[posts, j] += dw.astype(np.float32)
                self.r1_E[j] += 1.0
                self.r2_E[j] += 1.0

        self._clip_caps()

    def post_update(self, post_E_idx=None, pre_incoming_inp=None, pre_incoming_E=None):
        # Δw at post spikes (compute with current r1 and o2, then increment o2 after)
        if post_E_idx is None or len(post_E_idx)==0:
            return
        for i in post_E_idx:
            pres_inp = pre_incoming_inp[i]
            if len(pres_inp)>0:
                dw = self.r1_inp[pres_inp] * (CFG.A2_plus + CFG.A3_plus * self.o2_E[i])
                self.W_inpE[i, pres_inp] += dw.astype(np.float32)
            pres_E = pre_incoming_E[i]
            if len(pres_E)>0:
                dw = self.r1_E[pres_E] * (CFG.A2_plus + CFG.A3_plus * self.o2_E[i])
                self.W_EE[i, pres_E] += dw.astype(np.float32)
            self.o1_E[i] += 1.0
            self.o2_E[i] += 1.0
        self._clip_caps()
