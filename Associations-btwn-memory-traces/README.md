
# Recurrent Spiking Network (Pokorny et al., 2020)

Minimal-but-complete Python implementation of the recurrent spiking network from:

> Pokorny et al., *Cerebral Cortex* (2020) — “STDP Forms Associations between Memory Traces in Networks of Spiking Neurons”.

This code emphasizes **correct model behavior** and a clean, modular layout that runs smoothly in Jupyter notebooks or as plain Python scripts.

---

## Contents

- [Folder Layout](#folder-layout)
- [Environment & Install](#environment--install)
- [Quick Start](#quick-start)
- [Configuration (read this first!)](#configuration-read-this-first)
- [Running the Simulator](#running-the-simulator)
- [Analysis Helpers](#analysis-helpers)
- [Progress Bars & Checkpoints](#progress-bars--checkpoints)
- [Tuning Cheat Sheet](#tuning-cheat-sheet)
- [Sanity & Debug Utilities](#sanity--debug-utilities)
- [Troubleshooting](#troubleshooting)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

---

## Folder Layout

```

.
├── config.py            # Central configuration (edit here for authoritative changes)
├── geometry.py          # 3D grid placement, pairwise distances
├── connectivity.py      # Distance-dependent Bernoulli connections & delays (Table 1)
├── init.py              # Weight & STP parameter sampling + steady-state correction
├── psp.py               # Double-exponential PSP kernel (peak-normalized, cut-off)
├── neurons.py           # Membrane potential, refractory sampling, stochastic spiking
├── stp.py               # UDF short-term plasticity state updates
├── stdp\_triplet.py      # Triplet STDP (eq. 19) with r/o traces & clipping caps
├── inputs.py            # Pattern/noise schedule & input Poisson spike generation
├── simulate.py          # Orchestrates phases, ring buffers, plasticity, checkpoints
├── analyze.py           # Assembly (PRU) & PCU detection utilities (paper criteria)
├── demo\_driver.py       # Minimal example driver
└── (optional) notebooks / tuning cells (your Jupyter workflow)

````

---

## Environment & Install

- **Python**: 3.9+ recommended  
- **Dependencies**: `numpy`, `scipy`, `matplotlib`, `tqdm`

Install:
```bash
pip install numpy scipy matplotlib tqdm
````

(If you use `requirements.txt`, install that instead.)

---

## Quick Start

Run from a shell:

```bash
python demo_driver.py
```

Or in Jupyter:

```python
import simulate          # prefer module import to avoid stale bindings
out = simulate.run(progress=True)
```

**Outputs** (in `out`, a dict):

* `E_spike_times`, `I_spike_times`: list-of-ndarray spike times (ms)
* `weights`: dict of final weight matrices (`W_inpE`, `W_EE`, `W_IE`, `W_II`)
* `snapshots`: optional weight snapshots at phase boundaries
* `schedule`: list of segments with kind/label/length (ms)
* `patterns`: dict with input channel indices per color (blue/green/red)
* `onsets`: pattern onset times per color (ms)

---

## Configuration (read this first!)

This project follows a **“hard-config” rule**: **edits to `config.py` are authoritative**.
If you change parameters at runtime in a notebook, **reload** or **restart the kernel** to ensure `simulate.py` sees them, or pass a `cfg` explicitly if your `simulate.run` supports it.

Open `config.py` and adjust:

* `FAST_DEMO`: `True` for short runs, `False` for full protocol
* `demo_scale`: scales all phase durations (e.g., `0.01` = 1% of full)
* `gamma_w`: scales **Inp→E** and **E→E** initial means (synaptic drive)
* `gamma_exc`: scales generic excitatory bias (sets baseline excitability)
* `scale_IE`: (optional) multiplier on **I→E** weights (default `1.0`)
* `scale_II`: (optional) multiplier on **I→I** weights (default `1.0`)
* `seeds`: RNG seeds
* `caps`: STDP clipping caps (Inp→E: \[0, 200%], E→E: \[0, 1000%] by default)
* `version_tag`: free string — bump this whenever you change config (helps sanity logs)

After editing `config.py`, **save** and in Jupyter do:

```python
from importlib import reload
import config, simulate
reload(config); reload(simulate)
out = simulate.run(progress=True)
```

---

## Running the Simulator

**Full protocol** (phases 1–5):

1. 250 s noise
2. 250 s encoding (random 100 ms patterns; 0.5–3 s noise gaps)
3. 250 s noise
4. \~36–37 s association with **combined** (blue+green) pattern, 20 reps
5. 250 s noise

**Fast demo**: same structure with durations scaled by `demo_scale`.

Minimal script:

```python
import simulate
out = simulate.run(progress=True)

# Quick post-run summary
T_sec = sum(seg["T_ms"] for seg in out["schedule"]) / 1000.0
E = sum(len(s) for s in out["E_spike_times"]) / (432 * T_sec)
I = sum(len(s) for s in out["I_spike_times"]) / (108 * T_sec)
print(f"Mean E: {E:.2f} Hz, Mean I: {I:.2f} Hz")
```

---

## Analysis Helpers

**Assemblies (PRUs)** per paper (baseline −100–0 ms, response 10–110 ms, Wilcoxon, median ≥ 2 Hz):

```python
from analyze import detect_PRUs
labels = ["blue","green","red"]

# Onsets during encoding
enc0 = out["schedule"][0]["T_ms"]
enc1 = next(i for i,s in enumerate(out["schedule"]) if s["kind"]=="combined")
enc1_ms = sum(s["T_ms"] for s in out["schedule"][:enc1])  # start of association

# Build onset dict in [enc0, enc1_ms)
def onsets_between(schedule, labels, t0, t1):
    acc=0; d={k:[] for k in labels}
    for seg in schedule:
        if seg["kind"]=="pattern" and seg["label"] in labels and (t0 <= acc < t1):
            d[seg["label"]].append(acc)
        acc += seg["T_ms"]
    return {k: np.array(v, int) for k,v in d.items()}

import numpy as np
enc_on = onsets_between(out["schedule"], labels, enc0, enc1_ms)
prus = detect_PRUs(out["E_spike_times"], enc_on, labels)
print({k: len(v) for k,v in prus["assemblies"].items()})
```

**PCUs** (cells switching preference after association):

```python
from analyze import detect_PCUs
pcus = detect_PCUs(out)   # uses stored schedule/spikes
print("#PCUs:", len(pcus))
```
---

## Sanity & Debug Utilities

**Config banner** (optional): at the start of `simulate.run()` you can print:

```python
print(f"[CFG] tag={CFG.version_tag} demo_scale={CFG.demo_scale} "
      f"gamma_w={CFG.gamma_w} gamma_exc={CFG.gamma_exc} "
      f"scale_IE={getattr(CFG,'scale_IE',1.0)} scale_II={getattr(CFG,'scale_II',1.0)}")
```

**Network fingerprint cell** (in a notebook; run after saving config & reloading):

```python
from importlib import reload; import numpy as np, config, simulate
reload(config); reload(simulate)
out = simulate.run(progress=True)

T_ms = sum(s["T_ms"] for s in out["schedule"])
print("Total schedule (ms):", T_ms)
for name, W in out["weights"].items():
    print(f"{name} mean = {np.mean(W):.3f}")

T_s = T_ms / 1000.0
E = sum(len(s) for s in out["E_spike_times"]) / (432 * T_s)
I = sum(len(s) for s in out["I_spike_times"]) / (108 * T_s)
print(f"Mean E: {E:.2f} Hz, Mean I: {I:.2f} Hz")
```

---

## Citation

If you use this repository, please cite:

* Pokorny, I., Maass, W., Triesch, J., & Klampfl, S. (2020). *STDP Forms Associations between Memory Traces in Networks of Spiking Neurons.* Cerebral Cortex, 30(9), 5785–5804.

---

