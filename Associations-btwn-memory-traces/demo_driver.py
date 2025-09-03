# demo_driver.py
"""
Minimal driver:
1) Build & run default network (FAST_DEMO on by default for quick end-to-end).
2) Analyze assemblies pre/post association.
3) Print concise summaries: assembly sizes, mean weight changes, #PCUs.

To run the full-length protocol, set CFG.FAST_DEMO=False in config.py and re-run.
"""
import numpy as np
from simulate import run
from analyze import detect_PRUs, detect_PCUs

if __name__ == "__main__":
    out = run()

    # Extract trial onsets before and after association phase for analysis
    # Approximating: use first half of schedule for "before", second half for "after".
    # (In the paper, dedicated analysis runs are made with STDP disabled; here we
    # reuse the recorded spiking as a compact proxy.)
    schedule = out["schedule"]
    total_T = sum(seg["T_ms"] for seg in schedule)
    cut = total_T//2
    labels = ["blue","green","red"]

    # Split onsets by time
    onsets_before = {lab: [] for lab in labels}
    onsets_after  = {lab: [] for lab in labels}
    t_acc = 0
    for seg in schedule:
        if seg["kind"] in ("pattern",):
            if seg["label"] in labels:
                (onsets_before if t_acc<cut else onsets_after)[seg["label"]].append(t_acc)
        t_acc += seg["T_ms"]
    # Convert to arrays
    for d in (onsets_before, onsets_after):
        for k in d: d[k] = np.array(d[k], dtype=int)

    # PRUs/assemblies using “before"
    E_spikes = out["E_spike_times"]
    res_before = detect_PRUs(E_spikes, onsets_before, labels)
    assemblies = res_before["assemblies"]
    sizes = {k: len(v) for k,v in assemblies.items()}

    # PCUs using before vs after
    n_pcus, pcus = detect_PCUs(E_spikes, E_spikes, onsets_before, onsets_after, assemblies)

    # Weight summaries
    W_inpE, W_EE = out["weights"]["W_inpE"], out["weights"]["W_EE"]
    W_inpE0, W_EE0 = out["init_weights"]["W_inpE"], out["init_weights"]["W_EE"]
    dW_inpE = (W_inpE - W_inpE0)
    dW_EE   = (W_EE - W_EE0)

    print("Assemblies (before):", sizes)
    print("Mean Δw Inp→E:", float(dW_inpE.mean()))
    print("Mean Δw E→E  :", float(dW_EE.mean()))
    print("#PCUs:", n_pcus)
