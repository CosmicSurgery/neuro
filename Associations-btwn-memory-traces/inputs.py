# inputs.py
"""
Input patterns and schedule (Sections 2.2–2.3; Fig. 2).
- 3 stationary patterns, 100 ms, 20 on-channels at 40 Hz, others 0 Hz; pairwise overlap ≤ 2.
- During pattern: superimpose 3 Hz Poisson on all 200 channels.
- Between patterns: 5 Hz Poisson on all channels.
- 5 phases with STDP active throughout; association uses 20 presentations of combined (blue+green).

We generate Poisson events segment-wise (piecewise-constant rate).
"""
import numpy as np
from config import CFG
rng = np.random.default_rng(CFG.seed+2)

def make_three_patterns():
    N = CFG.N_inp
    P = []
    used = set()
    for _ in range(3):
        while True:
            on = rng.choice(N, CFG.on_channels, replace=False)
            on_set = set(on.tolist())
            ok = True
            for prev in P:
                if len(on_set.intersection(prev)) > CFG.max_overlap:
                    ok = False; break
            if ok:
                P.append(on_set); break
    return dict(blue=np.array(sorted(P[0])), green=np.array(sorted(P[1])), red=np.array(sorted(P[2])))

def poisson_events(rate_hz, T_ms):
    # Draw N~Poisson(rate*T) then uniform place; for small rates per channel this is fine
    expected = rate_hz * (T_ms/1000.0)
    n = rng.poisson(expected)
    if n == 0: return np.array([], dtype=int)
    return np.sort(rng.integers(0, T_ms, size=n, endpoint=False))

def build_schedule(pattern_sets):
    """
    Returns a list of segments for the whole experiment:
    Each segment: dict(kind='noise'|'pattern'|'combined', label=..., T_ms=int, on_channels=np.array([...]) or None)
    """
    def dur_s(seconds): return int(round(seconds*1000))
    if CFG.FAST_DEMO:
        scale = CFG.demo_scale
    else:
        scale = 1.0

    phases = []
    phases += [("noise", None, dur_s(CFG.T_init_s*scale))]

    # Encoding: ~250 s with random pattern interleaved noise gaps 0.5–3 s
    T_enc = dur_s(CFG.T_encode_s*scale)
    # Generate sequence until >= T_enc (then trim as in paper)
    seq = []
    t = 0
    prev = None
    labels = ["blue", "green", "red"]
    while t < T_enc + 5000:  # slack
        # noise gap
        gap = rng.uniform(0.5, 3.0)
        gap_ms = dur_s(gap*scale if CFG.FAST_DEMO else gap)
        seq.append(("noise", None, gap_ms))
        t += gap_ms
        # choose next pattern
        if prev is None or rng.random() < 0.75:
            choices = [l for l in labels if l != prev]
            lab = rng.choice(choices)
        else:
            lab = prev
        seq.append(("pattern", lab, CFG.pattern_ms))
        t += CFG.pattern_ms
        prev = lab
    # trim to start/end with noise (paper rule)
    if seq and seq[0][0] == "pattern":
        seq = seq[1:]
    if seq and seq[-1][0] == "pattern":
        seq = seq[:-1]
    # accumulate until reaching T_enc-ish
    enc = []
    acc = 0
    for kind, lab, T_ms in seq:
        if acc + T_ms > T_enc and kind == "pattern": break
        enc.append((kind, lab, T_ms))
        acc += T_ms
    phases += enc

    # Rest1
    phases += [("noise", None, dur_s(CFG.T_rest1_s*scale))]

    # Association: exactly 20 combined patterns with the same gap rules
    assoc = []
    count = 0
    while count < 20:
        gap = rng.uniform(0.5, 3.0)
        gap_ms = dur_s(gap*scale if CFG.FAST_DEMO else gap)
        assoc.append(("noise", None, gap_ms))
        assoc.append(("combined", "blue+green", CFG.pattern_ms))
        count += 1
    phases += assoc

    # Rest2
    phases += [("noise", None, dur_s(CFG.T_rest2_s*scale))]

    # Attach on-channel arrays
    pat = pattern_sets
    def on_for(label):
        if label == "blue": return pat["blue"]
        if label == "green": return pat["green"]
        if label == "red": return pat["red"]
        if label == "blue+green":
            x = np.union1d(pat["blue"], pat["green"])
            return x  # overlap clipped later at 40 Hz (rate cap)
        return None

    out = []
    for kind, lab, T_ms in phases:
        out.append(dict(kind=kind, label=lab, T_ms=T_ms, on=on_for(lab)))
    return out

def generate_input_spikes(schedule):
    """
    Returns dict: keyed by absolute time (ms) -> np.array of input indices that spike at that ms.
    Pattern blocks add 3 Hz background on all channels; non-pattern 5 Hz on all channels.
    """
    N = CFG.N_inp
    t0 = 0
    spikes = {}
    for seg in schedule:
        T = seg["T_ms"]
        if seg["kind"] == "noise":
            # 5 Hz on all channels
            for ch in range(N):
                ts = poisson_events(CFG.bg_noise_hz, T)
                for dt in ts:
                    spikes.setdefault(t0+dt, []).append(ch)
        else:
            # base 0 Hz (off) or 40 Hz on on-channels
            on = seg["on"]
            on_set = set(on.tolist()) if on is not None else set()
            for ch in range(N):
                base = CFG.ron_hz if ch in on_set else CFG.roff_hz
                ts_main = poisson_events(base, T)
                ts_bg = poisson_events(CFG.bg_in_pattern_hz, T)
                for dt in np.concatenate([ts_main, ts_bg]):
                    spikes.setdefault(t0+dt, []).append(ch)
        t0 += T
    # Make arrays and sort
    for t in list(spikes.keys()):
        spikes[t] = np.array(spikes[t], dtype=int)
    return spikes
