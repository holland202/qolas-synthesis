#!/usr/bin/env python3
"""QOLAS-REPAIR-001 feasibility counts (C-FEAS), run before registration.

Uses the ORIGINAL code (qolas_fd36e73.py, a verbatim copy of qolas.py at fd36e73). Prints:
  1. the gate vocabulary, by kind;
  2. |Tr(G^H G')|/16 over all ordered pairs of distinct gates (the fidelity a single-gate
     substitution produces, whatever the rest of the sequence is);
  3. how often self-test check [4/8] "random" fails with the original sampler,
     QR(randn + i randn)[0], over 20000 seeded draws.
Run from the repository root: python3 experiments/qolas_repair_001/feasibility.py
"""
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qolas_fd36e73 as q  # noqa: E402

g = q.GateSet(4)
kinds = collections.Counter(name.split("[")[0] for tid, name in g.token_to_name.items() if tid in g.token_to_gate)
print(f"1. vocab_size {g.vocab_size}: specials 4 (PAD BOS EOS UNK), gates {len(g.token_to_gate)} = "
      + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))

toks = sorted(g.token_to_gate)
vals = []
for a in toks:
    for b in toks:
        if a != b:
            vals.append(g.pauli.compute_fidelity(g.token_to_gate[a], g.token_to_gate[b]))
vals = np.array(vals)
print(f"2. distinct ordered pairs {len(vals)}: |Tr(G^H G')|/16 max {vals.max():.6f}, min {vals.min():.6f}, "
      f"pairs >= 0.995: {(vals >= 0.995).sum()}, distinct values {len(np.unique(vals.round(9)))}")

rng = np.random.default_rng(20261029)
N = 20000
f = np.empty(N)
for i in range(N):
    Q = np.linalg.qr(rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16)))[0]
    f[i] = abs(np.trace(Q)) / 16
low, high = int((f <= 0.05).sum()), int((f >= 0.45).sum())
print(f"3. original check [4/8] bound 0.05 < f < 0.45, original sampler, N={N}, seed 20261029: "
      f"fails {low + high} (low {low}, high {high}), rate {(low + high) / N:.4f}; "
      f"mean |Tr Q|^2 {np.mean((16 * f) ** 2):.4f} (Haar: 1)")
