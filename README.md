# qolas

**Neural quantum circuit synthesis — architecture and self-test.**

QOLAS is a transformer that treats quantum circuit synthesis as a sequence
task: given a 4-qubit unitary (as a Pauli Transfer Matrix), it is designed to
output a gate sequence (H, X, Y, Z, S, T, CNOT, CZ, SWAP) approximating it.

## Status (honest)

- Architecture: real — GQA, RoPE, SwiGLU, RMSNorm. ~1.1M parameters (4.0 MB) as configured — the mobile config the self-test builds. (An earlier README said ~16M; the shipped config is the smaller one.)
- Self-test: passes 21/21 (Pauli basis, PTM, fidelity, model build, forward
  pass, synthesis, fidelity gate, data generation). Run: `python3 qolas.py`.
  It is seeded, so two runs print the same, and its exit status is 0 only if
  every check passes. `python3 qolas.py --sabotage eos|greedy|fidelity|gate`
  re-introduces one defect so you can watch the matching check fail.
- Quantum math: tested — fidelity(U,U)=1, fidelity(X,I)=0, and phase,
  symmetry and unitary-invariance properties on 50 seeded Haar-random unitaries.
- Repaired 2026-10-29 (QOLAS-REPAIR-001, self-tested). The earlier version
  trained without EOS targets, sampled when asked for greedy decoding, had a
  fidelity check that failed about 1 run in 72 at random, had a synthesis check
  that could not fail, and exited 0 even when a check failed. Registration,
  harness and results: [docs/QOLAS_REPAIR_001_RESULTS.md](docs/QOLAS_REPAIR_001_RESULTS.md).
- Training: NOT yet done. An untrained model does not synthesize correct
  circuits — it outputs near-random gates. Synthesis quality is unvalidated
  until the model is trained. This repo is the architecture and its self-test,
  not a trained synthesizer.

## Run

    pip install torch numpy
    python3 qolas.py

Author: Chad Edward Holland, 2026. Runs on-device (Termux, PyTorch, numpy).

The 2026-10-29 repair: AI participation by Claude (Anthropic, Opus 5.5), which
registered, wrote and ran it at Chad Holland's direction → human validation:
direction only, not reviewed line by line → human editing/curation: none →
responsibility: Chad Holland. Not yet run on the S25.

