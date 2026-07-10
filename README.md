# qolas

**Neural quantum circuit synthesis — architecture and self-test.**

QOLAS is a transformer that treats quantum circuit synthesis as a sequence
task: given a 4-qubit unitary (as a Pauli Transfer Matrix), it is designed to
output a gate sequence (H, X, Y, Z, S, T, CNOT, CZ, SWAP) approximating it.

## Status (honest)

- Architecture: real — GQA, RoPE, SwiGLU, RMSNorm. ~16M parameters.
- Self-test: passes 15/15 (Pauli basis, PTM, fidelity, model build, forward
  pass, synthesis loop, data generation). Run: `python3 qolas.py`
- Quantum math: verified — fidelity(X,X)=1.0, fidelity(X,I)=0.0.
- Training: NOT yet done. **An untrained model does not synthesize correct
  circuits** — it outputs near-random gates. Synthesis quality is unvalidated
  until the model is trained. This repo is the architecture and its self-test,
  not a trained synthesizer.

## Run

```bash
pip install torch numpy
python3 qolas.py     # runs the 15/15 self-test
**Step 4 — LICENSE:**
```bash
cat > LICENSE << 'ENDFILE'
All rights reserved. Copyright (c) 2026 Chad Edward Holland.
Reference implementation is public for review.
Contact for licensing or collaboration: c.holland.arch@proton.me
