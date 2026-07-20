# Reproduction log

Independent cold-clone reproductions of `python qsleuth.py`. The README invites
these ("Reproduced it? Open a reproduction issue"). Failures are logged here with
equal prominence to successes.

## 2026-07-19 — Claude Fable 5 (Anthropic), sandbox

**Result: ALL 9 SELF-TESTS PASSED. Every README number reproduced exactly.**

- Commit: `5234b7a`  ·  `qsleuth.py` sha256 prefix `9ae80f31518adcb4`
- Environment: Python 3.12.3, NumPy 2.4.4, Linux x86_64 (glibc 2.39)
  — a different stack from the author's aarch64 / Python 3.14 phone;
  the numbers matched across both architectures.
- Method: `git clone` (depth 1) → `python3 qsleuth.py`, run twice, cold.
- Runtime: 46.7 s and 46.6 s (README claims ~30–60 s ✓).

| Test | README claim | Reproduced |
|---|---|---|
| S1 | Bloch/PTM exact, n=1 and n=2 | exact ✓ |
| S2 | K discovered = 5 (true 5) | K = 5 ✓ |
| S3 | recovery ‖learned−true‖_F ≈ 0.093 | 0.093 ✓ |
| S4 | attribution 1.000 (80 fresh) | 1.000 ✓ |
| S5 | atoms CPTP ≥ −0.001; random control −1.39 | −0.001 / −1.393 ✓ |
| S6 | entangling/local labels exact | exact ✓ |
| S7 | 90% conformal coverage | 0.933 ✓ |
| S8 | intruder TPR 1.00 / FPR 0.05 | 1.000 / 0.050 ✓ |
| S9 | audit chain + tamper detection | verifies; tamper detected ✓ |

Notes: all four `.py` files compile clean. No fixes were required to reach 9/9 —
this repository's README claims and its code agree, on first run, on foreign
hardware. Recorded as the baseline reproduction.
