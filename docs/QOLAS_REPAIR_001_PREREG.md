# QOLAS-REPAIR-001: five defects in qolas.py at fd36e73, and a repair that has to prove them fixed

**State:** REGISTERED, NOT RUN. Nothing is built. This file is committed alone, with the feasibility counts and a
verbatim copy of the original code, before the repair or the harness exists. It is not edited after this commit;
errors found later go in the results.
**Authors:** Direction: Chad Holland (2026-10-29; choice of path delegated). Design, feasibility script and this
text: Claude (Anthropic, Opus 5.5). Human review level: direction only. Everything run under this registration is
**self-tested**.
**Method:** SWAY: [METHOD.md](https://github.com/holland202/principia-artificialis/blob/78726b8/METHOD.md) (md5
`50e8a91a448251f8fc5d70f6971b74ae`), WORKFLOW.md (md5 `8ab6970501f53cb288d35f19787be529`) and CONTROLS.md (md5
`2a68368e99f87430bd15e8074811452c`), as on principia-artificialis `main` at `78726b8`.
**Subject:** `qolas.py` at `fd36e73` (sha256 `de76c246013b9c1d6d7b24476cdac3b2ffda43cd9c3b3bf3ddd4570415dc8b12`),
copied verbatim to [`experiments/qolas_repair_001/qolas_fd36e73.py`](../experiments/qolas_repair_001/qolas_fd36e73.py).

The model stays **NOT TRAINED**. Nothing here measures synthesis quality.

## Question

Do the five defects below exist in `qolas.py` at `fd36e73`, and does a repair remove each one, with checks that can
be shown failing, without breaking what the self-test already checked?

**What the code can express.** Read before registering (VERIFIED FROM SOURCE, not run):

| # | Defect | Where |
|---|---|---|
| D1 | Training targets never contain EOS. `generate_batch` builds `inp=[BOS]+t[:-1]`, `tgt=t`. A model trained on this data is never shown when to stop, and `generate` stops only on EOS or after 50 steps | `SyntheticDataGenerator.generate_batch` |
| D2 | "greedy" is not greedy. `synthesize(..., "greedy")` sets `temp=1.0`, and `generate` always samples with `torch.multinomial`. Greedy synthesis is sampling at temperature 1, so it is not repeatable | `SynthesisEngine.synthesize`, `QOLASModel.generate` |
| D3 | Self-test check [4/8] "random" is unseeded and its bound is a sampler artifact. It asserts `0.05 < |Tr(Ur)|/16 < 0.45` for `Ur = qr(randn + i·randn)[0]`. Without fixing the phases of R's diagonal, that Q is not Haar-distributed; the bound fits that biased sampler and fails at random | `test()`, check [4/8] |
| D4 | Self-test check [7/8] cannot fail. It checks only that the result has a `gate_sequence` key and that `fidelity` is a float, so any synthesizer passes, including one that returns nothing | `test()`, check [7/8] |
| D5 | The script's exit status ignores the result. `if __name__=="__main__":test()` discards `test()`'s return value, so the script exits 0 even when a check prints FAIL. CI or a shell can never see a failure | last line |

## Design (C-DESIGN)

- **Design:** audit plus a software experiment on one codebase at one commit, in one environment.
- **Strongest claims it can carry:** existence (each defect is present at `fd36e73`, shown by running the original
  code) and exact counts for the repaired code on these seeded inputs.
- **Out of scope:** synthesis quality, training, behaviour on the S25, and any other version of QOLAS.

## Feasibility counts (C-FEAS)

From [`feasibility.py`](../experiments/qolas_repair_001/feasibility.py), run on the original code before this
registration; output in [`feasibility_output.txt`](../experiments/qolas_repair_001/feasibility_output.txt):

1. `vocab_size` 60: 4 specials (PAD, BOS, EOS, UNK) and 56 gates (H, X, Y, Z, S, T, Sdg, Tdg on each of 4 qubits;
   CNOT 12; CZ 6; SWAP 6).
2. Over the 3,080 ordered pairs of distinct gates, `|Tr(G^H G')|/16` is at most **0.923880** and at least 0; no pair
   reaches 0.995. A sequence that differs from the target in one gate has fidelity `|Tr(G^H G')|/16`, whatever the
   other gates are (`|Tr((A G B)^H (A G' B))| = |Tr(G^H G')|`). So a one-gate substitution always fails the 0.995
   threshold.
3. With the original sampler and seed 20261029, check [4/8]'s bound fails **276 of 20,000** draws (rate 0.0138,
   about 1 run in 72), all on the low side. The mean of `|Tr Q|^2` is 4.0507; for a Haar unitary it is 1.

## The repair (to be built after this commit)

`qolas.py` changes:

- **D1:** `inp = [BOS] + t`, `tgt = t + [EOS]`, then PAD. Generators take an optional seeded `numpy` Generator.
- **D2:** `generate` takes the argmax when `temp <= 0`; `synthesize(..., "greedy")` passes `temp = 0`.
- **D3:** check 4 keeps `exact=1`. It adds `fidelity(X[0], I) = 0`. It replaces the bound with properties that hold
  for every unitary, tested on 50 seeded Haar unitaries (phase-corrected QR): `f(U,U) = 1`, `f(U, e^{iφ}U) = 1`,
  `f(U,V) = f(V,U)`, `f(WU, WV) = f(U,V)`, `0 ≤ f ≤ 1`.
- **D4:** a fidelity gate (threshold `fidelity_threshold` = 0.995, unchanged). The self-test checks that the true
  sequence passes, that a one-gate substitution fails, and that the untrained model fails (a negative control).
- **D5:** `sys.exit(0 if test() else 1)`.
- `test()` seeds NumPy and PyTorch. `python3 qolas.py --sabotage MODE` re-introduces one defect (`eos`, `greedy`,
  `fidelity`, `gate`) so the self-test can be shown failing.

The repaired self-test has **21 checks** in nine sections. It keeps 14 of the 15 original checks; only check 4's
"random" goes. It adds seven:

- in [4/9] Fidelity: `X vs I` and `properties`;
- in [7/9] Synthesis: `greedy=argmax`;
- [8/9] Fidelity gate, which is new: `oracle passes`, `substitution fails` and `untrained model fails`;
- in [9/9] DataGen (formerly [8/8]): `EOS`.

## Instrument

A harness, `experiments/qolas_repair_001/run_qolas_repair_001.py`, imports both the original copy and the repaired
`qolas.py` (M4: the real classes, nothing copied). It prints one HELD / REFUTED / UNRESOLVED line per prediction, a
`VERDICT n of 17 as registered` line and a `DIGEST` (sha256 of the canonical, rounded results). It then pins
`RECORDED` in a separate commit. Its `--sabotage` switch runs every "repaired" arm on the original code instead;
the verdict must then differ and the harness must exit 1.

**Seeds:** NumPy `default_rng(1)` for training examples, `default_rng(2)` for the 5 synthesis targets,
`default_rng(3)` for the 20 gate targets, `default_rng(4)` for substitutions, `default_rng(20261030)` for Haar
unitaries; `torch.manual_seed(0)` before building the model, and `torch.manual_seed(s)`, s = 0..9, before each
repeated synthesis. The original code draws from the legacy global `np.random`; for its arms the harness calls
`np.random.seed(1)` and `np.random.seed(2)` instead.

## Gates (instrument gates, not predictions)

Any gate failing → exit 2, no verdict.

- **G1:** the original copy's sha256 is `de76c246…8b12`.
- **G2:** the repaired `qolas.py` has a different sha256 (printed).
- **G3:** the Haar sampler's mean `|Tr U|^2` over the 20,000 draws is in [0.9, 1.1].
- **G4:** none of the 20 gate targets has `fidelity(U, I) ≥ 0.995` (no degenerate target).
- **G5:** the harness run with `--sabotage` exits 1 with a different verdict line.

## Predictions (registered before observing)

Arms marked **(rival)** run the original code. Each is the simplest rival to the claim that the defect exists: if
the rival holds, the defect is not real.

| ID | Prediction | Precisely wrong if |
|---|---|---|
| P1a (rival) | Original generator, 200 seeded examples: **0** targets contain EOS | any target contains EOS |
| P1b | Repaired generator, 200 seeded examples: **200** have `inp = [BOS] + t + PAD…` and `tgt = t + [EOS] + PAD…`, with 3 ≤ len(t) ≤ 10 and every token of t a gate token | any example deviates |
| P2a (rival) | Original "greedy", 5 seeded targets × 10 torch seeds: **≥ 9 distinct** token sequences for **every** target | any target gives fewer than 9 |
| P2b | Repaired greedy, same targets and seeds: **exactly 1** distinct sequence per target. Every emitted token equals the argmax of the logits recomputed for its prefix | any target gives 2 or more, or any token is not the argmax. UNRESOLVED (not HELD) if fewer than 10 steps are checked in total |
| P2c | Repaired sampling (temperature 0.8), same targets and seeds: **≥ 9 distinct** per target (sampling still samples) | any target gives fewer than 9 |
| P3a | *Retrodiction, not counted:* the original bound fails 276 of 20,000 draws with the original sampler (feasibility count 3) | — |
| P3b | Haar sampler, seed 20261030, 20,000 draws: the original bound fails at a rate in **[0.45, 0.50]**. The expected rate is 1 − e^(−0.64) = 0.4727, since `|Tr U|^2` is close to Exp(1) for Haar U(16) | the rate is outside [0.45, 0.50] |
| P3c | The repaired fidelity properties hold on **200 of 200** seeded Haar unitaries (tolerance 1e-9), and `fidelity(P[q], I) = 0` for P ∈ {X, Y, Z} on all 4 qubits (**12 of 12**, tolerance 1e-12) | any failure |
| P3d | Three fidelity mutants are each caught by the repaired check 4: (a) no complex conjugate, (b) divide by d² instead of d, (c) real part instead of absolute value. **3 of 3 killed** | any mutant passes |
| P4a (rival) | Original check [7/8] reports OK on both its items when the synthesizer is replaced by one returning (i) no gates and (ii) 50 random gate tokens: **4 of 4** items OK | any item FAILs |
| P4b | Fidelity gate, 20 seeded targets of 3–10 gates: the true sequence passes **20 of 20** (fidelity ≥ 1 − 1e-9) | any fails |
| P4c | One-gate substitution (random position, random different gate): fidelity equals `|Tr(G^H G')|/16` within 1e-9 in **20 of 20**, and the gate fails **20 of 20** | any mismatch or pass |
| P4d | The untrained seeded model, greedy, passes the gate on **at most 1** of the 20 targets | 2 or more pass |
| P5a (rival) | The original file run as a script with `numpy.allclose` forced to False prints `FAIL identity` and **exits 0**, with no traceback | it exits non-zero, or no FAIL is printed. UNRESOLVED if a traceback appears |
| P5b | The repaired file under the same patch **exits 1** and prints `FAIL identity` | it exits 0 |
| P6a | The repaired self-test prints `RESULT: 21/21 PASS` and exits 0 | anything else |
| P6b | Two runs of the repaired self-test are byte-identical once the numbers in the DataGen timing label are masked | any other difference |
| P6c | Each sabotage mode (`eos`, `greedy`, `fidelity`, `gate`) makes the repaired self-test print at least one FAIL and exit 1: **4 of 4** | any mode exits 0 |

## Pre-mortem

The likeliest ways this instrument records nothing, or the wrong thing:

1. **Both arms load the same file**, so "original" and "repaired" agree. This would show as the rival arms and the
   repaired arms holding or failing together. G1 and G2 print both hashes.
2. **Greedy determinism holds vacuously**: if the untrained model's argmax is EOS at step 0 for every target, every
   output is empty and trivially identical. P2b counts the steps it checked and goes UNRESOLVED under 10.
3. **The Haar sampler is itself wrong** (phases not fixed). P3b would then read near 0.014 instead of 0.47. G3
   checks `E|Tr U|^2 ≈ 1` first.
4. **The P5 patch raises instead of failing a check.** The exit code would then come from a crash, not from the
   exit-status logic. P5a/P5b go UNRESOLVED if a traceback appears.

## Generated inputs (C-SYNTH)

The targets are random gate sequences and the unitaries are seeded Haar draws. The results will report:

- the length distribution of the 20 gate targets and the 200 training examples;
- the gate kinds they use and the number of distinct targets;
- five named targets (gate names, fidelities) printed for a person to look at.

That look is Chad's and is recorded as pending until he does it.

## Trigger table (M16)

| Trigger | Control | Answer |
|---|---|---|
| Feasibility | C-FEAS | yes: the three counts above, committed with this registration |
| Noise | C-NOISE | yes: repeated synthesis is the subject of P2. Everything else is seeded, and P6b checks that two self-test runs match |
| Statistics | C-STAT | yes, one: P3b estimates a failure rate from 20,000 draws. Its band [0.45, 0.50] is about ±7 standard errors around the theoretical 0.4727, fixed here. Everything else is an exact count over fixed seeded inputs |
| Evidence | C-EVID | no: no rule's handling of evidence is in question |
| Independence | C-INDEP | yes: self-tested. Claude designed the checks, wrote the repair and runs the harness |
| External | C-EXT | no: no outside material is relied on. The Haar-sampling fact (phase-corrected QR) and Tr U ≈ complex Gaussian are standard results; P3b and G3 test them here rather than assume them |
| Device | C-DEVICE | yes: x86_64 container only (Python 3.13.18, NumPy 2.5.4, PyTorch 2.15.0 CPU). NOT VALIDATED on the S25 |
| Verdict code | C-BUILD | yes: the self-test and the harness return verdicts. Both fail closed: the self-test exits 1 on any FAIL, and the harness exits 2 on a gate failure |
| Exploration | C-EXPLORE | yes: before registering, the original self-test was run once (15/15, exit 0), and the check-4 failure rate was measured (feasibility count 3, kept as retrodiction P3a). No repaired code exists |
| Method comparison | C-METHCOMP | no |
| People's data | C-PRIVATE | no: no data about people |
| Design | C-DESIGN | yes: see Design. Existence claims and exact counts only |
| Held-out | C-HOLDOUT | no: nothing is tuned. The 0.995 threshold is the existing config value, and the seeds are fixed above |
| Generated inputs | C-SYNTH | yes: see Generated inputs |

## Limits

- One commit, one environment, one seed set.
- An untrained model stands in for a synthesizer. P4d says the gate rejects a model that cannot synthesize; it says
  nothing about a trained one.
- The DataGen check's speed bound (`rate > 1/s`) depends on the machine. It is kept as it was and is not repaired.
- `REPRODUCED.md` in this repository is a reproduction log for a different project (qsleuth). This study flags it
  and does not touch it; removing or moving it is the owner's decision.

## Next unrun test (M15)

Train the model (`Trainer.s1`, `Trainer.s2`) on the repaired generator, then measure the fidelity gate's pass rate
on held-out seeded targets, against the untrained model as baseline. Run it on the S25 as well.
