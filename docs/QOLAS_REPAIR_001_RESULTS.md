# QOLAS-REPAIR-001: results

**Status:** 17 of 17 as registered. SELF-TESTED in one x86_64 container, NOT VALIDATED on the S25. The model is
still NOT TRAINED, and nothing here measures synthesis quality.
**Registration:** [QOLAS_REPAIR_001_PREREG.md](QOLAS_REPAIR_001_PREREG.md), commit `8a06455`, pushed before the
repair or the harness existed. **Build:** `a94da43`. **Pin:** `183218e`, `RECORDED = (17 × HELD, DIGEST e7ff2ee8…)`.

"17 of 17 as registered" means the predictions held. P1a, P2a, P3b, P4a and P5a, and the retrodiction P3a, predicted
that the original code was defective. Their holding confirms the defects; it does not mean the original passed
anything.

## What could have gone wrong, first

- **Self-tested.** Claude (Anthropic, Opus 5.5) registered the defects, wrote the repair, wrote the harness and
  judged the run. Nothing here is independent review. The harness decides each prediction with its own checks
  (its own logits recomputation, EOS parser and Haar sampler) rather than with the repaired code's helpers. That
  separates the checks from the code under test; it does not make them independent of their author.
- **Observed before the counted run.** The registration lists the two observations made before it: the original
  self-test once, and the check-4 failure rate. After registering, the repaired self-test and its four sabotage
  modes were run while building, before the harness's counted run: 21/21 and exit 0; each mode exit 1. Those runs
  observed what P6a and P6c later measured. No prediction, threshold or seed was changed after them.
- **One harness revision.** After its first two runs, the harness was changed to save every raw output it parses
  (`raw/`, `raw_sabotage/`). No check changed. The revised harness printed text identical to the first run, with
  the same DIGEST.
- **A seed the registration left ambiguous.** "`default_rng(20261030)` for Haar unitaries" was read as one fresh
  stream for P3b and another fresh one for P3c. So P3c's 600 unitaries are the first 600 of P3b's 20,000.
- **The sabotage verdict varies between runs.** The first sabotage run, before the revision, gave 10 of 17; the
  two after it gave 11 of 17. The difference is P6b. Under sabotage, P6b compares two runs of the *original*,
  unseeded self-test, and once they differed. That run's raw outputs were not saved, so the differing line is
  unknown. The likeliest cause, the original check 4 failing about 1 run in 72, is a hypothesis. Either way, it is
  the D3 defect showing itself.
- **Not repaired:** DataGen's speed check (`rate > 1/s`) still depends on the machine. `REPRODUCED.md` is still a
  reproduction log for a different project (qsleuth). Removing or moving it is the owner's call.
- **Not done:** the person-look at the five named generated targets (C-SYNTH) is pending; it is Chad's. No run on
  the S25.

## Raw output (pinned run, `experiments/qolas_repair_001/run3_pinned.txt`)

```
G1 original copy sha256 de76c246013b9c1d… matches registration: True
G2 repaired qolas.py sha256 36bb601cf1280a7b… differs from original: True
P1a original targets containing EOS: 0 of 200 -> HELD
P1b repaired examples with BOS+t / t+EOS structure: 200 of 200 (gates per example: {3: 33, 4: 28, 5: 16, 6: 31, 7: 19, 8: 26, 9: 28, 10: 19}) -> HELD
P2a original 'greedy', distinct sequences over 10 seeds per target: [10, 10, 10, 10, 10] -> HELD
P2b repaired greedy: distinct [1, 1, 1, 1, 1]; argmax at 250 of 250 steps -> HELD
P2c repaired sampling (T=0.8): distinct [10, 10, 10, 10, 10] -> HELD
G3 Haar sampler mean |Tr U|^2 = 1.0003 (must be in [0.9, 1.1]): True
P3b original bound under Haar draws: fails 9394 of 20000, rate 0.4697 (expected 0.4727, band [0.45, 0.50]) -> HELD
P3c repaired fidelity properties: 200 of 200; Pauli vs identity = 0: 12 of 12 -> HELD
P3d fidelity mutants caught by check 4: 3 of 3 {'a_no_conj': True, 'b_d_squared': True, 'c_real_part': True} -> HELD
P4a original check [7/8] with a synthesizer returning nothing / random gates: 4 of 4 OK -> HELD
G4 gate targets with fidelity(U, I) >= 0.995: 0 (must be 0): True
P4b oracle passes the gate: 20 of 20 -> HELD
P4c one-gate substitution: fidelity = |Tr(G^H G')|/16 in 20 of 20, gate fails 20 of 20 -> HELD
P4d untrained model passes: 0 of 20 (best fidelity 0.353553) -> HELD
P5a original with numpy.allclose forced False: exit 0, FAIL identity printed True -> HELD
P5b repaired, same patch: exit 1, FAIL identity printed True -> HELD
P6a repaired self-test: exit 0, 'RESULT: 21/21 PASS' -> HELD
P6b two runs identical after masking the timing label: True -> HELD
P6c sabotage modes: {'eos': {'exit': 1, 'fail_lines': 1}, 'greedy': {'exit': 1, 'fail_lines': 1}, 'fidelity': {'exit': 1, 'fail_lines': 4}, 'gate': {'exit': 1, 'fail_lines': 2}} -> HELD
VERDICT 17 of 17 as registered
DIGEST e7ff2ee8506462b48b48e90b1d89ac8553aa5f8ab0b5b33bb8932032c52d8c41
RECORDED: reproduced
```

Runs 1 and 2 printed the same text. The `--sabotage` run, with every "repaired" arm on the original code, gives
**11 of 17** and exits 1 ([sabotage.txt](../experiments/qolas_repair_001/sabotage.txt)). It fails P1b, P2b, P3d,
P5b, P6a and P6c, which is gate G5.

The raw outputs were opened before this was written. In `P5a_original_allclose_false.txt`, the original prints
`FAIL identity` and `RESULT: 14/15 PASS`, then exits 0. In `P5b_repaired_allclose_false.txt`, the repaired file
prints the same FAIL and exits 1. Each mutant's `[4/9]` section shows `FAIL properties`; mutant (b) also fails
`exact=1`. Each sabotage mode fails its own check.

## Predictions

| ID | Prediction | Result |
|---|---|---|
| P1a (rival) | original: 0 of 200 targets contain EOS | **HELD**: 0 |
| P1b | repaired: 200 of 200 have the BOS+t / t+EOS structure | **HELD**: 200 |
| P2a (rival) | original "greedy": ≥ 9 distinct of 10 per target | **HELD**: 10, 10, 10, 10, 10 |
| P2b | repaired greedy: 1 distinct per target, every token the argmax | **HELD**: 1 each; 250 of 250 steps |
| P2c | repaired sampling still varies: ≥ 9 distinct per target | **HELD**: 10 each |
| P3a | retrodiction: 276 of 20,000 with the original sampler | reported, not counted |
| P3b | original bound under Haar draws fails at a rate in [0.45, 0.50] | **HELD**: 0.4697 (9,394 of 20,000) |
| P3c | repaired properties 200 of 200; Pauli vs identity 12 of 12 | **HELD** |
| P3d | 3 of 3 fidelity mutants caught | **HELD** |
| P4a (rival) | original check [7/8] passes a synthesizer that returns nothing or random gates: 4 of 4 OK | **HELD** |
| P4b | the true sequence passes the gate, 20 of 20 | **HELD** |
| P4c | one-gate substitution: fidelity = \|Tr(G^H G')\|/16 and the gate fails, 20 of 20 | **HELD** |
| P4d | the untrained model passes at most 1 of 20 | **HELD**: 0 (best 0.353553) |
| P5a (rival) | original exits 0 with a FAIL printed | **HELD** |
| P5b | repaired exits 1 | **HELD** |
| P6a | `RESULT: 21/21 PASS`, exit 0 | **HELD** |
| P6b | two runs identical, timing masked | **HELD** |
| P6c | 4 of 4 sabotage modes exit 1 with a FAIL | **HELD** |

## What this shows, and what it does not

**Shows (existence, at `fd36e73`, on these seeded inputs, in this container):**

- The original training targets had no EOS.
- Its "greedy" decoding was sampling: 10 different outputs from 10 seeds, for every target.
- Its check 4 rested on a sampler artifact. The bound fails 1.4% of the time with the sampler it was written
  for, and 47% of the time with a correct Haar sampler, close to the 47.3% theory gives.
- Its check 7 passed synthesizers that return nothing.
- Its exit status ignored failures.

The repaired code passes checks for each of these, and each of those checks has been watched failing.

**Mathematical, not just measured:** a one-gate substitution changes fidelity to exactly `|Tr(G^H G')|/16`, which
is at most 0.923880 for this gate set (feasibility). A circuit one gate away from the target can never pass a
0.995 gate. P4c confirms the identity numerically on 20 cases; it was not discovered there.

**Does not show:**

- that QOLAS can synthesize anything (the model is NOT TRAINED);
- that the self-test now catches every defect (only the five registered ones were tested);
- anything on the S25, or with other versions of PyTorch or NumPy.

## Still open

- **Training (the registered door):** train with the repaired generator and measure the gate's pass rate on
  held-out targets, against the untrained baseline.
- The S25 run: `python3 qolas.py`, then each `--sabotage` mode, then the harness.
- The person-look at T0–T4 in the pinned output.
- `REPRODUCED.md`: keep, move or delete (owner's call).

## Provenance

AI participation: Claude (Anthropic, Opus 5.5) designed, registered, built, ran and wrote this up on 2026-10-29,
at Chad Holland's direction → human validation: direction only, not reviewed line by line → human
editing/curation: none → responsibility: Chad Holland.
