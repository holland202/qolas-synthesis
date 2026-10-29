# Lesson: checks that cannot fail (from the QOLAS repair)

A short lesson from building QOLAS-REPAIR-001. It teaches; the evidence is in
[QOLAS_REPAIR_001_RESULTS.md](../QOLAS_REPAIR_001_RESULTS.md).

## 1. A self-test's exit status is part of the test

`if __name__ == "__main__": test()` throws away what `test()` returns. The script printed `FAIL identity` and
`RESULT: 14/15 PASS`, then exited 0. A person reading the screen sees the failure. A shell (`&&`), a CI job or
another script sees only the 0. The repair is one line:

```python
sys.exit(0 if test() else 1)
```

The habit to take away: after writing any check, make it fail on purpose once and look at `$?`. Check `$?` before
piping, because `cmd | tail` reports tail's status, not cmd's.

## 2. A check must be able to say no

The old synthesis check asked whether the result had a `gate_sequence` key and whether `fidelity` was a float. A
synthesizer returning nothing passes both. The repair asks questions that have wrong answers:

- the true circuit must pass;
- a circuit one gate away must fail;
- an untrained model must fail.

`python3 qolas.py --sabotage gate` replaces the gate with one that passes everything, and two checks go red. A check
you have never seen fail is a log line, not a check.

## 3. "Random unitary" needs one more step

`numpy.linalg.qr(randn + 1j*randn)` gives a unitary Q, but not a uniformly random one. The phases on the diagonal
of R leak into Q. Fix them:

```python
Q, R = np.linalg.qr(Z)
d = np.diag(R)
Q = Q * (d / np.abs(d))   # now Haar-random
```

How to tell: for a Haar-random U, the average of |Tr U|² is exactly 1. Without the fix it came out 4.05; with it,
1.0003. The old self-test's bound `0.05 < |Tr U|/16 < 0.45` only fit the biased sampler. With correct sampling it
fails about 47% of the time. The repair tests properties that hold for every unitary (f(U,U) = 1, global phase
ignored, symmetry, unitary invariance) instead of a range tuned to one sampler.

## 4. Greedy means argmax

Sampling at temperature 1 is not greedy decoding. If "greedy" gives a different answer on each run, it is sampling.
The test is cheap: run it under several seeds and compare.
