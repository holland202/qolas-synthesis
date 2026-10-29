#!/usr/bin/env python3
"""QOLAS-REPAIR-001 harness. Registered in docs/QOLAS_REPAIR_001_PREREG.md (commit 8a06455) before this file existed.

Run from the repository root:
    python3 experiments/qolas_repair_001/run_qolas_repair_001.py             # exit 0 only on the RECORDED outcome
    python3 experiments/qolas_repair_001/run_qolas_repair_001.py --sabotage  # repaired arms run on the original code; must exit 1

Imports the original code (verbatim copy of qolas.py at fd36e73) and the repaired qolas.py, and runs their real
classes. Checks that decide a prediction are written here, not taken from the repaired code: the argmax check
recomputes logits itself, the EOS check parses the tensors itself, and the self-test runs as a subprocess.
Exit codes: 2 = an instrument gate failed (no verdict); 0 = outcome equals RECORDED (or, before pinning, all 17
HELD); 1 = anything else.
"""
import collections
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ORIG_PATH = os.path.join(HERE, "qolas_fd36e73.py")
FIXED_PATH = os.path.join(ROOT, "qolas.py")
ORIG_SHA = "de76c246013b9c1d6d7b24476cdac3b2ffda43cd9c3b3bf3ddd4570415dc8b12"
IDS = ["P1a", "P1b", "P2a", "P2b", "P2c", "P3b", "P3c", "P3d", "P4a", "P4b", "P4c", "P4d",
       "P5a", "P5b", "P6a", "P6b", "P6c"]
RECORDED = None  # pinned after the first full run, in a separate commit: (statuses tuple, digest)


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def haar(rng, d=16):
    """The harness's own Haar sampler (phase-corrected QR), independent of the repaired code's."""
    z = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    q, r = np.linalg.qr(z)
    dg = np.diag(r)
    return q * (dg / np.abs(dg))


def gen_batch(mod, gs, B, seed):
    d = mod.SyntheticDataGenerator(mod.QOLASConfig(), gs)
    try:
        return d.generate_batch(B, rng=np.random.default_rng(seed))
    except TypeError:  # the original has no rng parameter: seed its global stream instead
        np.random.seed(seed)
        return d.generate_batch(B)


def targets(mod, gs, n, seed):
    """n gate sequences of 3..10 gates: (tokens, unitary)."""
    out = []
    rng = np.random.default_rng(seed)
    legacy = "rng" not in mod.GateSet.random_sequence.__code__.co_varnames
    if legacy:
        np.random.seed(seed)
    for _ in range(n):
        if legacy:
            _, t = gs.random_sequence(np.random.randint(3, 11))
        else:
            _, t = gs.random_sequence(int(rng.integers(3, 11)), rng)
        out.append((t, gs.apply_sequence(gs.tokens_to_gates(t))))
    return out


def build(mod):
    c = mod.QOLASConfig()
    gs = mod.GateSet(c.n_qubits)
    c.vocab_size = gs.vocab_size
    torch.manual_seed(0)
    m = mod.QOLASModel(c, gs.vocab_size)
    return c, gs, m, mod.SynthesisEngine(m, c, gs)


def next_logits(m, pr, gen):
    """Next-token logits, computed the way the original generate() did, without the repaired code's helper."""
    ge = m.te(gen)
    x = torch.cat([pr, ge], dim=1)
    mask = m._mask(pr.size(1), x.size(1), x.device)
    h = x
    for layer in m.layers:
        h = layer(h, mask)
    return m.head(m.norm(h)[:, -1, :])


def argmax_steps(m, gs, U, toks):
    """(steps checked, steps whose token is the argmax)."""
    m.eval()
    with torch.no_grad():
        pr = m.proj(m.enc(torch.from_numpy(gs.pauli.compute_ptm(U)).float().unsqueeze(0)))
        gen = torch.tensor([[1]])  # BOS
        good = 0
        for t in toks:
            good += int(int(next_logits(m, pr, gen).argmax(dim=-1)) == t)
            gen = torch.cat([gen, torch.tensor([[t]])], dim=1)
    return len(toks), good


def eos_structure_ok(inp, tgt, gs):
    inp, tgt = inp.tolist(), tgt.tolist()
    if 2 not in tgt:
        return False
    k = tgt.index(2)
    t = tgt[:k]
    return (3 <= k <= 10 and all(x in gs.token_to_gate for x in t) and inp[0] == 1 and inp[1:k + 1] == t
            and all(x == 0 for x in inp[k + 1:]) and all(x == 0 for x in tgt[k + 1:]))


def section(text, label):
    """Lines of the self-test section whose header starts with `label` (e.g. '[7/')."""
    out, on = [], False
    for ln in text.splitlines():
        if ln.startswith("["):
            on = ln.startswith(label)
            continue
        if on and ln.strip():
            out.append(ln)
    return out


def run_script(path, args=(), patch_allclose=False):
    if patch_allclose:
        code = ("import numpy, runpy, sys; numpy.allclose = lambda *a, **k: False; "
                f"sys.argv = ['qolas.py']; runpy.run_path({path!r}, run_name='__main__')")
        cmd = [sys.executable, "-B", "-c", code]
    else:
        cmd = [sys.executable, "-B", path, *args]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode, p.stdout.decode("utf-8", errors="replace")


def save_raw(sabotage, name, text):
    """Keep every raw output a prediction is decided from, so it can be opened and read (SWAY A4 D7)."""
    d = os.path.join(HERE, "raw_sabotage" if sabotage else "raw")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".txt"), "w", encoding="utf-8") as fh:
        fh.write(text)


def main(argv):
    sabotage = argv == ["--sabotage"]
    if argv and not sabotage:
        sys.exit(__doc__)
    print(f"QOLAS-REPAIR-001 harness{'  *** SABOTAGE: repaired arms run on the original code ***' if sabotage else ''}")
    print(f"environment: {platform.machine()} | Python {platform.python_version()} | NumPy {np.__version__} | "
          f"PyTorch {torch.__version__}")
    gates, R, S = {}, {}, {}

    orig_sha, fixed_sha = sha256(ORIG_PATH), sha256(FIXED_PATH)
    gates["G1"] = orig_sha == ORIG_SHA
    gates["G2"] = fixed_sha != ORIG_SHA
    print(f"G1 original copy sha256 {orig_sha[:16]}… matches registration: {gates['G1']}")
    print(f"G2 repaired qolas.py sha256 {fixed_sha[:16]}… differs from original: {gates['G2']}")
    orig = load(ORIG_PATH, "qolas_original")
    fixed = orig if sabotage else load(FIXED_PATH, "qolas_repaired")
    fixed_path = ORIG_PATH if sabotage else FIXED_PATH

    # ---- P1: EOS in training targets
    _, gs_o, _, _ = build(orig)
    _, _, T = gen_batch(orig, gs_o, 200, 1)
    n_eos = int(sum(bool((T[i] == 2).any()) for i in range(200)))
    R["P1a"] = {"targets_with_eos": n_eos}
    S["P1a"] = "HELD" if n_eos == 0 else "REFUTED"
    _, gs_f, _, _ = build(fixed)
    _, I, T = gen_batch(fixed, gs_f, 200, 1)
    ok = [eos_structure_ok(I[i], T[i], gs_f) for i in range(200)]
    lens = collections.Counter(int((T[i] != 0).sum()) - 1 for i in range(200)) if all(ok) else {}
    R["P1b"] = {"examples_ok": int(sum(ok)), "gate_count_distribution": dict(sorted(lens.items()))}
    S["P1b"] = "HELD" if sum(ok) == 200 else "REFUTED"
    print(f"P1a original targets containing EOS: {n_eos} of 200 -> {S['P1a']}")
    print(f"P1b repaired examples with BOS+t / t+EOS structure: {sum(ok)} of 200 "
          f"(gates per example: {dict(sorted(lens.items()))}) -> {S['P1b']}")

    # ---- P2: greedy decoding
    def distinct_runs(engine, U, strategy):
        seqs = []
        for s in range(10):
            torch.manual_seed(s)
            seqs.append(tuple(engine.synthesize(U, strategy)["tokens"]))
        return seqs

    _, gs_o, m_o, e_o = build(orig)
    d2a = [len(set(distinct_runs(e_o, U, "greedy"))) for _, U in targets(orig, gs_o, 5, 2)]
    R["P2a"] = {"distinct_per_target": d2a}
    S["P2a"] = "HELD" if min(d2a) >= 9 else "REFUTED"
    _, gs_f, m_f, e_f = build(fixed)
    t5 = targets(fixed, gs_f, 5, 2)
    d2b, checked, good = [], 0, 0
    for _, U in t5:
        seqs = distinct_runs(e_f, U, "greedy")
        d2b.append(len(set(seqs)))
        n, g = argmax_steps(m_f, gs_f, U, list(seqs[0]))
        checked, good = checked + n, good + g
    R["P2b"] = {"distinct_per_target": d2b, "steps_checked": checked, "steps_argmax": good}
    S["P2b"] = ("UNRESOLVED" if checked < 10 else
                "HELD" if max(d2b) == 1 and good == checked else "REFUTED")
    d2c = [len(set(distinct_runs(e_f, U, "sample"))) for _, U in t5]
    R["P2c"] = {"distinct_per_target": d2c}
    S["P2c"] = "HELD" if min(d2c) >= 9 else "REFUTED"
    print(f"P2a original 'greedy', distinct sequences over 10 seeds per target: {d2a} -> {S['P2a']}")
    print(f"P2b repaired greedy: distinct {d2b}; argmax at {good} of {checked} steps -> {S['P2b']}")
    print(f"P2c repaired sampling (T=0.8): distinct {d2c} -> {S['P2c']}")

    # ---- P3: fidelity check
    rng = np.random.default_rng(20261030)
    I16 = np.eye(16, dtype=complex)
    pb_o = orig.PauliBasis(4)  # the original check: 0.05 < compute_fidelity(I, Ur) < 0.45
    tr2, fails = [], 0
    for _ in range(20000):
        U = haar(rng)
        tr2.append(abs(np.trace(U)) ** 2)
        fails += not (0.05 < pb_o.compute_fidelity(I16, U) < 0.45)
    gates["G3"] = 0.9 <= float(np.mean(tr2)) <= 1.1
    rate = fails / 20000
    R["P3b"] = {"fails": fails, "rate": round(rate, 4), "mean_abs_tr_sq": round(float(np.mean(tr2)), 4)}
    S["P3b"] = "HELD" if 0.45 <= rate <= 0.50 else "REFUTED"
    print(f"G3 Haar sampler mean |Tr U|^2 = {np.mean(tr2):.4f} (must be in [0.9, 1.1]): {gates['G3']}")
    print(f"P3b original bound under Haar draws: fails {fails} of 20000, rate {rate:.4f} "
          f"(expected 0.4727, band [0.45, 0.50]) -> {S['P3b']}")

    pb = fixed.PauliBasis(4)
    rng = np.random.default_rng(20261030)
    prop_ok = 0
    for _ in range(200):
        A, B, W = haar(rng), haar(rng), haar(rng)
        phi = rng.uniform(0, 2 * np.pi)
        f = pb.compute_fidelity(A, B)
        prop_ok += int(abs(pb.compute_fidelity(A, A) - 1) < 1e-9
                       and abs(pb.compute_fidelity(A, np.exp(1j * phi) * A) - 1) < 1e-9
                       and abs(f - pb.compute_fidelity(B, A)) < 1e-9
                       and abs(pb.compute_fidelity(W @ A, W @ B) - f) < 1e-9 and 0 <= f <= 1 + 1e-9)
    pauli_zero = sum(int(abs(pb.compute_fidelity(gs_f.token_to_gate[gs_f.gate_to_token[(P, q)]], I16)) < 1e-12)
                     for P in ("X", "Y", "Z") for q in range(4))
    R["P3c"] = {"properties_ok": prop_ok, "pauli_vs_identity_zero": pauli_zero}
    S["P3c"] = "HELD" if prop_ok == 200 and pauli_zero == 12 else "REFUTED"
    print(f"P3c repaired fidelity properties: {prop_ok} of 200; Pauli vs identity = 0: {pauli_zero} of 12 "
          f"-> {S['P3c']}")

    mutants = {
        "a_no_conj": "lambda self,U1,U2: float(np.abs(np.trace(U1.T@U2))/self.d)",
        "b_d_squared": "lambda self,U1,U2: float(np.abs(np.trace(U1.conj().T@U2))/self.d**2)",
        "c_real_part": "lambda self,U1,U2: float(np.real(np.trace(U1.conj().T@U2))/self.d)",
    }
    killed = {}
    for name, lam in mutants.items():
        code = ("import importlib.util, sys, numpy as np\n"
                f"spec = importlib.util.spec_from_file_location('q', {fixed_path!r}); q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)\n"
                f"q.PauliBasis.compute_fidelity = {lam}\n"
                "q.test()\n")
        p = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = p.stdout.decode("utf-8", errors="replace")
        save_raw(sabotage, f"P3d_mutant_{name}", out)
        killed[name] = any(ln.strip().startswith("FAIL") for ln in section(out, "[4/"))
    R["P3d"] = {"killed": killed}
    S["P3d"] = "HELD" if all(killed.values()) else "REFUTED"
    print(f"P3d fidelity mutants caught by check 4: {sum(killed.values())} of 3 {killed} -> {S['P3d']}")

    # ---- P4: synthesis check and fidelity gate
    items_ok = 0
    gate_tokens = sorted(build(orig)[1].token_to_gate)
    for variant in ("empty", "random"):
        def fake_generate(self, ptm, max_steps=50, temp=0.8, _v=variant):
            if _v == "empty":
                return torch.zeros((1, 0), dtype=torch.long)
            r = np.random.default_rng(7)
            return torch.tensor([[int(x) for x in r.choice(gate_tokens, 50)]])
        saved = orig.QOLASModel.generate
        orig.QOLASModel.generate = fake_generate
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                orig.test()
        finally:
            orig.QOLASModel.generate = saved
        save_raw(sabotage, f"P4a_original_selftest_{variant}", buf.getvalue())
        items_ok += sum(ln.strip().startswith("OK") for ln in section(buf.getvalue(), "[7/"))
    R["P4a"] = {"items_ok": items_ok}
    S["P4a"] = "HELD" if items_ok == 4 else "REFUTED"
    print(f"P4a original check [7/8] with a synthesizer returning nothing / random gates: {items_ok} of 4 OK "
          f"-> {S['P4a']}")

    c_f, gs_f, m_f, e_f = build(fixed)
    passes = getattr(e_f, "passes", None) or (lambda f: f >= c_f.fidelity_threshold)
    t20 = targets(fixed, gs_f, 20, 3)
    deg = sum(gs_f.pauli.compute_fidelity(U, np.eye(16, dtype=complex)) >= 0.995 for _, U in t20)
    gates["G4"] = deg == 0
    print(f"G4 gate targets with fidelity(U, I) >= 0.995: {deg} (must be 0): {gates['G4']}")
    oracle = sum(int(passes(gs_f.pauli.compute_fidelity(U, gs_f.apply_sequence(gs_f.tokens_to_gates(t))))
                     and gs_f.pauli.compute_fidelity(U, gs_f.apply_sequence(gs_f.tokens_to_gates(t))) >= 1 - 1e-9)
                 for t, U in t20)
    r4 = np.random.default_rng(4)
    sub_match = sub_fail = 0
    look = []
    model_f = []
    for k, (t, U) in enumerate(t20):
        i = int(r4.integers(len(t)))
        new = int(r4.choice([x for x in gs_f.token_to_gate if x != t[i]]))
        t2 = t[:i] + [new] + t[i + 1:]
        f = gs_f.pauli.compute_fidelity(U, gs_f.apply_sequence(gs_f.tokens_to_gates(t2)))
        expect = gs_f.pauli.compute_fidelity(gs_f.token_to_gate[t[i]], gs_f.token_to_gate[new])
        sub_match += int(abs(f - expect) < 1e-9)
        sub_fail += int(not passes(f))
        torch.manual_seed(0)
        fm = e_f.synthesize(U, "greedy")["fidelity"]
        model_f.append(fm)
        if k < 5:
            look.append({"target": [gs_f.token_to_name[x] for x in t], "swapped": f"{gs_f.token_to_name[t[i]]} -> "
                         f"{gs_f.token_to_name[new]} at {i}", "substitution_fidelity": round(f, 6),
                         "untrained_model_fidelity": round(fm, 6)})
    model_pass = sum(int(passes(f)) for f in model_f)
    R["P4b"] = {"oracle_pass": oracle}
    S["P4b"] = "HELD" if oracle == 20 else "REFUTED"
    R["P4c"] = {"fidelity_matches_formula": sub_match, "gate_fails": sub_fail}
    S["P4c"] = "HELD" if sub_match == 20 and sub_fail == 20 else "REFUTED"
    R["P4d"] = {"untrained_passes": model_pass, "max_fidelity": round(max(model_f), 6)}
    S["P4d"] = "HELD" if model_pass <= 1 else "REFUTED"
    tl = collections.Counter(len(t) for t, _ in t20)
    kinds = collections.Counter(gs_f.token_to_name[x].split("[")[0] for t, _ in t20 for x in t)
    print(f"P4b oracle passes the gate: {oracle} of 20 -> {S['P4b']}")
    print(f"P4c one-gate substitution: fidelity = |Tr(G^H G')|/16 in {sub_match} of 20, gate fails {sub_fail} of 20 "
          f"-> {S['P4c']}")
    print(f"P4d untrained model passes: {model_pass} of 20 (best fidelity {max(model_f):.6f}) -> {S['P4d']}")
    print(f"C-SYNTH gate targets: lengths {dict(sorted(tl.items()))}, distinct {len(set(tuple(t) for t, _ in t20))} "
          f"of 20, gate kinds {dict(sorted(kinds.items()))}")
    print("C-SYNTH five named targets for a person to look at:")
    for k, row in enumerate(look):
        print(f"  T{k}: {' '.join(row['target'])} | swap {row['swapped']} -> fidelity {row['substitution_fidelity']} "
              f"| untrained model {row['untrained_model_fidelity']}")

    # ---- P5: exit status
    rc, out = run_script(ORIG_PATH, patch_allclose=True)
    save_raw(sabotage, "P5a_original_allclose_false", out + f"exit {rc}\n")
    R["P5a"] = {"exit": rc, "fail_identity": "FAIL identity" in out, "traceback": "Traceback" in out}
    S["P5a"] = ("UNRESOLVED" if "Traceback" in out else
                "HELD" if rc == 0 and "FAIL identity" in out else "REFUTED")
    rc, out = run_script(fixed_path, patch_allclose=True)
    save_raw(sabotage, "P5b_repaired_allclose_false", out + f"exit {rc}\n")
    R["P5b"] = {"exit": rc, "fail_identity": "FAIL identity" in out, "traceback": "Traceback" in out}
    S["P5b"] = ("UNRESOLVED" if "Traceback" in out else
                "HELD" if rc == 1 and "FAIL identity" in out else "REFUTED")
    print(f"P5a original with numpy.allclose forced False: exit {R['P5a']['exit']}, FAIL identity printed "
          f"{R['P5a']['fail_identity']} -> {S['P5a']}")
    print(f"P5b repaired, same patch: exit {R['P5b']['exit']}, FAIL identity printed {R['P5b']['fail_identity']} "
          f"-> {S['P5b']}")

    # ---- P6: the repaired self-test
    rc1, out1 = run_script(fixed_path)
    rc2, out2 = run_script(fixed_path)
    save_raw(sabotage, "P6_selftest_run1", out1 + f"exit {rc1}\n")
    save_raw(sabotage, "P6_selftest_run2", out2 + f"exit {rc2}\n")
    mask = lambda s: re.sub(r"50 in [\d.]+s \([\d.]+/s\)", "50 in Xs (X/s)", s)
    result_line = next((ln for ln in out1.splitlines() if ln.startswith("RESULT:")), "")
    R["P6a"] = {"exit": rc1, "result": result_line}
    S["P6a"] = "HELD" if rc1 == 0 and result_line == "RESULT: 21/21 PASS" else "REFUTED"
    R["P6b"] = {"identical_after_mask": mask(out1) == mask(out2)}
    S["P6b"] = "HELD" if mask(out1) == mask(out2) else "REFUTED"
    sab = {}
    for mode in ("eos", "greedy", "fidelity", "gate"):
        rc, out = run_script(fixed_path, ["--sabotage", mode])
        save_raw(sabotage, f"P6c_sabotage_{mode}", out + f"exit {rc}\n")
        fail_lines = [ln for ln in out.splitlines() if re.match(r"^  FAIL ", ln)]
        sab[mode] = {"exit": rc, "fail_lines": len(fail_lines)}
    R["P6c"] = {"modes": sab}
    S["P6c"] = "HELD" if all(v["exit"] == 1 and v["fail_lines"] >= 1 for v in sab.values()) else "REFUTED"
    print(f"P6a repaired self-test: exit {rc1}, '{result_line}' -> {S['P6a']}")
    print(f"P6b two runs identical after masking the timing label: {R['P6b']['identical_after_mask']} -> {S['P6b']}")
    print(f"P6c sabotage modes: {sab} -> {S['P6c']}")

    # ---- verdict
    if not all(gates.values()):
        print(f"GATE FAILURE {[g for g, v in gates.items() if not v]}: no verdict")
        return 2
    held = tuple(S[i] for i in IDS)
    canon = json.dumps({"statuses": dict(zip(IDS, held)), "results": R}, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canon.encode()).hexdigest()
    n_held = sum(s == "HELD" for s in held)
    print(f"VERDICT {n_held} of {len(IDS)} as registered")
    print(f"DIGEST {digest}")
    with open(os.path.join(HERE, "results_sabotage.json" if sabotage else "results.json"), "w") as fh:
        json.dump({"statuses": dict(zip(IDS, held)), "results": R, "digest": digest}, fh, indent=1, sort_keys=True)
    if RECORDED is None:
        print("RECORDED: not pinned yet")
        return 0 if n_held == len(IDS) else 1
    match = (held, digest) == RECORDED
    print(f"RECORDED: {'reproduced' if match else 'NOT reproduced'}")
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
