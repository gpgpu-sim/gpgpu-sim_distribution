#!/usr/bin/env python3
"""Reference results for the functional-check tier.

Most pannotia and lonestargpu apps do not check their own answer, so the
tier compares what they print and write under GPGPU-Sim against the same
binaries run natively on a GPU.

  functional-refs.py capture DEFINE.yml OUT.json   run every app in DEFINE.yml
                                                   on the GPU (twice) and save
                                                   a summary of its results
  functional-refs.py compare REFS.json SIM_RUN     compare the summaries of the
                                                   simulator's runs in SIM_RUN
                                                   (accel-sim-framework/sim_run_*),
                                                   and fail any run whose own
                                                   check failed

A summary keeps the stdout lines that carry a result (KEY_LINES) and a digest
of each output file (OUTPUT_FILES): a hash when the file holds only integers,
otherwise the count, sum and evenly spaced samples of its numbers. Numbers are
compared exactly when they are integers and to about the printed precision
when they are not. A field that differs between the two GPU runs is recorded
as nondeterministic and not compared.
"""

import glob
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys

import yaml

KEY_LINES = re.compile(
    r"no of errors = "
    r"|result: weight"
    r"|total number of colors used"
    r"|number of iterations"
    r"|bad triangles"
    r"|Results are correct|mismatch at"
    r"|FAILED"
    # lonestar-bh: position of body 0 after the last time step
    r"|^-?\d\.\d+e[+-]\d+ -?\d\.\d+e[+-]\d+ -?\d\.\d+e[+-]\d+$"
)
OUTPUT_FILES = ["result.out", "bfs-output.txt", "sssp-output.txt"]
NUMBER = re.compile(r"-?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|nan|inf)")
# lines with which an app reports that its own check failed
SELF_CHECK_FAILED = re.compile(
    r"FAILED|mismatch at|no of errors = [1-9]|^[1-9]\d* final bad triangles")
SAMPLES = 64
REL_TOL = 1e-3


def argfolder(args):
    # Accel-Sim's run folder name for an argument string (common.py)
    args = "" if args is None else str(args)
    return re.sub(r"[^a-z^A-Z^0-9]", "_", args.strip()) if args.strip() else "NO_ARGS"


def key_lines(text):
    return [l.strip() for l in text.splitlines() if KEY_LINES.search(l.strip())]


def digest(path):
    toks = NUMBER.findall(open(path, errors="replace").read())
    if all(re.fullmatch(r"-?\d+", t) for t in toks):
        return {"n": len(toks), "sha256": hashlib.sha256(" ".join(toks).encode()).hexdigest()}
    vals = [float(t) for t in toks]
    step = max(1, len(vals) // SAMPLES)
    return {
        "n": len(vals),
        "sum": math.fsum(v for v in vals if math.isfinite(v)),
        "samples": [toks[i] for i in range(0, len(toks), step)][:SAMPLES],
    }


def summarize(run_dir, stdout_text):
    files = {}
    for f in OUTPUT_FILES:
        p = os.path.join(run_dir, f)
        if os.path.isfile(p):
            files[f] = digest(p)
    return {"lines": key_lines(stdout_text), "files": files}


def unit(tok):
    # one unit in the last printed digit of a number
    m = re.fullmatch(r"-?\d+(?:\.(\d*))?(?:[eE]([+-]?\d+))?", tok)
    if not m:
        return 0.0
    return 10.0 ** (int(m.group(2) or 0) - len(m.group(1) or ""))


def same_number(a, b):
    if a == b:
        return True
    if re.fullmatch(r"-?\d+", a) and re.fullmatch(r"-?\d+", b):
        return False
    try:
        x, y = float(a), float(b)
    except ValueError:
        return False
    # 1.5 units: values that round to neighbouring printed digits still match
    return abs(x - y) <= max(REL_TOL * max(abs(x), abs(y)), 1.5 * max(unit(a), unit(b)))


def same_line(a, b):
    ta, tb = NUMBER.findall(a), NUMBER.findall(b)
    return (NUMBER.sub("#", a) == NUMBER.sub("#", b) and len(ta) == len(tb)
            and all(same_number(x, y) for x, y in zip(ta, tb)))


def differences(ref, got):
    """Fields of summary got that do not match summary ref."""
    out = []
    skip = set(ref.get("nondet", []))
    if "lines" not in skip:
        if len(ref["lines"]) != len(got["lines"]) or not all(
                same_line(a, b) for a, b in zip(ref["lines"], got["lines"])):
            out.append(("lines", ref["lines"], got["lines"]))
    for f, r in ref["files"].items():
        if f in skip:
            continue
        g = got["files"].get(f)
        if g is None:
            out.append((f, r, "missing"))
        elif "sha256" in r:
            if r != g:
                out.append((f, r, g))
        elif ("sum" not in g or r["n"] != g["n"] or not same_number(repr(r["sum"]), repr(g["sum"]))
              or len(r["samples"]) != len(g["samples"])
              or not all(same_number(x, y) for x, y in zip(r["samples"], g["samples"]))):
            out.append((f, r, g))
    return out


def capture(define, out_json):
    import tempfile
    d = yaml.safe_load(open(define))
    only = os.environ.get("SUITES", "")
    only = only.split(",") if only else list(d)
    timeout = int(os.environ.get("REF_TIMEOUT", "1800"))
    refs = {}
    work = tempfile.mkdtemp(prefix="gpu-refs-")
    for suite, v in d.items():
        if suite not in only:
            continue
        exec_dir = os.path.expandvars(v["exec_dir"])
        data_dirs = os.path.expandvars(v["data_dirs"])
        for e in v["execs"]:
            exe, argl = list(e.items())[0]
            for a in argl:
                args = a.get("args") or ""
                key = "%s/%s" % (exe, argfolder(args))
                runs = []
                for n in (1, 2):
                    rd = os.path.join(work, str(n), exe, argfolder(args))
                    os.makedirs(rd)
                    data = os.path.join(data_dirs, exe, "data")
                    if os.path.isdir(data):
                        os.symlink(data, os.path.join(rd, "data"))
                    cmd = shlex.quote(os.path.join(exec_dir, exe)) + " " + args
                    try:
                        p = subprocess.run(cmd, shell=True, cwd=rd, timeout=timeout,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           errors="replace")
                        rc, text = p.returncode, p.stdout
                    except subprocess.TimeoutExpired as t:
                        rc, text = "timeout", (t.stdout or b"").decode(errors="replace")
                    s = summarize(rd, text)
                    s["rc"] = rc
                    runs.append(s)
                    if rc != 0:
                        print("%s: run %d exited with %s; last lines:\n%s" % (key, n, rc, "\n".join(text.splitlines()[-10:])))
                        break
                ref = runs[0]
                if len(runs) == 2:
                    nondet = []
                    if differences(runs[0], runs[1]):
                        nondet = [f for f, _, _ in differences(runs[0], runs[1])]
                    if nondet:
                        ref["nondet"] = nondet
                refs[key] = ref
                print("%s: rc=%s lines=%d files=%s%s" % (
                    key, ref["rc"], len(ref["lines"]), ",".join(ref["files"]),
                    " nondeterministic=" + ",".join(ref["nondet"]) if ref.get("nondet") else ""))
    json.dump(refs, open(out_json, "w"), indent=1, sort_keys=True)


def compare(refs_json, sim_run):
    refs = json.load(open(refs_json))
    rc = 0
    seen = 0
    for o in sorted(glob.glob(os.path.join(sim_run, "*", "*", "*", "*.o[0-9]*"))):
        rd = os.path.dirname(o)
        exe, af = rd.split(os.sep)[-3:-1]
        key = "%s/%s" % (exe, af)
        text = open(o, errors="replace").read()
        bad = [l for l in key_lines(text) if SELF_CHECK_FAILED.search(l)]
        if bad:
            rc = 1
            print("::error::%s failed its own check: %s" % (key, "; ".join(bad[:3])))
            continue
        if key not in refs:
            print("%s: no GPU reference" % key)
            continue
        ref = refs[key]
        if ref.get("rc") != 0:
            print("%s: the GPU run failed (%s); not compared" % (key, ref.get("rc")))
            continue
        seen += 1
        diff = differences(ref, summarize(rd, text))
        if diff:
            rc = 1
            print("::error::%s differs from the GPU" % key)
            for field, r, g in diff:
                print("  %s\n    GPU: %s\n    sim: %s" % (field, json.dumps(r)[:600], json.dumps(g)[:600]))
        else:
            nd = " (not compared: %s)" % ",".join(ref["nondet"]) if ref.get("nondet") else ""
            print("%s: matches the GPU%s" % (key, nd))
    print("%d runs compared with GPU references" % seen)
    return rc


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "capture":
        capture(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    else:
        sys.exit(__doc__)
