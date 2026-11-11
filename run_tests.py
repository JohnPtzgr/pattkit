"""Run every test script; exit non-zero on the first failure."""
import subprocess
import sys
import os

here = os.path.dirname(os.path.abspath(__file__))
for name in ("lexer", "engine", "recover", "json", "scale"):
    r = subprocess.run([sys.executable, os.path.join(here, "tests", f"test_{name}.py")],
                       capture_output=True, text=True)
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    print(f"{'PASS' if r.returncode == 0 else 'FAIL'}  test_{name}.py   {last}")
    if r.returncode:
        print(r.stdout[-1500:], r.stderr[-1500:])
        sys.exit(1)
