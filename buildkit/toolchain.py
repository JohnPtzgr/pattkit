"""
buildkit.toolchain: wraps the C compiler. Parses GCC diagnostics so they can be
re-pointed at .light sources (codegen emits #line directives when enabled).
"""
import re
import subprocess
from dataclasses import dataclass, field


@dataclass
class CompileResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    diagnostics: list = field(default_factory=list)


class Toolchain:
    DIAG = re.compile(r"^([^:\n]+):(\d+):(\d+):\s*(?:fatal\s+)?(\w+):\s*(.*)$")

    def __init__(self, cc="gcc", cflags=""):
        self.cc = cc
        self.cflags = cflags.split() if cflags else []

    def compile(self, c_path, o_path):
        return self._run([self.cc, *self.cflags, "-c", str(c_path), "-o", str(o_path)])

    def link(self, o_paths, out_path, libs=()):
        return self._run([self.cc, *self.cflags, *map(str, o_paths),
                          "-o", str(out_path), *libs])

    def _run(self, cmd):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return CompileResult(proc.returncode == 0, proc.returncode,
                             proc.stdout, proc.stderr, self._diags(proc.stderr))

    def _diags(self, stderr):
        out = []
        for line in stderr.splitlines():
            m = self.DIAG.match(line.strip())
            if m:
                f, ln, col, sev, msg = m.groups()
                out.append({"file": f, "line": int(ln), "col": int(col),
                            "severity": sev, "message": msg})
        return out
