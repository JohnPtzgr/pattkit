"""
buildkit.driver: M1 orchestrator. Sequential, no caching yet.

    manifest -> for each source: parse -> Boundary AST -> codegen -> gcc -> .o
    then link all .o -> executable
"""
import subprocess
from pathlib import Path

from ..parsekit import ParseError
from .manifest import Manifest
from .toolchain import Toolchain


class BuildError(Exception):
    pass


class BuildDriver:
    def __init__(self, project_toml, verbose=True):
        self.manifest = Manifest.load(project_toml)
        self.toolchain = Toolchain(self.manifest.cc, self.manifest.cflags)
        self.build_dir = self.manifest.root / self.manifest.build_dir
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(f"[build] {msg}")

    # ---- whole project ----
    def build(self):
        objects = [self._compile_module(src) for src in self.manifest.sources]
        out = self.build_dir / self.manifest.output
        libs = [self.manifest.runtime] if self.manifest.runtime else []
        res = self.toolchain.link(objects, out, libs)
        if not res.ok:
            raise BuildError(f"link failed:\n{res.stderr}")
        self._log(f"linked {out}")
        return out

    def run(self, *args):
        exe = self.build()
        return subprocess.run([str(exe), *args])

    def clean(self):
        for p in self.build_dir.iterdir():
            if p.is_file():
                p.unlink()
        self._log(f"cleaned {self.build_dir}")

    # ---- one module ----
    def _compile_module(self, src_path):
        lang = self.manifest.language
        text = src_path.read_text(encoding="utf-8")

        # 1. parse (parsekit)
        try:
            ast = lang.parser.parse(text)
        except ParseError as e:
            raise BuildError(f"{src_path.name}: {e}") from None

        # 2. Boundary AST (astkit)
        boundary = lang.builder.build(ast)

        # 3. C source (codegenkit)
        module = src_path.stem
        result = lang.codegen.generate_module(
            boundary, module_name=module, header_name=f"{module}.h",
        )
        c_path = self.build_dir / f"{module}.c"
        c_path.write_text(result.c_source, encoding="utf-8")
        self._log(f"codegen {src_path.name} -> {c_path.name}")

        # 4. object file (gcc)
        o_path = self.build_dir / f"{module}.o"
        res = self.toolchain.compile(c_path, o_path)
        if not res.ok:
            raise BuildError(f"compiling {src_path.name} failed:\n{res.stderr}")
        self._log(f"compiled {c_path.name} -> {o_path.name}")
        return o_path
