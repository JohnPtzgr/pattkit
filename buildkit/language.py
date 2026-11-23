"""
buildkit.language: load a language definition (lang.toml) and wire up
parsekit + astkit + codegenkit accordingly. The language is pure data.
"""
from pathlib import Path

from ..parsekit import Parser, TwoPassLexer, LexConfig
from ..astkit.builder import GenericASTBuilder
from ..codegenkit import Codegen, CodegenOptions
from .toml import load as load_toml


class LanguageError(Exception):
    pass


class Language:
    """A fully-configured language: parser + builder + codegen, from lang.toml."""

    def __init__(self, base_dir, data):
        self.base = Path(base_dir)
        self.data = data
        lang = data.get("language", {})
        self.name = lang.get("name", "lang")
        self.module = lang.get("module", self.name)
        self.parser = self._build_parser(data.get("frontend", {}))
        self.builder = GenericASTBuilder()
        self.codegen = self._build_codegen(data.get("backend", {}))

    # ---- path helpers (relative to lang.toml's directory) ----
    def _resolve(self, rel):
        return self.base / rel

    def _read(self, rel):
        return self._resolve(rel).read_text(encoding="utf-8")

    # ---- input: source -> parsekit AST ----
    def _build_parser(self, fe):
        grammar = fe.get("grammar")
        if isinstance(grammar, dict):
            if "file" in grammar:
                grammar_text = self._read(grammar["file"])
            elif "inline" in grammar:
                grammar_text = grammar["inline"]
            else:
                raise LanguageError("frontend.grammar needs 'file' or 'inline'")
        elif isinstance(grammar, str):
            grammar_text = self._read(grammar)           # bare string = file path
        else:
            raise LanguageError("frontend.grammar is required")

        lx = fe.get("lexer", {})
        lexer = TwoPassLexer(LexConfig(
            layout=lx.get("layout", True),
            space_tokens=lx.get("space_tokens", False),
            brackets=lx.get("brackets", "lenient"),
            comments=lx.get("comments", "drop"),
            tab=lx.get("tab", 2),
        ))
        return Parser(
            grammar_text,
            fe.get("start", "program"),
            lexer=lexer,
            builtins=fe.get("builtins", {}),
        )

    # ---- output: Boundary AST -> C ----
    def _build_codegen(self, be):
        opts = CodegenOptions(
            module=self.module,
            default_includes=tuple(be.get("default_includes", [])),
            banner=be.get("banner", True),
            line_directives=be.get("line_directives", False),
        )
        types = be.get("types", {})
        if types:
            opts.type_map.update(types)                  # lang overrides/extends defaults
        cg = Codegen(opts, builder=self.builder)
        for pat in be.get("patterns", []):
            cg.load_file(self._resolve(pat))
        return cg

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            raise LanguageError(f"language file not found: {path}")
        return cls(path.parent, load_toml(path))
