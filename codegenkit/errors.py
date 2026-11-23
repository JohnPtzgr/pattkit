"""Diagnostics for the codegen layer."""


def where(tok):
    if tok is None:
        return "<unknown>"
    file = getattr(tok, "file", None) or getattr(tok, "path", None) or "<source>"
    line = getattr(tok, "line", 0) or 0
    col = getattr(tok, "col", 0) or 0
    return f"{file}:{line}:{col}"


def node_label(node):
    tag = getattr(node, "tag", None)
    if tag:
        return f"{type(node).__name__}({tag})"
    return type(node).__name__


class CodegenError(Exception):
    """Raised for both DSL errors and emission failures."""

    def __init__(self, message, *, tok=None, origin=None, lineno=None, stack=()):
        self.message = message
        self.tok = tok
        self.origin = origin
        self.lineno = lineno
        self.stack = tuple(stack)
        super().__init__(self._format())

    def _format(self):
        bits = [self.message]
        if self.origin:
            bits.append(f"  in pattern {self.origin}:{self.lineno or 0}")
        if self.tok is not None:
            bits.append(f"  at {where(self.tok)}")
        if self.stack:
            trail = " -> ".join(self.stack[-8:])
            bits.append(f"  while emitting: {trail}")
        return "\n".join(bits)


class DSLSyntaxError(CodegenError):
    pass
