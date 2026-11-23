"""Parser for the .cgen pattern DSL.

    BVarDecl(name, type_hint, value):
        emit "${type_hint} ${name} = ${value};"

    "light:shared_let"(name, value) when value != None:
        emit "..."
"""
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from .errors import DSLSyntaxError
from .expr import compile_expr, split_pipes


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_UNESCAPE = {
    "n": "\n", "t": "\t", "r": "\r", "\\": "\\",
    '"': '"', "'": "'", "0": "\0",
}


@dataclass
class Interp:
    source: str
    code: Any


@dataclass
class Template:
    parts: List[Any] = field(default_factory=list)  # str | Interp

    @property
    def passthrough(self) -> bool:
        """True for `emit "${stmt}"` — a bare, lone interpolation."""
        return len(self.parts) == 1 and isinstance(self.parts[0], Interp)


def _match_brace(s: str, start: int) -> int:
    depth, quote, i = 0, None, start
    while i < len(s):
        ch = s[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced '${' in template")


def parse_template(raw: str, origin: str, lineno: int) -> Template:
    parts, buf, i = [], [], 0
    while i < len(raw):
        ch = raw[i]
        if ch == "\\" and i + 1 < len(raw):
            nxt = raw[i + 1]
            buf.append("$" if nxt == "$" else _UNESCAPE.get(nxt, "\\" + nxt))
            i += 2
            continue
        if ch == "$" and i + 1 < len(raw) and raw[i + 1] == "{":
            try:
                end = _match_brace(raw, i + 1)
            except ValueError as e:
                raise DSLSyntaxError(str(e), origin=origin, lineno=lineno) from None
            if buf:
                parts.append("".join(buf))
                buf = []
            src = split_pipes(raw[i + 2:end])
            parts.append(Interp(src, compile_expr(src, origin, lineno)))
            i = end + 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf))
    return Template(parts)


# ---------------------------------------------------------------------------
# Statements / rules
# ---------------------------------------------------------------------------

@dataclass
class Emit:
    channel: str
    template: Template
    lineno: int = 0


@dataclass
class Indent:
    lineno: int = 0


@dataclass
class Dedent:
    lineno: int = 0


@dataclass
class Let:
    name: str
    code: Any
    lineno: int = 0


@dataclass
class For:
    var: str
    code: Any
    body: List[Any]
    lineno: int = 0


@dataclass
class If:
    branches: List[Tuple[Optional[Any], List[Any]]]
    lineno: int = 0


@dataclass
class ErrorStmt:
    template: Template
    lineno: int = 0


@dataclass
class LineStmt:
    lineno: int = 0


@dataclass
class Pass:
    lineno: int = 0


@dataclass
class Rule:
    key: str
    params: List[str]
    guard: Optional[Any]
    body: List[Any]
    origin: str
    lineno: int

    @property
    def where(self):
        return f"{self.origin}:{self.lineno}"


# ---------------------------------------------------------------------------
# Line scanning
# ---------------------------------------------------------------------------

@dataclass
class _Line:
    indent: int
    text: str
    lineno: int


def _strip_comment(s: str) -> str:
    quote, i = None, 0
    while i < len(s):
        ch = s[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            return s[:i]
        i += 1
    return s


def _logical_lines(source: str, origin: str) -> List[_Line]:
    out = []
    for n, raw in enumerate(source.splitlines(), 1):
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise DSLSyntaxError(
                "tabs in indentation; use spaces", origin=origin, lineno=n
            )
        text = _strip_comment(raw).rstrip()
        if not text.strip():
            continue
        out.append(_Line(len(text) - len(text.lstrip()), text.strip(), n))
    return out


def _string_arg(text: str, origin: str, lineno: int) -> str:
    if not text or text[0] not in "\"'":
        raise DSLSyntaxError(
            f"expected a quoted template, got {text!r}",
            origin=origin, lineno=lineno,
        )
    quote, i, out = text[0], 1, []
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            out.append(text[i:i + 2])
            i += 2
            continue
        if ch == quote:
            if text[i + 1:].strip():
                raise DSLSyntaxError(
                    f"unexpected trailing text: {text[i + 1:].strip()!r}",
                    origin=origin, lineno=lineno,
                )
            return "".join(out)
        out.append(ch)
        i += 1
    raise DSLSyntaxError("unterminated template string", origin=origin, lineno=lineno)


_HEADER = re.compile(
    r"""^(?:"(?P<qt>[^"]+)"|'(?P<qt2>[^']+)'|(?P<name>[A-Za-z_][\w.]*|\*))
         \s*(?:\((?P<params>[^)]*)\))?
         \s*(?:when\s+(?P<guard>.+?))?
         \s*:$""",
    re.VERBOSE,
)

_CHANNELS = {"emit", "header", "include", "top", "local", "pre"}


# ---------------------------------------------------------------------------
# Block parsing
# ---------------------------------------------------------------------------

def _parse_block(lines, i, parent_indent, origin) -> Tuple[List[Any], int]:
    if i >= len(lines) or lines[i].indent <= parent_indent:
        raise DSLSyntaxError(
            "expected an indented block", origin=origin,
            lineno=lines[i - 1].lineno if i else 0,
        )
    level = lines[i].indent
    body: List[Any] = []
    while i < len(lines) and lines[i].indent >= level:
        ln = lines[i]
        if ln.indent > level:
            raise DSLSyntaxError(
                "unexpected indentation", origin=origin, lineno=ln.lineno
            )
        text = ln.text

        if text.startswith(("elif ", "else:")):
            if not body or not isinstance(body[-1], If):
                raise DSLSyntaxError(
                    "'elif'/'else' without 'if'", origin=origin, lineno=ln.lineno
                )
            iff = body[-1]
            if text == "else:":
                if any(c is None for c, _ in iff.branches):
                    raise DSLSyntaxError(
                        "duplicate 'else'", origin=origin, lineno=ln.lineno
                    )
                sub, i = _parse_block(lines, i + 1, level, origin)
                iff.branches.append((None, sub))
            else:
                cond = compile_expr(text[5:].rstrip(":"), origin, ln.lineno)
                sub, i = _parse_block(lines, i + 1, level, origin)
                iff.branches.append((cond, sub))
            continue

        stmt, i = _parse_stmt(lines, i, level, origin)
        if stmt is not None:
            body.append(stmt)
    return body, i


def _parse_stmt(lines, i, level, origin):
    ln = lines[i]
    text, n = ln.text, ln.lineno

    head = text.split(" ", 1)[0].rstrip(":")
    rest = text[len(head):].lstrip()
    if rest.startswith(":"):
        rest = rest[1:].lstrip()

    if head in _CHANNELS:
        channel = "local" if head == "pre" else head
        return Emit(
            channel,
            parse_template(_string_arg(rest, origin, n), origin, n),
            n,
        ), i + 1

    if text == "indent":
        return Indent(n), i + 1
    if text == "dedent":
        return Dedent(n), i + 1
    if text == "pass":
        return Pass(n), i + 1
    if text == "line":
        return LineStmt(n), i + 1

    if head == "error":
        return ErrorStmt(
            parse_template(_string_arg(rest, origin, n), origin, n), n
        ), i + 1

    if head == "let":
        m = re.match(r"^([A-Za-z_]\w*)\s*=\s*(.+)$", rest)
        if not m:
            raise DSLSyntaxError(
                "expected `let name = expression`", origin=origin, lineno=n
            )
        return Let(m.group(1), compile_expr(m.group(2), origin, n), n), i + 1

    if head == "for":
        m = re.match(r"^([A-Za-z_]\w*)\s+in\s+(.+):$", rest)
        if not m:
            raise DSLSyntaxError(
                "expected `for name in expression:`", origin=origin, lineno=n
            )
        body, j = _parse_block(lines, i + 1, level, origin)
        return For(
            m.group(1), compile_expr(m.group(2), origin, n), body, n
        ), j

    if head == "if":
        if not rest.endswith(":"):
            raise DSLSyntaxError(
                "expected `if expression:`", origin=origin, lineno=n
            )
        cond = compile_expr(rest[:-1], origin, n)
        body, j = _parse_block(lines, i + 1, level, origin)
        return If([(cond, body)], n), j

    raise DSLSyntaxError(f"unknown statement: {text!r}", origin=origin, lineno=n)


def parse(source: str, origin: str = "<codegen>") -> List[Rule]:
    lines = _logical_lines(source, origin)
    rules, i = [], 0
    while i < len(lines):
        ln = lines[i]
        m = _HEADER.match(ln.text)
        if not m:
            raise DSLSyntaxError(
                f"expected a pattern header like `BVarDecl(name, value):`, got {ln.text!r}",
                origin=origin, lineno=ln.lineno,
            )
        key = m.group("qt") or m.group("qt2") or m.group("name")
        params = [
            p.strip() for p in (m.group("params") or "").split(",") if p.strip()
        ]
        for p in params:
            if not re.fullmatch(r"[A-Za-z_]\w*", p):
                raise DSLSyntaxError(
                    f"bad parameter name {p!r}", origin=origin, lineno=ln.lineno
                )
        guard = (
            compile_expr(m.group("guard"), origin, ln.lineno)
            if m.group("guard") else None
        )
        body, i = _parse_block(lines, i + 1, ln.indent, origin)
        rules.append(Rule(key, params, guard, body, origin, ln.lineno))
    return rules
