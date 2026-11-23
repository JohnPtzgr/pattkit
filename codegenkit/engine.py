"""The codegen engine: walks the Boundary AST and emits C."""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import dsl
from .errors import CodegenError, node_label, where
from .expr import _SAFE_BUILTINS


_C_KEYWORDS = frozenset("""
auto break case char const continue default do double else enum extern float for
goto if inline int long register restrict return short signed sizeof static struct
switch typedef union unsigned void volatile while _Bool _Complex _Alignas
""".split())

DEFAULT_TYPE_MAP = {
    "i8": "int8_t", "i16": "int16_t", "i32": "int32_t", "i64": "int64_t",
    "u8": "uint8_t", "u16": "uint16_t", "u32": "uint32_t", "u64": "uint64_t",
    "isize": "intptr_t", "usize": "uintptr_t",
    "f32": "float", "f64": "double",
    "bool": "bool", "str": "const char *", "string": "const char *",
    "void": "void", "unit": "void", "None": "void",
}


@dataclass
class CodegenOptions:
    indent: str = "    "
    module: str = "main"
    on_missing: str = "error"          # error | comment | empty | text
    default_type: str = "int"
    type_map: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TYPE_MAP))
    default_includes: tuple = ("<stdint.h>", "<stdbool.h>", "<stddef.h>")
    line_directives: bool = False
    banner: bool = True
    max_depth: int = 400
    mangle_prefix: bool = True


@dataclass
class CodegenResult:
    c_source: str
    h_source: Optional[str]
    includes: List[str]
    symbols: Dict[str, str]


class Emitter:
    """Line-oriented buffer."""

    __slots__ = ("lines", "unit")

    def __init__(self, unit: str = "    "):
        self.lines: List[str] = []
        self.unit = unit

    def write(self, text: str, level: int = 0):
        pad = self.unit * max(level, 0)
        for line in str(text).split("\n"):
            self.lines.append(pad + line if line.strip() else "")

    def text(self) -> str:
        return "\n".join(self.lines)

    def __bool__(self):
        return any(l.strip() for l in self.lines)


class _Frame:
    __slots__ = ("node", "rule", "env", "inline", "buf", "level", "hoists")

    def __init__(self, node, rule, env, inline, unit):
        self.node = node
        self.rule = rule
        self.env = env
        self.inline = inline
        self.buf = Emitter(unit)
        self.level = 0
        self.hoists: List[tuple] = []


def _is_tok(x) -> bool:
    return hasattr(x, "text") and hasattr(x, "kids") and hasattr(x, "type")


def _tok_of(node):
    tok = getattr(node, "tok", None)
    if tok is not None:
        return tok
    payload = getattr(node, "payload", None)
    return payload if _is_tok(payload) else None


def node_field(node, name, default=None):
    """Resolve a pattern parameter against a node."""
    if isinstance(node, dict):
        return node.get(name, default)
    if hasattr(node, name) and not name.startswith("_"):
        return getattr(node, name)
    payload = getattr(node, "payload", None)
    if isinstance(payload, dict) and name in payload:
        return payload[name]
    for tok in (getattr(node, "tok", None), payload if _is_tok(payload) else None):
        fields = getattr(tok, "fields", None)
        if isinstance(fields, dict) and name in fields:
            return fields[name]
    return default


class Codegen:
    def __init__(self, options: Optional[CodegenOptions] = None, builder=None):
        self.opts = options or CodegenOptions()
        self.builder = builder
        self._rules: Dict[str, List[dsl.Rule]] = {}
        self._stack: List[str] = []
        self._helpers: Dict[str, Any] = {}
        self._reset()
        self._install_helpers()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_source(self, source: str, origin: str = "<codegen>") -> "Codegen":
        for rule in dsl.parse(source, origin):
            self._rules.setdefault(rule.key, []).append(rule)
        return self

    def load_file(self, path) -> "Codegen":
        p = Path(path)
        return self.load_source(p.read_text(encoding="utf-8"), str(p))

    @property
    def rules(self) -> Dict[str, List[dsl.Rule]]:
        return self._rules

    # ------------------------------------------------------------------
    # File state
    # ------------------------------------------------------------------

    def _reset(self):
        self.body = Emitter(self.opts.indent)
        self.header = Emitter(self.opts.indent)
        self.top = Emitter(self.opts.indent)
        self.includes: List[str] = []
        self._include_seen = set()
        self._symbols: Dict[str, str] = {}
        self._mangled: Dict[tuple, str] = {}
        self._counter = 0
        self._parent: Optional[_Frame] = None

    def _fresh(self, prefix="tmp") -> str:
        self._counter += 1
        return f"__{prefix}{self._counter}"

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _keys(self, node) -> List[str]:
        keys = []
        tag = getattr(node, "tag", None)
        if tag:
            keys.append(tag)
        for cls in type(node).__mro__:
            if cls is object:
                break
            keys.append(cls.__name__)
        keys.append("*")
        return keys

    def _select(self, node):
        for key in self._keys(node):
            for rule in reversed(self._rules.get(key, ())):
                env = self._base_env(node)
                env["node"] = node
                for p in rule.params:
                    env[p] = node_field(node, p)
                if rule.guard is not None and not self._eval(rule.guard, env, rule, node):
                    continue
                return rule, env
        return None, None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _eval(self, code, env, rule, node):
        try:
            return eval(code, env)
        except CodegenError:
            raise
        except Exception as e:
            raise CodegenError(
                f"{type(e).__name__}: {e}",
                tok=_tok_of(node),
                origin=rule.origin if rule else None,
                lineno=rule.lineno if rule else None,
                stack=self._stack,
            ) from None

    def _base_env(self, node):
        env = {"__builtins__": dict(_SAFE_BUILTINS)}
        env.update(self._helpers)
        env["node"] = node
        env["tok"] = _tok_of(node)
        env["tag"] = getattr(node, "tag", None) or type(node).__name__
        env["opts"] = self.opts
        return env

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def generate(self, node, inline: bool = False) -> str:
        return self._render_node(node, inline)

    def _render_node(self, node, inline: bool) -> str:
        if node is None:
            return ""
        if len(self._stack) > self.opts.max_depth:
            raise CodegenError(
                "codegen recursion limit exceeded (cyclic patterns?)",
                tok=_tok_of(node), stack=self._stack,
            )

        rule, env = self._select(node)
        if rule is None:
            return self._missing(node)

        frame = _Frame(node, rule, env, inline, self.opts.indent)
        self._stack.append(f"{node_label(node)} @{rule.where}")
        parent, self._parent = self._parent, frame
        try:
            if self.opts.line_directives:
                self._exec(dsl.LineStmt(rule.lineno), frame)
            self._exec_block(rule.body, frame)
            if not inline:
                self._flush(frame)
                if frame.hoists:
                    for text, lvl in frame.hoists:
                        frame.buf.write(text, frame.level + lvl)
                    frame.hoists.clear()
            elif frame.hoists and parent is not None:
                parent.hoists.extend(frame.hoists)
        finally:
            self._parent = parent
            self._stack.pop()
        return frame.buf.text()

    def _missing(self, node) -> str:
        policy = self.opts.on_missing
        tok = _tok_of(node)
        if policy == "error":
            tag = getattr(node, "tag", None)
            raise CodegenError(
                f"no codegen pattern for {node_label(node)}"
                + (f" (tag {tag!r})" if tag else ""),
                tok=tok, stack=self._stack,
            )
        if policy == "comment":
            return f"/* unhandled {node_label(node)} at {where(tok)} */"
        if policy == "text":
            return getattr(tok, "text", "") or ""
        return ""

    def _render_value(self, value, inline: bool, sep: str) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, (list, tuple)):
            items = [v for v in value if v is not None]
            return sep.join(self._render_value(v, inline, ", ") for v in items)
        if _is_tok(value):
            if self.builder is not None:
                return self._render_node(self.builder.build(value), inline)
            return getattr(value, "text", "") or ""
        return self._render_node(value, inline)

    def _eval_template(self, template: dsl.Template, frame: _Frame) -> str:
        passthrough = template.passthrough
        out = []
        for part in template.parts:
            if isinstance(part, str):
                out.append(part)
                continue
            value = self._eval(part.code, frame.env, frame.rule, frame.node)
            out.append(self._render_value(
                value,
                inline=not passthrough,
                sep="\n" if passthrough else ", ",
            ))
        return "".join(out)

    def _flush(self, frame: _Frame):
        if frame.inline or not frame.hoists:
            return
        for text, lvl in frame.hoists:
            frame.buf.write(text, frame.level + lvl)
        frame.hoists.clear()

    # ------------------------------------------------------------------
    # Statement execution
    # ------------------------------------------------------------------

    def _exec_block(self, body, frame: _Frame):
        for stmt in body:
            self._exec(stmt, frame)

    def _exec(self, stmt, frame: _Frame):
        if isinstance(stmt, dsl.Emit):
            text = self._eval_template(stmt.template, frame)
            ch = stmt.channel
            if ch == "emit":
                self._flush(frame)
                frame.buf.write(text, frame.level)
            elif ch == "local":
                frame.hoists.append((text, frame.level))
            elif ch == "header":
                self.header.write(text, 0)
            elif ch == "top":
                self.top.write(text, 0)
            elif ch == "include":
                t = text.strip()
                if t and t not in self._include_seen:
                    self._include_seen.add(t)
                    self.includes.append(t)
            return

        if isinstance(stmt, dsl.Indent):
            frame.level += 1
            return
        if isinstance(stmt, dsl.Dedent):
            frame.level = max(0, frame.level - 1)
            return
        if isinstance(stmt, dsl.Pass):
            return

        if isinstance(stmt, dsl.Let):
            frame.env[stmt.name] = self._eval(stmt.code, frame.env, frame.rule, frame.node)
            return

        if isinstance(stmt, dsl.For):
            seq = self._eval(stmt.code, frame.env, frame.rule, frame.node)
            if seq is None:
                seq = ()
            elif not isinstance(seq, (list, tuple)):
                seq = (seq,)
            for item in seq:
                frame.env[stmt.var] = item
                self._exec_block(stmt.body, frame)
            return

        if isinstance(stmt, dsl.If):
            for cond, body in stmt.branches:
                if cond is None or self._eval(cond, frame.env, frame.rule, frame.node):
                    self._exec_block(body, frame)
                    return
            return

        if isinstance(stmt, dsl.LineStmt):
            tok = frame.env.get("tok")
            if tok is not None:
                line = getattr(tok, "line", 0) or 0
                file = getattr(tok, "file", None) or getattr(tok, "path", None) or "<source>"
                frame.buf.write(f'#line {line} "{file}"', frame.level)
            return

        if isinstance(stmt, dsl.ErrorStmt):
            raise CodegenError(
                self._eval_template(stmt.template, frame),
                tok=frame.env.get("tok"),
                origin=frame.rule.origin,
                lineno=stmt.lineno,
                stack=self._stack,
            )

        raise CodegenError(f"unknown DSL statement {type(stmt).__name__}")

    # ------------------------------------------------------------------
    # Helpers available inside patterns
    # ------------------------------------------------------------------

    def _install_helpers(self):
        H = self._helpers

        def ident(name):
            s = re.sub(r"\W", "_", str(name))
            if not s or s[0].isdigit():
                s = "_" + s
            if s in _C_KEYWORDS:
                s += "_"
            return s

        def mangle(name, kind=None):
            key = (kind or "", str(name))
            if key in self._mangled:
                return self._mangled[key]
            base = ident(name)
            if self.opts.mangle_prefix:
                base = f"{ident(self.opts.module)}__{base}"
            sym, n = base, 1
            while sym in self._symbols:
                n += 1
                sym = f"{base}_{n}"
            self._symbols[sym] = str(name)
            self._mangled[key] = sym
            return sym

        def render(value, sep=", "):
            return self._render_value(value, inline=True, sep=sep)

        def ctype(t, default=None):
            if t is None:
                return default or self.opts.default_type
            if isinstance(t, str):
                return self.opts.type_map.get(t, t)
            name = getattr(t, "name", None)
            if name:
                return self.opts.type_map.get(name, name)
            text = render(t).strip()
            return self.opts.type_map.get(text, text)

        def join(items, sep=", "):
            return self._render_value(items, inline=True, sep=sep)

        def trim(value, chars=";"):
            return render(value).strip().rstrip(chars).strip()

        def cstr(s):
            body = (
                str(s)
                .replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\t", "\\t")
                .replace("\r", "\\r")
            )
            return f'"{body}"'

        H.update(
            ident=ident, mangle=mangle, sym=self._fresh,
            render=render, ctype=ctype, join=join, trim=trim, cstr=cstr,
            field=node_field,
            text=lambda n: getattr(_tok_of(n) or n, "text", "") or "",
            kids=lambda n: [
                k for k in getattr(_tok_of(n), "kids", ()) or ()
                if getattr(k, "type", None) != "_"
            ],
            line_of=lambda n: getattr(_tok_of(n), "line", 0),
            file_of=lambda n: (
                getattr(_tok_of(n), "file", None) or getattr(_tok_of(n), "path", "")
            ),
            tag_of=lambda n: getattr(n, "tag", None) or type(n).__name__,
        )

    # ------------------------------------------------------------------
    # Module assembly
    # ------------------------------------------------------------------

    def generate_module(
        self,
        nodes,
        module_name: Optional[str] = None,
        header_name: Optional[str] = None,
    ) -> CodegenResult:
        self._reset()
        if module_name:
            self.opts.module = module_name
        if isinstance(nodes, (list, tuple)):
            top_nodes = list(nodes)
        else:
            top_nodes = [nodes]

        for node in top_nodes:
            if node is None:
                continue
            text = self._render_node(node, inline=False)
            if text.strip():
                self.body.write(text, 0)

        includes = list(dict.fromkeys([*self.opts.default_includes, *self.includes]))
        parts = []
        if self.opts.banner:
            parts.append(
                f"/* Generated by PattKit — module '{self.opts.module}'. Do not edit. */"
            )
        if includes:
            parts.append("\n".join(f"#include {i}" for i in includes))
        if self.top:
            parts.append("/* ---- file scope ---- */\n" + self.top.text())
        if self.header:
            parts.append("/* ---- declarations ---- */\n" + self.header.text())
        if self.body:
            parts.append("/* ---- definitions ---- */\n" + self.body.text())
        c_source = "\n\n".join(parts) + "\n"

        h_source = None
        if header_name:
            guard = re.sub(r"\W", "_", header_name).upper() + "_"
            h_parts = [f"#ifndef {guard}", f"#define {guard}"]
            if includes:
                h_parts.append("\n".join(f"#include {i}" for i in includes))
            if self.header:
                h_parts.append(self.header.text())
            h_parts.append(f"#endif /* {guard} */")
            h_source = "\n\n".join(h_parts) + "\n"

        return CodegenResult(c_source, h_source, includes, dict(self._symbols))
