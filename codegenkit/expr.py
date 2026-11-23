"""Sandboxed expression compilation for the codegen DSL."""
import ast


_SAFE_BUILTINS = {
    "len": len, "str": str, "int": int, "float": float, "bool": bool,
    "list": list, "tuple": tuple, "dict": dict, "range": range,
    "enumerate": enumerate, "zip": zip, "min": min, "max": max,
    "sorted": sorted, "any": any, "all": all, "repr": repr, "abs": abs,
    "None": None, "True": True, "False": False,
}

_DENIED = (
    ast.Lambda, ast.Await, ast.Yield, ast.YieldFrom,
    ast.NamedExpr, ast.Starred,
)


def _validate(tree, source, origin, lineno):
    for n in ast.walk(tree):
        if isinstance(n, _DENIED):
            raise Exception(f"{type(n).__name__} not allowed: {source!r}")
        if isinstance(n, ast.Attribute) and n.attr.startswith("_"):
            raise Exception(f"private attr not allowed: .{n.attr}")


def compile_expr(source: str, origin: str = "<codegen>", lineno: int = 0):
    source = source.strip()
    if not source:
        raise Exception("empty expression")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as e:
        raise Exception(f"bad expression {source!r}: {e.msg}")
    _validate(tree, source, origin, lineno)
    return compile(tree, f"<codegen:{origin}:{lineno}>", "eval")


def split_pipes(source: str) -> str:
    """Rewrite `expr | f(a, b)` into `f((expr), a, b)` (left-folded)."""
    parts, depth, quote, buf = [], 0, None, []
    i = 0
    while i < len(source):
        ch = source[i]
        if quote:
            if ch == "\\":
                if i + 1 < len(source):
                    buf.append(source[i + 1])
                    i += 2
                    continue
            elif ch == quote:
                quote = None
            buf.append(ch)
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "|" and depth == 0 and (i + 1 >= len(source) or source[i + 1] != "|"):
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))

    expr = parts[0].strip()
    for filt in parts[1:]:
        filt = filt.strip()
        if "(" in filt and filt.rstrip().endswith(")"):
            head, rest = filt.split("(", 1)
            args = rest.rstrip()[:-1].strip()
            expr = f"{head.strip()}(({expr})" + (f", {args})" if args else ")")
        else:
            expr = f"{filt}(({expr}))"
    return expr
