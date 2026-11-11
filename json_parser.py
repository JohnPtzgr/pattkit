"""JSON: ranges over pairs `<...>`, separator lists `% ","`, label patterns `value/string`."""
import json

from parsekit import Parser, TwoPassLexer, LexConfig

GRAMMAR = r'''
layer scalars:
  value/string: STRING
  value/number: "-"? DIGITS
  value/bool:   "true" | "false"
  value/null:   "null"
layer containers:
  pair:         key=value/string ":" val=value
  value/array:  <"[" ((items=value) % ",")? "]">
  value/object: <"{" ((members=pair) % ",")? "}">
'''


def make_parser():
    return Parser(GRAMMAR, "value", lexer=TwoPassLexer(LexConfig(layout=False, space_tokens=True, brackets="lenient")))


def _list(x):
    return [] if x is None else (x if isinstance(x, list) else [x])


def to_python(n):
    """AST -> Python value (numbers: plain integers / decimals only)."""
    L = n.label
    if L == "string":
        return json.loads(n.text)
    if L == "number":
        return float(n.text) if "." in n.text else int(n.text)
    if L == "bool":
        return n.text == "true"
    if L == "null":
        return None
    if L == "array":
        return [to_python(v) for v in _list(n.fields.get("items"))]
    if L == "object":
        return {to_python(p.fields["key"]): to_python(p.fields["val"])
                for p in _list(n.fields.get("members"))}
    raise ValueError(f"unexpected node {n!r}")
