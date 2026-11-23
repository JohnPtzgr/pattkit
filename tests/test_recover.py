from _util import check, diag
from parsekit import Parser, ParseError, TwoPassLexer, LexConfig, OPERATORS, sexp
from clike import make_parser as clike, GRAMMAR as CLIKE_G
from pylike import make_parser as pylike

# ───────── 1. hand-written error patterns (C-like example) ─────────
p = clike()
d = diag(p, "typedef int T;\nT * x")
check("pattern: missing ';' named in message", d.rule == "missing_semi" and "'x'" in d.message, d)
check("pattern: location is the match start", (d.line, d.col) == (2, 1), (d.line, d.col))
check("edit search also proposes the fix",
      any("insert ';' after 'x'" in s.describe() for s in d.suggestions), d.format())

# ───────── 2. edit search (no hand-written patterns) ─────────
FREE = TwoPassLexer(LexConfig(layout=False, space_tokens=True, brackets="lenient"))
G2 = r'''
layer groups:
  group: <"(" inner=+{kind = expr} ")">
layer expr:
  operators expr over DIGITS IDENTIFIER group
    left "+" "-",
    left "*" "/"
'''
p = Parser(G2, "expr", lexer=FREE)
d = diag(p, "(1 + 2")
check("1 edit: missing ')' (rightmost fix first)",
      d.suggestions and d.suggestions[0].describe().startswith("insert ')' after '2'"), d.format())
d = diag(p, "1 + * 2")
check("1 edit: missing operand", d.suggestions and len(d.suggestions[0].edits) == 1, d.format())
d = diag(p, "1 + * 2 *")
check("2 edits found", d.suggestions and len(d.suggestions[0].edits) == 2, d.format())
d = diag(p, "( + * ) ) ) 1 2 3 4")
check("hopeless input: no bogus fix, still a message", not d.suggestions and d.message, d.format())
ast, dg = p.parse_recover("(1 + 2")
check("parse_recover returns repaired AST", ast is not None and dg is not None and sexp(ast) == "(1 + 2)")
ast, dg = p.parse_recover("1 + 2")
check("parse_recover on valid input: no diagnostic", ast is not None and dg is None)
ast, dg = p.parse_recover("1 + `")
check("lexer errors become diagnostics", ast is None and "TOKENIZATION_FAIL" in dg.message, dg)
try:
    Parser(G2, "expr", lexer=FREE, recovery=False).parse("(1 +")
except ParseError as e:
    check("recovery=False gives a plain error", e.diagnostic is None and "cannot reduce" in str(e))

# ───────── 3. layout language: NEWLINE / INDENT / DEDENT from the 2-pass lexer ─────────
p = pylike()
src = """x = 1 + 2
if x
  y = x * 3
  if y
    z = y /// nested
  w = 4 /. block
           comment ./
v = x
"""
ast = p.parse(src)
check("indented program parses", ast.type == "module" and len(ast.fields["body"]) == 3)
inner = ast.fields["body"][1].fields["body"]
check("nested block structure", inner.type == "suite" and
      [b.type for b in inner.fields["body"]] == ["assign", "ifstmt", "assign"])
d = diag(p, "if x\n")
check("recover rule on lexer-produced tokens", d.rule == "empty_block" and "line 1" in d.message, d)
d = diag(p, "x = 1 +\n")
check("missing operand across NEWLINE gets a fix",
      d.suggestions and "insert" in d.suggestions[0].describe(), d.format())
print(d.format())
print("recover: all passed")
