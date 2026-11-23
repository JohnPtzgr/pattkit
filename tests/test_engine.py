from _util import check, fails
from parsekit import Parser, TwoPassLexer, LexConfig, OPERATORS, sexp, dump
from clike import make_parser as clike

FREE = TwoPassLexer(LexConfig(layout=False, space_tokens=True, brackets="lenient",
                              operators=OPERATORS + [("**", "POW")]))

# ── counters: a^n b^n c^n ──
p = Parser(r'''
layer main:
  abc: xs="a"{n} ys="b"{n} zs="c"{n}
''', "abc", lexer=FREE)
ast = p.parse("a a a b b b c c c")
check("a^3 b^3 c^3 -> n=3", ast.params == {"n": 3} and len(ast.fields["xs"]) == 3)
check("a a b b c rejected", fails(p, "a a b b c") is not None)
check("a a b b c c accepted", p.parse("a a b b c c").params == {"n": 2})

# ── expressions: operator table, `_` significance, PSP group ──
G2 = r'''
layer groups:
  group: <"(" inner=+{kind = expr} ")">
layer calls:
  call: callee=IDENTIFIER !_ args=group
layer expr:
  operators expr over DIGITS IDENTIFIER group call
    right "=",
    chain "<" ">" "==",
    left "+" "-",
    left "*" "/",
    prefix "-",
    right "**"
'''
p = Parser(G2, "expr", lexer=FREE)
for src, want in [
    ("1 + 2 * 3 - 4", "((1 + (2 * 3)) - 4)"),
    ("2 ** 3 ** 2", "(2 ** (3 ** 2))"),
    ("-a ** b", "(-(a ** b))"),
    ("-a * b", "((-a) * b)"),
    ("(1 + 2) * 3", "((1 + 2) * 3)"),
    ("a = b = 1 + 2", "(a = (b = (1 + 2)))"),
    ("1 + /// comment\n 2", "(1 + 2)"),
]:
    got = sexp(p.parse(src))
    check(f"{src!r} -> {got}", got == want)
check("f(x) call: adjacent", "call" in sexp(p.parse("f(x) * 2")))
check("f (x): whitespace is significant", fails(p, "f (x)") is not None)
n = p.parse("a < b < c")
check("chain: one node, lists", n.label == "chain" and len(n.fields["operands"]) == 3
      and [o.text for o in n.fields["ops"]] == ["<", "<"])
n = p.parse("1 + a < b == c")      # '<' and '==' share one chain level
check("chain binds weaker than +", n.label == "chain" and n.fields["operands"][0].label == "binary")

# ── juxtaposition (ML-style application) ──
ML = Parser(r'''
layer expr:
  operators expr over IDENTIFIER DIGITS
    left "+",
    apply
''', "expr", lexer=FREE)
n = ML.parse("f x + g y z")
check("apply: (f x) + ((g y) z)", n.label == "binary" and n.fields["lhs"].label == "apply"
      and n.fields["rhs"].fields["fn"].label == "apply")
check("apply does not cross a line break", fails(ML, "f\nx") is not None)

# ── labels in patterns, separator lists, multiple openers ──
P = Parser(r'''
layer expr:
  operators expr over IDENTIFIER DIGITS
    left "+"
layer stmts:
  stmt/bin:   e=expr/binary ";"
  stmt/plain: e=(IDENTIFIER | DIGITS) ";"
layer top:
  prog: items=stmt+
''', "prog", lexer=FREE)
t = P.parse("a + b ; c ;")
check("expr/binary label pattern", [i.label for i in t.fields["items"]] == ["bin", "plain"])

L = Parser(r'''
layer main:
  lst: "[" ((x=DIGITS) % ",") "]"
''', "lst", lexer=FREE)
check("X % S captures each item", len(L.parse("[1, 2 , 3]").fields["x"]) == 3)
check("X % S needs at least one", fails(L, "[]") is not None)

B = Parser(r'''
layer main:
  block: <("do" | "begin") x=* "end">
''', "block", lexer=FREE)
n = B.parse("do a begin b end end")
check("alternation of openers + nesting", n.type == "block" and n.fields["x"][1].type == "block")
check("unbalanced block rejected", fails(B, "do a end end") is not None)

# ── unknown class + hand-written recover rule ──
U = Parser(r'''
names IDENTIFIER
unknown UNKNOWN
layer decls:
  var: "let" name=IDENTIFIER as VAR ";"
layer uses:
  use: v=VAR ";"
layer top:
  prog: items=(var | use)+
recover:
  undeclared: u=UNKNOWN => "name '{u}' is not declared (line {line})"
''', "prog", lexer=FREE)
check("declared name parses", U.parse("let x ; x ;").type == "prog")
from _util import diag
d = diag(U, "let x ; y ;")
check("undeclared name -> UNKNOWN -> message", d.rule == "undeclared" and "'y'" in d.message, d)

# ── name table: scopes are captures; inner declaration shadows, outer is untouched ──
# NOTE: the scope-introducing block must come in an EARLIER layer than the declarations,
# otherwise the declaration layer would see through the still-unreduced braces.
S = Parser(r'''
names IDENTIFIER
layer blocks:
  scope: <"{" body=* "}">
layer decls:
  typedef: "typedef" name=IDENTIFIER as TYPENAME ";"
layer vars:
  vardecl: ty=TYPENAME "*"? name=IDENTIFIER ";"
layer expr:
  operators expr over IDENTIFIER
    left "*"
layer stmts:
  exprstmt: e=expr ";"
layer top:
  prog: items=(typedef | scope | vardecl | exprstmt)+
''', "prog", lexer=FREE)
t = S.parse("typedef T ; T * a ; { typedef a ; a * b ; } a * c ;")
check("outer typedef visible; inner typedef does not leak",
      [i.type for i in t.fields["items"]] == ["typedef", "vardecl", "scope", "exprstmt"],
      [i.type for i in t.fields["items"]])
# the block body ran through ALL layers as an isolated run, so it became a `prog` itself
inner = t.fields["items"][2].fields["body"]
check("inside the block `a` is a type", inner.type == "prog" and
      [k.type for k in inner.fields["items"]] == ["typedef", "vardecl"], inner)

# ── lint ──
w = Parser(r'''
layer a:
  r1: Foo "**" IDENTIFIER
  r2: IDENTIFIER
layer b:
  top: r1
''', "top", lexer=TwoPassLexer(LexConfig(layout=False, space_tokens=True, brackets="lenient"))).warnings
check("lint: unknown name", any("'Foo'" in x for x in w), w)
check("lint: literal the lexer cannot produce", any("'**'" in x for x in w), w)
check("lint: unused rule", any("'r2'" in x for x in w), w)
check("lint: clean grammar has no warnings", clike().warnings == [], clike().warnings)

# ── C-like example: typedef / generics through the name table ──
p = clike()
src = """
typedef int T;
template List;
T * x;          /// declaration: T is a type
a * x;          /// multiplication: a is not
List<int> l;
List<List<int>> ll;
a < b;
a < b < c;
"""
prog = p.parse(src)
kinds = [(i.type, i.label) for i in prog.fields["items"]]
check("C-like item kinds", kinds == [("decl", "typedef"), ("decl", "template"), ("vardecl", None),
      ("exprstmt", None), ("vardecl", None), ("vardecl", None), ("exprstmt", None), ("exprstmt", None)], kinds)
ty = prog.fields["items"][5].fields["ty"]
check("nested generic shape", ty.label == "generic" and ty.fields["args"].label == "generic")
q = p.parse("typedef int T;\nQ * y;\n").fields["items"][1]
check("undeclared Q * y is a multiplication", q.type == "exprstmt")
q = p.parse("typedef int Q;\nQ * y;\n").fields["items"][1]
check("declared Q * y is a declaration", q.type == "vardecl")
check("garbage rejected", fails(p, "T * ;") is not None)
d = dump(p.parse("typedef int T;\nT * x;"))
check("dump is JSON-friendly", d["type"] == "program" and d["fields"]["items"][1]["type"] == "vardecl")
print("engine: all passed")
