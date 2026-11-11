"""Parser facade: two-pass lexer + layered engine + recovery layer + grammar linter."""
from .engine import Engine, Grammar, Scope
from .lexer import LexConfig, LexError, TwoPassLexer
from .recover import Diagnostic


class ParseError(SyntaxError):
    """Parse failure; `.diagnostic` carries the recovery layer's report."""

    def __init__(self, msg, diagnostic=None):
        super().__init__(msg)
        self.diagnostic = diagnostic


class Parser:
    """
    Parser(grammar, start, lexer=None, builtins=None, recovery=True, lint=True)

    lexer     any object with .tokenize(src) -> [Tok..., EOF]; defaults to the two-pass
              lexer with `_` gap tokens (so grammars can use `_` / `!_`).
    builtins  {text: class} preloaded into the root name table (e.g. {"int": "TYPENAME"}).
    """

    def __init__(self, grammar, start, lexer=None, builtins=None, recovery=True,
                 lint=True, **recover_opts):
        self.lexer = lexer or TwoPassLexer(LexConfig(space_tokens=True, brackets="lenient"))
        self.g, self.start = Grammar(grammar), start
        self.builtins = builtins or {}
        self.recovery = None
        if recovery:
            from .recover import Recovery
            self.recovery = Recovery(self, **recover_opts)
        self.warnings = self.lint() if lint else []

    # ---------------- running ----------------
    def root_scope(self):
        root = Scope()
        root.table.update(self.builtins)
        return root

    def reduce(self, toks):
        """Run all layers. Returns (remaining tokens, AST or None)."""
        eng = Engine(self.g)
        res = eng.run(list(toks), self.root_scope())
        sig = [t for t in res if t.type != "_"]
        ast = sig[0] if len(sig) == 1 and eng.kind_ok(sig[0], self.start) else None
        return res, ast

    def parse(self, src):
        """AST, or ParseError (with .diagnostic). Lexer failures raise LexError."""
        toks = self.lexer.tokenize(src)[:-1]
        res, ast = self.reduce(toks)
        if ast is not None:
            return ast
        if self.recovery is None:
            sig = [t for t in res if t.type != "_"]
            raise ParseError(f"cannot reduce to {self.start!r}; remaining: "
                             + " ".join(t.type for t in sig[:14]))
        diag = self.recovery.diagnose(toks, res, src)
        raise ParseError(diag.format(), diag)

    def parse_recover(self, src):
        """(ast, diagnostic). If a repair was found, the AST is built from the repaired tokens."""
        try:
            toks = self.lexer.tokenize(src)[:-1]
        except LexError as e:
            return None, Diagnostic(e.line, e.col, f"{e.code}: {e.detail}".rstrip(": "))
        res, ast = self.reduce(toks)
        if ast is not None:
            return ast, None
        diag = self.recovery.diagnose(toks, res, src)
        for sg in diag.suggestions:
            _, fixed = self.reduce(sg.tokens)
            if fixed is not None:
                return fixed, diag
        return None, diag

    # ---------------- static checks ----------------
    def lint(self):
        """Grammar warnings: unknown token/rule names, literals the lexer can never produce
        as one token, wildcard targets that nothing produces, rules that are never used."""
        g, lx, warns = self.g, self.lexer, []
        produced = set(lx.token_types()) if hasattr(lx, "token_types") else None
        rule_types, classes, consumed, lits = set(), set(), {self.start}, {}
        if g.unknown:
            classes.add(g.unknown)
        classes |= set(self.builtins.values())

        def collect(n):
            if isinstance(n, tuple):
                if n and n[0] == "cap" and n[3]:
                    classes.add(n[3])
                for x in n[1:]:
                    collect(x)
            elif isinstance(n, list):
                for x in n:
                    collect(x)

        for layer in g.layers:
            for kind, obj in layer.steps:
                if kind == "rules":
                    for r in obj:
                        rule_types.add(r.type)
                        collect(r.pat)
                else:
                    rule_types.add(obj.result)
        known = None if produced is None else produced | rule_types | classes

        def visit(n, where):
            t = n[0]
            if t == "tok":
                consumed.add(n[1])
                if known is not None and n[1] not in known:
                    warns.append(f"{where}: unknown token or rule {n[1]!r}")
            elif t == "lit":
                lits.setdefault(n[1], where)
            elif t in ("seq", "alt"):
                for x in n[1]:
                    visit(x, where)
            elif t in ("rep", "cnt", "look"):
                visit(n[1], where)
            elif t == "cap":
                visit(n[2], where)
            elif t == "any" and n[3]:
                consumed.add(n[3])
                if known is not None and n[3] not in known:
                    warns.append(f"{where}: wildcard target {n[3]!r} is never produced")
            elif t == "bal":
                for x in n[1:]:
                    visit(x, where)

        for layer in g.layers:
            for kind, obj in layer.steps:
                if kind == "rules":
                    for r in obj:
                        visit(r.pat, f"layer {layer.name}, rule {r.type}")
                else:
                    consumed |= obj.operands
                    for _, ops in obj.levels:
                        for o in ops:
                            lits.setdefault(o, f"layer {layer.name}, operators {obj.result}")
        for e in g.errors:
            visit(e.pat, f"recover rule {e.name}")
        if hasattr(lx, "fragment"):
            for text, where in lits.items():
                try:
                    ts = lx.fragment(text)
                except SyntaxError:
                    ts = None
                if ts is None or len(ts) != 1 or ts[0].text != text:
                    got = "nothing lexable" if ts is None else [t.text for t in ts]
                    warns.append(f"{where}: literal {text!r} can never match; "
                                 f"the lexer produces {got} for it")
        for name in sorted(rule_types - consumed):
            warns.append(f"rule {name!r} is produced but never used")
        return warns


# ───────────────────────── AST helpers ─────────────────────────
def dump(x, ws=False):
    """Universal AST -> plain dicts/lists (JSON-friendly)."""
    if isinstance(x, list):
        return [dump(i, ws) for i in x]
    if not x.kids:
        return {"type": x.type, "text": x.text, **({"class": x.cls} if x.cls else {})}
    named = {id(v) for f in x.fields.values() for v in (f if isinstance(f, list) else [f])}
    return {"type": x.type, "label": x.label, "params": x.params or None,
            "fields": {k: dump(v, ws) for k, v in x.fields.items()},
            "children": [dump(k, ws) for k in x.kids
                         if (ws or k.type != "_") and id(k) not in named]}


def sexp(x):
    """Compact S-expression rendering for debugging and tests."""
    if isinstance(x, list):
        return "[" + " ".join(sexp(i) for i in x) + "]"
    if not x.kids:
        return x.text
    if "inner" in x.fields:
        return sexp(x.fields["inner"])
    if x.label == "binary":
        return f"({sexp(x.fields['lhs'])} {x.fields['op'].text} {sexp(x.fields['rhs'])})"
    if x.label == "prefix":
        return f"({x.fields['op'].text}{sexp(x.fields['arg'])})"
    body = " ".join(sexp(k) for k in x.kids if k.type != "_")
    return f"({x.type}{'/' + x.label if x.label else ''} {body})"
