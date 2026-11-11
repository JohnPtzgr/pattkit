"""
recover: error-recovery layer. Runs only when the normal layers fail to reduce the
input to the start symbol. Two stages:

 1. Hand-written error patterns (grammar section `recover:`), written in the same
    pattern language and matched against the tokens left over after reduction:

        recover:
          missing_semi: ty=type name=ID EOF  => "declaration of '{name}' is missing ';'"

    Message placeholders: {capture}  {capture.line}  {capture.col}  {line} {col}
    {text} (matched span) {found} (first unreduced token).  First match, leftmost wins;
    among rules matching at the same place, file order wins.

 2. Backup edit search: tries up to `max_edits` (default 2) substitutions / insertions
    (optionally deletions) of tokens near the failure point and keeps the edit
    sequences after which the whole input reduces. Scored by how little is left
    unreduced; a beam keeps the search cheap; trial and time budgets bound it.
"""
import time
from copy import copy
from dataclasses import dataclass, field

from .engine import Engine, DList, klist
from .tokens import Tok


def name_of(t):
    return t.text if t.text.strip() else t.type


@dataclass
class Edit:
    kind: str                      # "insert" | "substitute" | "delete"
    pos: int                       # significant-token index the edit was made at
    new: Tok = None
    old: Tok = None
    near: Tok = None               # token before an insertion

    def describe(s):
        if s.kind == "substitute":
            return f"replace {name_of(s.old)!r} with {name_of(s.new)!r} at {s.old.line}:{s.old.col}"
        if s.kind == "delete":
            return f"delete {name_of(s.old)!r} at {s.old.line}:{s.old.col}"
        if s.near is not None:
            return (f"insert {name_of(s.new)!r} after {name_of(s.near)!r} "
                    f"({s.near.line}:{s.near.col + len(s.near.text)})")
        return f"insert {name_of(s.new)!r} at the start of the input"


@dataclass
class Suggestion:
    edits: list
    tokens: list = field(repr=False, default_factory=list)

    def describe(s):
        return "; ".join(e.describe() for e in s.edits)


@dataclass
class Diagnostic:
    line: int
    col: int
    message: str
    rule: str = None               # name of the matched error pattern, if any
    suggestions: list = field(default_factory=list)
    snippet: str = ""

    def format(s):
        out = [f"{s.line}:{s.col}: {s.message}"]
        if s.snippet:
            out += ["  " + s.snippet, "  " + " " * max(s.col - 1, 0) + "^"]
        if s.suggestions:
            out.append("  possible fixes:")
            out += [f"    {i}. {sg.describe()}" for i, sg in enumerate(s.suggestions, 1)]
        return "\n".join(out)


class Val:
    """Capture wrapper for message formatting: str() is the text, .line/.col work."""

    def __init__(s, x):
        s.toks = [t for t in (x if isinstance(x, list) else [x]) if t.type != "_"]

    def __str__(s):
        return " ".join(t.text for t in s.toks)

    @property
    def line(s):
        return s.toks[0].line if s.toks else 0

    @property
    def col(s):
        return s.toks[0].col if s.toks else 0


class Recovery:
    def __init__(self, parser, max_edits=2, window=2, beam=6, max_trials=4000,
                 budget=2.0, ops=("insert", "substitute"), suggestions=3):
        self.p, self.max_edits, self.window, self.beam = parser, max_edits, window, beam
        self.max_trials, self.budget, self.ops = max_trials, budget, ops
        self.max_suggestions = suggestions
        self._lits = None

    # ───────────── public ─────────────
    def diagnose(self, toks, res, src=None):
        sig = [t for t in res if t.type != "_"]
        leaves = [t for t in sig if not t.kids]
        stuck = leaves[0] if leaves else (sig[0] if sig else None)
        hit = self.match_patterns(res, stuck)
        if hit:
            rule, msg, at = hit
        else:
            rule, at = None, stuck
            msg = (f"cannot reduce to {self.p.start!r}: unexpected "
                   f"{name_of(stuck)!r}" if stuck else f"cannot reduce to {self.p.start!r}")
            if len(sig) > 1:
                msg += "  [left: " + " ".join(
                    (t.label and f"{t.type}/{t.label}") or t.type for t in sig[:10]) + "]"
        line, col = (at.line, at.col) if at else (1, 1)
        snippet = ""
        if src is not None:
            lines = src.split("\n")
            snippet = lines[line - 1] if 0 < line <= len(lines) else ""
        sugg = self.search(toks, leaves, sig)
        return Diagnostic(line, col, msg, rule, sugg, snippet)

    # ───────────── stage 1: hand-written patterns ─────────────
    def match_patterns(self, res, stuck):
        errs = self.p.g.errors
        if not errs:
            return None
        eng = Engine(self.p.g)
        eng.scope = self.p.root_scope()
        dl = DList(res)
        c = dl.head.nxt
        while True:
            if c.tok.type != "_":
                for r in errs:
                    for e, kids, items, env in eng.m(r.pat, c, None, None, {}):
                        kl = klist(kids)
                        return r.name, self.fmt(r.msg, kl, items, c, stuck), self.first(kl, c)
            if c.nxt is None:
                return None
            c = c.nxt

    @staticmethod
    def first(kids, c):
        return next((k for k in kids if k.type != "_"), c.tok)

    @staticmethod
    def fmt(msg, kids, items, c, stuck):
        vals = {}
        for name, val, _ in klist(items):
            vals[name] = Val(val if name not in vals else vals[name].toks + (
                val if isinstance(val, list) else [val]))
        at = Recovery.first(kids, c)
        vals.update(line=at.line, col=at.col, text=" ".join(k.text for k in kids if k.type != "_"),
                    found=name_of(stuck) if stuck else "end of input")
        try:
            return msg.format_map(vals)
        except (KeyError, AttributeError, IndexError):
            return msg

    # ───────────── stage 2: edit search ─────────────
    def literals(self):
        if self._lits is None:
            out = set()

            def walk(n):
                if isinstance(n, tuple):
                    if n and n[0] == "lit":
                        out.add(n[1])
                        return
                    for x in n[1:]:
                        walk(x)
                elif isinstance(n, list):
                    for x in n:
                        walk(x)

            for layer in self.p.g.layers:
                for kind, obj in layer.steps:
                    if kind == "rules":
                        for r in obj:
                            walk(r.pat)
                    else:
                        for _, ops in obj.levels:
                            out |= set(ops)
            self._lits = sorted(out)
        return self._lits

    def lex_literal(self, text):
        lx = self.p.lexer
        try:
            if hasattr(lx, "fragment"):
                ts = lx.fragment(text)
            else:
                ts = [t for t in lx.tokenize(text) if t.type not in ("_", "EOF")]
        except SyntaxError:
            return []
        return ts if len(ts) == 1 and ts[0].text == text else []

    def vocab(self, toks):
        out, seen = [], set()
        for text in self.literals():
            for t in self.lex_literal(text):
                if (t.type, t.text) not in seen:
                    seen.add((t.type, t.text))
                    out.append(t)
        lits, rep = set(self.literals()), {}
        for t in toks:                    # one representative per token type present
            if t.type == "_":
                continue
            if t.type not in rep or (rep[t.type].text in lits and t.text not in lits):
                rep[t.type] = t
        for t in rep.values():
            if (t.type, t.text) not in seen:
                seen.add((t.type, t.text))
                out.append(t)
        return out[:80]

    @staticmethod
    def key(ts):
        return tuple((t.type, t.text) for t in ts if t.type != "_")

    def search(self, toks, leaves, sig):
        if not sig and not toks:
            return []
        t0, trials, cache = time.time(), [0], {}
        W = self.window

        def over():
            return trials[0] >= self.max_trials or time.time() - t0 > self.budget

        def score(ts):
            k = self.key(ts)
            if k in cache:
                return cache[k]
            trials[0] += 1
            try:
                res, ast = self.p.reduce(ts)
            except (SyntaxError, RecursionError):
                cache[k] = (10 ** 6, 0)
                return cache[k]
            if ast is not None:
                cache[k] = (0, 0)
            else:
                sg = [t for t in res if t.type != "_"]
                cache[k] = (len(sg), sum(1 for t in sg if not t.kids))
            return cache[k]

        orig = [t for t in toks if t.type != "_"]
        if not orig or score(toks) == (0, 0):
            return []
        index = {}
        for i, t in enumerate(orig):
            index.setdefault(t.pos, i)
        centers = {index.get(leaves[0].pos, 0), index.get(leaves[-1].pos, 0), len(orig) - 1} \
            if leaves else {len(orig) - 1}
        vocab = self.vocab(toks)

        def single_edits(ts):
            sg = [i for i, t in enumerate(ts) if t.type != "_"]
            n = len(sg)
            subs = sorted({p for c in centers for p in range(c - W, c + W + 1) if 0 <= p < n})
            gaps = sorted({p for c in centers for p in range(c - W, c + W + 2) if 0 <= p <= n})
            if "substitute" in self.ops:
                for p in subs:
                    for v in vocab:
                        if (v.type, v.text) != (ts[sg[p]].type, ts[sg[p]].text):
                            yield "substitute", p, v
            if "insert" in self.ops:
                for p in gaps:
                    for v in vocab:
                        yield "insert", p, v
            if "delete" in self.ops:
                for p in subs:
                    yield "delete", p, None

        def apply(ts, kind, p, v):
            sg = [i for i, t in enumerate(ts) if t.type != "_"]
            if kind == "delete":
                i = sg[p]
                return ts[:i] + ts[i + 1:], Edit("delete", p, old=ts[i])
            new = copy(v)
            if kind == "substitute":
                i = sg[p]
                old = ts[i]
                new.pos, new.line, new.col = old.pos, old.line, old.col
                return ts[:i] + [new] + ts[i + 1:], Edit("substitute", p, new, old=old)
            if p > 0:
                near = ts[sg[p - 1]]
                i = sg[p - 1] + 1
                new.pos, new.line = near.pos + len(near.text), near.line
                new.col = near.col + len(near.text)
            else:
                near, i = None, 0
                ref = ts[sg[0]]
                new.pos, new.line, new.col = ref.pos, ref.line, ref.col
            return ts[:i] + [new] + ts[i:], Edit("insert", p, new, near=near)

        level1, found = [], []
        for kind, p, v in single_edits(toks):
            if over():
                break
            ts2, ed = apply(toks, kind, p, v)
            sc = score(ts2)
            level1.append((sc, [ed], ts2))
            if sc == (0, 0):
                found.append((sc, [ed], ts2))
        if not found and self.max_edits >= 2:
            for sc, eds, ts in sorted(level1, key=lambda x: x[0])[:self.beam]:
                for kind, p, v in single_edits(ts):
                    if over():
                        break
                    ts2, ed = apply(ts, kind, p, v)
                    if score(ts2) == (0, 0):
                        found.append(((0, 0), eds + [ed], ts2))
        present = {(t.type, t.text) for t in orig}

        def rank(x):
            eds = x[1]
            novelty = sum(1 for e in eds if e.new is not None and (e.new.type, e.new.text) not in present)
            # fewer edits; then tokens already used in the input; then the rightmost position
            # (a left-to-right parser would notice the error as late as possible)
            return (len(eds), novelty, -sum(e.pos for e in eds))

        found.sort(key=rank)
        out, seen = [], set()
        for _, eds, ts in found:
            k = self.key(ts)
            if k not in seen:
                seen.add(k)
                out.append(Suggestion(eds, ts))
            if len(out) >= self.max_suggestions:
                break
        return out
