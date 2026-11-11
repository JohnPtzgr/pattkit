"""
engine: grammar loader + layered bottom-up token-rewriting engine.

Grammar file
------------
    names IDENTIFIER              # token types the name table may re-classify
    unknown UNKNOWN               # (optional) class given to names missing from the table
    stops ";" "{" "}"             # a bare * / + never crosses these tokens
    layer NAME [strict]:          # layers run in order, each to a fixpoint
      type/label:  pattern        # rule: rewrites a matched slice into ONE token
      operators RESULT over T1 T2 # operator table, levels LOW priority -> HIGH
        left "+" "-", right "**", prefix "-", postfix "++", nonassoc "<",
        chain "<" ">" "==", apply
    recover:                      # see recover.py
      name: pattern => "message with {captures}"

Pattern elements
----------------
    NAME  NAME/label   token by type (or class from the name table), optionally by label
    "text"  \\<         leaf token by text
    _  _{nl} _{!nl} _{nl>1} _{w=4} _{w=n}    whitespace token and its conditions;  !_ = none
    A B   A | B   (..)   A*  A+  A?  A*?  A+?       (postfix only when written adjacent)
    X{n}  X{3}  X{n+1}               X repeated exactly n times (n shared, backtracked)
    X % S                            one or more X separated by S   ((x=X) % S captures each)
    *  +  *?  +?                     wildcard: any tokens, 0+/1+, greedy/lazy
    *{kind = T}                      wildcard slice must reduce (through all layers) to one T
    <OPEN ... CLOSE>                 balanced pair; the inside is reduced as an isolated run
                                     and then matched (OPEN/CLOSE may be alternations)
    name=elem   name=ID as CLASS     capture / capture that declares the text in the scope
    &X  !X                           lookahead
Non-strict layers insert an implicit `_?` between elements unless `_` / `!_` is written.
"""
import re
import sys
from copy import copy

from .tokens import Tok

sys.setrecursionlimit(50000)
INF = 10 ** 9


# ═════════════════════════════ pattern language ═════════════════════════════
PT = re.compile(r'("(?:[^"\\]|\\.)*")|(\\.)|(_(?!\w))|([A-Za-z]\w*)|(\d+)'
                r'|(=>|\*\?|\+\?|[=|()*+?&!{}<>,\-/%])')
KINDS = {1: "str", 2: "esc", 3: "ws", 4: "id", 5: "int", 6: "op"}


def lex_pat(s):
    out, i = [], 0
    while True:
        j = i
        while i < len(s) and s[i].isspace():
            i += 1
        if i >= len(s):
            return out
        m = PT.match(s, i)
        if not m:
            raise SyntaxError(f"pattern: bad text near {s[i:i + 15]!r}")
        out.append((KINDS[m.lastindex], m.group(m.lastindex), i > j))
        i = m.end()


def unq(s):
    return s[1:-1].replace('\\"', '"').replace("\\\\", "\\")


def is_ws(n):
    t = n[0]
    if t == "ws":
        return True
    if t == "look":
        return is_ws(n[1])
    if t == "rep":
        return is_ws(n[1])
    if t == "cap":
        return is_ws(n[2])
    return False


class PatParser:
    def __init__(s, toks, skipmode=True):
        s.t = [(k, v) for k, v, _ in toks]
        s.sp = [sp for _, _, sp in toks]
        s.i, s.skip = 0, skipmode

    def peek(s):
        return s.t[s.i][1] if s.i < len(s.t) else None

    def eat(s):
        if s.i >= len(s.t):
            raise SyntaxError("pattern ended unexpectedly")
        s.i += 1
        return s.t[s.i - 1]

    def expect(s, v):
        got = s.eat()[1]
        if got != v:
            raise SyntaxError(f"pattern: expected {v!r}, got {got!r}")

    def adj(s):
        return s.i < len(s.t) and not s.sp[s.i]

    def rule(s):
        n = s.alt()
        if s.i < len(s.t):
            raise SyntaxError(f"pattern: unexpected {s.peek()!r}")
        return n

    def with_skips(s, items):
        if not s.skip:
            return items
        out = []
        for i, it in enumerate(items):
            if i and not (is_ws(items[i - 1]) or is_ws(it)):
                out.append(("skip",))
            out.append(it)
        return out

    def alt(s):
        alts = [s.seq()]
        while s.peek() == "|":
            s.eat()
            alts.append(s.seq())
        return alts[0] if len(alts) == 1 else ("alt", alts)

    def seq(s):
        items = []
        while s.peek() not in (None, "|", ")"):
            items.append(s.quant())
        items = s.with_skips(items)
        return items[0] if len(items) == 1 else ("seq", items)

    def counter(s):
        k, v = s.eat()
        if k == "int":
            return (None, int(v))
        var, c = v, 0
        if s.peek() in ("+", "-"):
            sign = 1 if s.eat()[1] == "+" else -1
            c = sign * int(s.eat()[1])
        return (var, c)

    def quant(s):
        n = s.atom()
        while s.peek() is not None and s.adj():
            v = s.peek()
            if v == "{":
                s.eat()
                a = s.counter()
                s.expect("}")
                n = ("cnt", n, a, s.skip)
            elif v in ("*", "+", "?", "*?", "+?"):
                s.eat()
                lo, hi, gr = {"*": (0, INF, True), "+": (1, INF, True), "?": (0, 1, True),
                              "*?": (0, INF, False), "+?": (1, INF, False)}[v]
                n = ("rep", n, lo, hi, gr, s.skip)
            else:
                break
        while s.peek() == "%":                      # X % SEP : one or more, separated
            s.eat()
            sep = s.atom()
            tail = ("seq", [("skip",), sep, ("skip",), n]) if s.skip else ("seq", [sep, n])
            n = ("seq", [n, ("rep", tail, 0, INF, True, False)])
        return n

    def wscond(s):
        s.eat()
        neg = s.peek() == "!"
        if neg:
            s.eat()
        name = s.eat()[1]
        if name == "nl":
            if s.peek() == ">":
                s.eat()
                c = ("nlgt", int(s.eat()[1]))
            else:
                c = ("nl", not neg)
        elif name == "w":
            s.expect("=")
            c = ("w", s.counter())
        else:
            raise SyntaxError(f"unknown whitespace condition {name!r}")
        s.expect("}")
        return c

    def atom(s):
        k, v = s.eat()
        if k == "str":
            return ("lit", unq(v))
        if k == "esc":
            return ("lit", v[1:])
        if k == "ws":
            return ("ws", s.wscond() if s.peek() == "{" and s.adj() else None)
        if k == "id":
            if s.peek() == "=":
                s.eat()
                sub, cls = s.quant(), None
                if s.peek() == "as":
                    s.eat()
                    cls = s.eat()[1]
                return ("cap", v, sub, cls)
            if s.peek() == "/" and s.adj():
                s.eat()
                return ("tok", v, s.eat()[1])
            return ("tok", v)
        if k == "op":
            if v in ("*", "+", "*?", "+?"):
                target = None
                if s.peek() == "{" and s.adj():
                    s.eat()
                    s.expect("kind")
                    s.expect("=")
                    target = s.eat()[1]
                    s.expect("}")
                return ("any", 0 if v[0] == "*" else 1, not v.endswith("?"), target)
            if v == "(":
                n = s.alt()
                s.expect(")")
                return n
            if v in ("&", "!"):
                return ("look", s.atom(), v == "&")
            if v == "<":
                return s.bal()
        raise SyntaxError(f"pattern: unexpected {v!r}")

    def bal(s):
        items = []
        while s.peek() != ">":
            if s.peek() is None:
                raise SyntaxError("pattern: unclosed <...>")
            items.append(s.quant())
        s.eat()
        if len(items) < 2:
            raise SyntaxError("pattern: <OPEN ... CLOSE> needs two delimiters")
        mid = s.with_skips(items[1:-1])
        inner = ("seq", mid) if len(mid) != 1 else mid[0]
        if s.skip:
            inner = ("seq", [("skip",), inner, ("skip",)])
        return ("bal", items[0], inner, items[-1])


# ═════════════════════════════ grammar loader ═════════════════════════════
class Rule:
    def __init__(s, type, label, pat):
        s.type, s.label, s.pat = type, label, pat


class ErrorRule:
    def __init__(s, name, pat, msg):
        s.name, s.pat, s.msg = name, pat, msg


class OpSpec:
    def __init__(s, result, operands, levels):
        s.result, s.operands, s.levels = result, set(operands) | {result}, levels


class Layer:
    def __init__(s, name, strict):
        s.name, s.strict, s.steps = name, strict, []


class Grammar:
    HDR = re.compile(r'^[ \t]*(layer\b[^\n:]*:|operators\b[^\n]*|names\b[^\n]*|'
                     r'unknown\b[^\n]*|stops\b[^\n]*|recover[ \t]*:|'
                     r'\w+(?:/\w+)?[ \t]*:)', re.M)

    def __init__(s, text):
        s.layers, s.names, s.stops, s.coerce, s.errors = [], set(), set(), {}, []
        s.unknown = None
        section, cur = "layer", None
        text = s.strip_comments(text)
        ms = list(s.HDR.finditer(text))
        for i, m in enumerate(ms):
            head = m.group(1).strip()
            body = text[m.end(): ms[i + 1].start() if i + 1 < len(ms) else len(text)]
            if head.startswith("layer") and not head.startswith("layers"):
                w = head[:-1].split()
                cur = Layer(w[1], "strict" in w[2:])
                s.layers.append(cur)
                section = "layer"
            elif head.startswith("names"):
                s.names |= set(head.split()[1:])
            elif head.startswith("unknown"):
                s.unknown = head.split()[1]
            elif head.startswith("stops"):
                s.stops |= {unq(v) for k, v, _ in lex_pat(head[5:]) if k == "str"}
            elif re.fullmatch(r"recover\s*:", head):
                section = "recover"
            elif section == "recover":
                name = head[:-1].strip().partition("/")[0]
                toks = lex_pat(body)
                arrows = [j for j, t in enumerate(toks) if t[1] == "=>"]
                if not arrows or arrows[0] + 1 >= len(toks) or toks[arrows[0] + 1][0] != "str":
                    raise SyntaxError(f"recover rule {name!r} needs: pattern => \"message\"")
                s.errors.append(ErrorRule(name, PatParser(toks[:arrows[0]], True).rule(),
                                          unq(toks[arrows[0] + 1][1])))
            elif cur is None:
                raise SyntaxError(f"rule outside any layer: {head!r}")
            elif head.startswith("operators"):
                w = head.split()
                spec = OpSpec(w[1], w[3:], s.levels(body))
                s.coerce[spec.result] = spec.operands
                cur.steps.append(("ops", spec))
            else:
                name, _, label = head[:-1].strip().partition("/")
                pat = PatParser(lex_pat(body), not cur.strict).rule()
                if not cur.steps or cur.steps[-1][0] != "rules":
                    cur.steps.append(("rules", []))
                cur.steps[-1][1].append(Rule(name, label or None, pat))

    @staticmethod
    def strip_comments(t):
        out = []
        for line in t.split("\n"):
            q, i = False, 0
            while i < len(line):
                ch = line[i]
                if ch == "\\":
                    i += 2
                    continue
                if ch == '"':
                    q = not q
                elif ch == "#" and not q:
                    line = line[:i]
                    break
                i += 1
            out.append(line)
        return "\n".join(out)

    @staticmethod
    def levels(body):
        toks, i, out = lex_pat(body), 0, []
        while i < len(toks):
            assoc = toks[i][1]
            if assoc not in ("left", "right", "nonassoc", "prefix", "postfix", "chain", "apply"):
                raise SyntaxError(f"operators: unknown associativity {assoc!r}")
            i += 1
            ops = []
            while i < len(toks) and toks[i][0] == "str":
                ops.append(unq(toks[i][1]))
                i += 1
            if not ops and assoc != "apply":
                raise SyntaxError(f"operators: level {assoc!r} has no operators")
            out.append((assoc, frozenset(ops)))
            if i < len(toks) and toks[i][1] == ",":
                i += 1
        return out


# ═════════════════════════════ runtime structures ═════════════════════════════
class Cell:
    __slots__ = ("tok", "prv", "nxt")

    def __init__(s, tok, prv=None, nxt=None):
        s.tok, s.prv, s.nxt = tok, prv, nxt


class DList:
    """Doubly linked token list with head/tail sentinels; replace() is O(1)."""

    def __init__(s, toks):
        s.head, s.tail = Cell(None), Cell(Tok("EOF"))
        p = s.head
        for t in toks:
            c = Cell(t, p)
            p.nxt = c
            p = c
        p.nxt, s.tail.prv = s.tail, p

    def toks(s):
        out, c = [], s.head.nxt
        while c.nxt is not None:
            out.append(c.tok)
            c = c.nxt
        return out

    def replace(s, a, b, tok):
        c = Cell(tok, a.prv, b.nxt)
        a.prv.nxt = c
        b.nxt.prv = c
        return c


class K:
    """Persistent cons list of consumed tokens: O(1) append, shared between backtracking branches."""
    __slots__ = ("prev", "tok", "n")

    def __init__(s, prev, tok):
        s.prev, s.tok, s.n = prev, tok, (prev.n if prev else 0) + 1


def klen(k):
    return k.n if k else 0


def kext(k, toks):
    for t in toks:
        k = K(k, t)
    return k


def klist(k, upto=0):
    out = []
    while k is not None and k.n > upto:
        out.append(k.tok)
        k = k.prev
    out.reverse()
    return out


class Scope:
    """Name table: text -> class token. A chain of layers; writes land at layer end."""

    def __init__(s, parent=None):
        s.parent, s.table, s.pending = parent, {}, {}

    def lookup(s, text):
        x = s
        while x:
            if text in x.table:
                return x.table[text]
            x = x.parent
        return None

    def commit(s):
        changed = any(s.table.get(k) != v for k, v in s.pending.items())
        s.table.update(s.pending)
        s.pending.clear()
        return changed


# ═════════════════════════════ engine ═════════════════════════════
class Engine:
    BACK = 8                 # cells to rewind after a replacement (passes repeat to a fixpoint)
    MAX_DEPTH = 300

    def __init__(self, grammar):
        self.g, self.scope = grammar, None
        self.memo, self.busy, self.depth = {}, set(), 0

    # ---------- helpers ----------
    @staticmethod
    def named(tk, name):
        return tk.type == name or tk.cls == name

    def kind_ok(self, tk, target):
        return (self.named(tk, target) or
                (tk.cls or tk.type) in self.g.coerce.get(target, ()))

    def is_(self, p, tk):
        if p[0] == "lit":
            return not tk.kids and tk.text == p[1]
        if p[0] == "tok":
            return self.named(tk, p[1]) and (len(p) < 3 or tk.label == p[2])
        if p[0] == "alt":
            return any(self.is_(a, tk) for a in p[1])
        raise SyntaxError("bad delimiter in <...>")

    @staticmethod
    def mknode(type, label, kids, fields, env):
        first = next((k for k in kids if k.type != "_"), kids[0])
        n = Tok(type, "".join(k.text for k in kids), first.pos, first.line, first.col,
                kids, label, fields)
        n.params = env
        return n

    # ---------- running ----------
    def run(self, toks, scope):
        old, self.scope = self.scope, scope
        try:
            dl = DList(toks)
            self.reclass(dl, scope)
            for layer in self.g.layers:
                while True:
                    changed = False
                    for kind, obj in layer.steps:
                        changed |= (self.apply_rules(dl, obj) if kind == "rules"
                                    else self.apply_ops(dl, obj))
                    if not changed:
                        break
                if scope.commit():              # the table changes only between layers
                    self.memo.clear()
                    self.reclass(dl, scope)
            return dl.toks()
        finally:
            self.scope = old

    def sub(self, toks):
        """Reduce a slice as an isolated run with a child scope; None if re-entrant."""
        if not toks:
            return []
        key = (id(toks[0]), id(toks[-1]), len(toks), id(self.scope))
        if key in self.memo:
            return self.memo[key][1]
        if key in self.busy:
            return None
        if self.depth > self.MAX_DEPTH:
            raise RecursionError("capture nesting too deep")
        self.busy.add(key)
        self.depth += 1
        try:
            res = self.run(list(toks), Scope(self.scope))
        finally:
            self.depth -= 1
            self.busy.discard(key)
        self.memo[key] = (toks, res)            # keep toks alive so ids stay unique
        return res

    def reclass(self, dl, scope):
        c, unk = dl.head.nxt, self.g.unknown
        while c.nxt is not None:
            t = c.tok
            if not t.kids and t.type in self.g.names:
                cls = scope.lookup(t.text) or unk
                if cls != t.cls:
                    t2 = copy(t)
                    t2.cls = cls
                    c.tok = t2
            c = c.nxt

    # ---------- rules: leftmost, longest across rules, tie -> earlier rule ----------
    def apply_rules(self, dl, rules):
        changed = False
        while True:
            did, c = False, dl.head.nxt
            while c.nxt is not None:
                best = None
                for r in rules:
                    for e, kids, items, env in self.m(r.pat, c, None, None, {}):
                        n, x = 0, c
                        while x is not e:
                            n += 1
                            x = x.nxt
                        if n == 0 or (klen(kids) == 1 and kids.tok.type == r.type):
                            break                # empty / no-op unary: not a reduction
                        if best is None or n > best[0]:
                            best = (n, r, e, kids, items, env)
                        break                    # first yield = preferred inner choice
                if best is None:
                    c = c.nxt
                    continue
                _, r, e, kids, items, env = best
                node = self.mknode(r.type, r.label, klist(kids), self.fields(items), env)
                c = dl.replace(c, e.prv, node)
                did = changed = True
                for _ in range(self.BACK):
                    if c.prv is not dl.head:
                        c = c.prv
            if not did:
                return changed

    def fields(self, items):
        f, owned = {}, set()
        for name, val, cls in klist(items):
            vals = val if isinstance(val, list) else [val]
            if name in f:
                if name not in owned:               # copy once, then extend in place
                    f[name] = list(f[name]) if isinstance(f[name], list) else [f[name]]
                    owned.add(name)
                f[name].extend(vals)
            else:
                f[name] = val
            if cls:
                for tk in vals:
                    self.scope.pending[tk.text] = cls
        return f

    # ---------- skeleton matcher (backtracking generators over cells) ----------
    def m(self, n, c, kids, items, env):
        t = n[0]
        if t == "tok":
            tk = c.tok
            if self.named(tk, n[1]) and (len(n) < 3 or tk.label == n[2]):
                if c.nxt is None:                 # EOF sentinel: matches, consumes nothing
                    yield c, kids, items, env
                else:
                    yield c.nxt, K(kids, tk), items, env
        elif t == "lit":
            if c.nxt is not None and not c.tok.kids and c.tok.text == n[1]:
                yield c.nxt, K(kids, c.tok), items, env
        elif t == "ws":
            tk = c.tok
            if c.nxt is None or tk.type != "_":
                return
            cond, e, ok = n[1], env, True
            if cond:
                if cond[0] == "nl":
                    ok = (tk.nl > 0) == cond[1]
                elif cond[0] == "nlgt":
                    ok = tk.nl > cond[1]
                else:
                    var, cc = cond[1]
                    if var is None:
                        ok = tk.w == cc
                    elif var in env:
                        ok = env[var] + cc == tk.w
                    else:
                        e = {**env, var: tk.w - cc}
            if ok:
                yield c.nxt, K(kids, tk), items, e
        elif t == "skip":
            if c.nxt is not None and c.tok.type == "_":
                yield c.nxt, K(kids, c.tok), items, env
            else:
                yield c, kids, items, env
        elif t == "seq":
            yield from self.seq(n[1], 0, c, kids, items, env)
        elif t == "alt":
            for a in n[1]:
                yield from self.m(a, c, kids, items, env)
        elif t == "rep":
            _, sub, lo, hi, greedy, sep = n
            for p, kd, it, e, _k in self.iterate(sub, c, kids, items, env, lo, hi, greedy, sep):
                yield p, kd, it, e
        elif t == "cnt":
            _, sub, (var, cc), sep = n
            if var is None or var in env:
                want = cc if var is None else env[var] + cc
                if want >= 0:
                    for p, kd, it, e, _k in self.iterate(sub, c, kids, items, env,
                                                         want, want, True, sep):
                        yield p, kd, it, e
            else:                                  # free n: longest first, bind n
                for p, kd, it, e, k in self.iterate(sub, c, kids, items, env, 1, INF, True, sep):
                    if k - cc >= 1:
                        yield p, kd, it, {**e, var: k - cc}
        elif t == "cap":
            _, name, sub, cls = n
            k0 = klen(kids)
            for e, kd, it, en in self.m(sub, c, kids, items, env):
                seg = [x for x in klist(kd, k0) if x.type != "_"]
                val = seg[0] if len(seg) == 1 and sub[0] not in ("rep", "cnt") else seg
                yield e, kd, K(it, (name, val, cls)), en
        elif t == "look":
            if any(True for _ in self.m(n[1], c, None, None, env)) == n[2]:
                yield c, kids, items, env
        elif t == "any":
            yield from self.any(n, c, kids, items, env)
        elif t == "bal":
            yield from self.bal(n, c, kids, items, env)
        else:
            raise SyntaxError(f"unknown pattern node {t!r}")

    def seq(self, ns, i, c, kids, items, env):
        if i == len(ns):
            yield c, kids, items, env
            return
        for p, kd, it, e in self.m(ns[i], c, kids, items, env):
            yield from self.seq(ns, i + 1, p, kd, it, e)

    def iterate(self, sub, c, kids, items, env, lo, hi, greedy, sep):
        """All feasible repeat counts of `sub`, with full backtracking but an explicit
        stack instead of recursion (long lists do not hit the Python stack limit).
        Yields (cell, kids, items, env, count): longest first if greedy, else shortest."""
        stack = [[(c, kids, items, env), None, 0, None]]
        while stack:
            fr = stack[-1]
            if fr[1] is None:
                st, k = fr[0], fr[2]
                if not greedy and k >= lo:
                    yield st + (k,)
                if k < hi:
                    cc, kk = st[0], st[1]
                    if sep and k > 0 and cc.nxt is not None and cc.tok.type == "_":
                        cc, kk = cc.nxt, K(kk, cc.tok)
                    fr[3] = cc
                    fr[1] = self.m(sub, cc, kk, st[2], st[3])
                else:
                    fr[1] = iter(())
            try:
                p, kd, it, e = next(fr[1])
            except StopIteration:
                stack.pop()
                if greedy and fr[2] >= lo:
                    yield fr[0] + (fr[2],)
                continue
            if p is not fr[3]:                    # an iteration must make progress
                stack.append([(p, kd, it, e), None, fr[2] + 1, None])

    def any(self, n, c, kids, items, env):
        """Wildcard. Candidate lengths are generated greedily (long -> short) or lazily
        (short -> long, stopping as soon as the caller accepts)."""
        _, lo, greedy, target = n
        stops = self.g.stops

        def take(seg):
            if target is None:
                return True, kext(kids, seg)
            i, j = 0, len(seg)
            while i < j and seg[i].type == "_":
                i += 1
            while j > i and seg[j - 1].type == "_":
                j -= 1
            core = seg[i:j]
            if not core:
                return False, None
            red = core if (len(core) == 1 and self.kind_ok(core[0], target)) else self.sub(core)
            if red is None:
                return False, None
            sig = [r for r in red if r.type != "_"]
            if len(sig) == 1 and self.kind_ok(sig[0], target):
                return True, kext(kext(kext(kids, seg[:i]), red), seg[j:])
            return False, None

        def stop(x):
            return not x.tok.kids and x.tok.text in stops

        if greedy:
            cells, x = [], c
            while x.nxt is not None and not stop(x):
                cells.append(x)
                x = x.nxt
            toks = [cc.tok for cc in cells]
            for k in range(len(cells), lo - 1, -1):
                ok, kd = take(toks[:k])
                if ok:
                    yield (cells[k] if k < len(cells) else x), kd, items, env
        else:
            x, seg = c, []
            while True:
                if len(seg) >= lo:
                    ok, kd = take(seg)
                    if ok:
                        yield x, kd, items, env
                if x.nxt is None or stop(x):
                    break
                seg.append(x.tok)
                x = x.nxt

    def bal(self, n, c, kids, items, env):
        """<OPEN inner CLOSE>: find the matching CLOSE, reduce what lies between as an
        isolated run, then match `inner` against the reduced tokens (full coverage)."""
        _, op, inner, cl = n
        if c.nxt is None or not self.is_(op, c.tok):
            return
        depth, x = 0, c
        while x.nxt is not None:
            if self.is_(op, x.tok):
                depth += 1
            elif self.is_(cl, x.tok):
                depth -= 1
                if depth == 0:
                    break
            x = x.nxt
        if x.nxt is None:
            return                                # unbalanced
        mid, y = [], c.nxt
        while y is not x:
            mid.append(y.tok)
            y = y.nxt
        red = self.sub(mid)
        if red is None:
            return
        dl = DList(red)
        for e, k2, i2, e2 in self.m(inner, dl.head.nxt, None, None, env):
            if e.nxt is None:                     # inner covered everything
                yield x.nxt, K(kext(K(kids, c.tok), klist(k2)), x.tok), kext(items, klist(i2)), e2
                return

    # ---------- operator tables ----------
    @staticmethod
    def pws(c):
        c = c.prv
        while c.tok is not None and c.tok.type == "_":
            c = c.prv
        return c

    @staticmethod
    def nws(c):
        c = c.nxt
        while c.nxt is not None and c.tok.type == "_":
            c = c.nxt
        return c

    def apply_ops(self, dl, spec):
        changed = False

        def operand(c):
            return (c.tok is not None and c.nxt is not None and
                    (c.tok.cls or c.tok.type) in spec.operands)

        def is_op(c, ops):
            t = c.tok
            return (t is not None and c.nxt is not None and not t.kids and t.type != "_"
                    and t.text in ops and (t.cls or t.type) not in spec.operands)

        def red(a, b, label, fields):
            kids, x = [], a
            while True:
                kids.append(x.tok)
                if x is b:
                    break
                x = x.nxt
            return dl.replace(a, b, self.mknode(spec.result, label, kids, fields, {}))

        def same_line(a, b):                 # no line break between cells a and b
            x = a.nxt
            while x is not b:
                if x.tok.nl > 0:
                    return False
                x = x.nxt
            return True

        for assoc, ops in reversed(spec.levels):          # highest priority first
            if assoc in ("left", "nonassoc"):
                c = dl.head.nxt
                while c.nxt is not None:
                    if is_op(c, ops):
                        L, R = self.pws(c), self.nws(c)
                        if operand(L) and operand(R):
                            cell = red(L, R, "binary", {"lhs": L.tok, "op": c.tok, "rhs": R.tok})
                            changed = True
                            if assoc == "nonassoc":
                                nx = self.nws(cell)
                                if is_op(nx, ops) and operand(self.nws(nx)):
                                    raise SyntaxError(
                                        f"{nx.tok.line}:{nx.tok.col}: operator {nx.tok.text!r} "
                                        "is non-associative and cannot be chained")
                            c = cell.nxt
                            continue
                    c = c.nxt
            elif assoc == "chain":                       # a < b < c  ->  one node, lists
                c = dl.head.nxt
                while c.nxt is not None:
                    if is_op(c, ops):
                        L, R = self.pws(c), self.nws(c)
                        if operand(L) and operand(R):
                            operands, opl, last = [L.tok, R.tok], [c.tok], R
                            while True:
                                nx = self.nws(last)
                                r2 = self.nws(nx) if is_op(nx, ops) else None
                                if r2 is None or not operand(r2):
                                    break
                                opl.append(nx.tok)
                                operands.append(r2.tok)
                                last = r2
                            cell = red(L, last, "chain", {"operands": operands, "ops": opl})
                            changed, c = True, cell.nxt
                            continue
                    c = c.nxt
            elif assoc in ("right", "prefix"):
                c = dl.tail.prv
                while c.tok is not None:
                    if is_op(c, ops):
                        L, R = self.pws(c), self.nws(c)
                        if assoc == "right" and operand(L) and operand(R):
                            cell = red(L, R, "binary", {"lhs": L.tok, "op": c.tok, "rhs": R.tok})
                            changed, c = True, cell.prv
                            continue
                        if assoc == "prefix" and operand(R) and not operand(L):
                            cell = red(c, R, "prefix", {"op": c.tok, "arg": R.tok})
                            changed, c = True, cell.prv
                            continue
                    c = c.prv
            elif assoc == "apply":                       # juxtaposition: f x y -> ((f x) y)
                c = dl.head.nxt
                while c.nxt is not None:
                    if operand(c):
                        R = self.nws(c)
                        if operand(R) and same_line(c, R):
                            cell = red(c, R, "apply", {"fn": c.tok, "arg": R.tok})
                            changed, c = True, cell
                            continue
                    c = c.nxt
            else:                                        # postfix
                c = dl.head.nxt
                while c.nxt is not None:
                    if is_op(c, ops):
                        L = self.pws(c)
                        if operand(L):
                            cell = red(L, c, "postfix", {"arg": L.tok, "op": c.tok})
                            changed, c = True, cell.nxt
                            continue
                    c = c.nxt
        return changed
