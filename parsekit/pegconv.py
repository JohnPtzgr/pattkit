"""
pegconv: convert a pegen-style grammar file (the format used by CPython's python.gram and
documented in PEP 617) into a parsekit `peg` layer.

    from parsekit.pegconv import convert
    text = convert(open("python.gram").read(), start="file",
                   token_map={"ENDMARKER": "EOF"}, drop_tokens={"TYPE_COMMENT"})

Handled: `rule: | alt | alt`, `rule[type]:`, comments, 'kw' / "soft" strings, groups, `[e]`,
`e? e* e+`, `s.e+` (gather), `&e` `!e` lookaheads, `~` (cut), alternatives that mention
`invalid_*` rules (dropped: they only matter for the second, error-reporting pass), rules
unreachable from the start rule (dropped), dropped tokens / rules (optional uses vanish,
alternatives that need them are removed).

The result has one rule line per alternative (labelled `a1`, `a2`, ...) so the matched
alternative is visible on the resulting nodes.
"""
import re

TOK = re.compile(r"""\s*(?:('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")|([A-Za-z_]\w*)|(&&|[|()\[\]*+?&!~.:]))""")


class ConvError(Exception):
    pass


# ───────────────────────── text preparation ─────────────────────────
def strip_comments(text):
    out = []
    for line in text.split("\n"):
        q, i = None, 0
        while i < len(line):
            ch = line[i]
            if q:
                if ch == "\\":
                    i += 1
                elif ch == q:
                    q = None
            elif ch in "'\"":
                q = ch
            elif ch == "#":
                line = line[:i]
                break
            i += 1
        out.append(line.rstrip())
    return out


def split_rules(lines):
    """-> [(name, body_text)] ; rules start at column 0 (after common margin removal)."""
    margin = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    lines = [l[margin:] for l in lines]
    hdr = re.compile(r"^([A-Za-z_]\w*)(?:\[[^\]]*\])?:(.*)$")
    rules, cur = [], None
    for l in lines:
        m = hdr.match(l)
        if m:
            cur = [m.group(1), m.group(2)]
            rules.append(cur)
        elif cur is not None and l.strip():
            cur[1] += "\n" + l
    return [(n, b) for n, b in rules]


# ───────────────────────── parsing ─────────────────────────
class P:
    def __init__(s, text, rule):
        s.t, s.i, s.rule, s.toks = text, 0, rule, []
        pos = 0
        while pos < len(text):
            m = TOK.match(text, pos)
            if not m:
                if text[pos:].strip():
                    raise ConvError(f"rule {rule!r}: cannot tokenize near {text[pos:pos + 20]!r}")
                break
            s.toks.append(("str", m.group(1)) if m.group(1) else
                          ("name", m.group(2)) if m.group(2) else ("op", m.group(3)))
            pos = m.end()

    def peek(s, k=0):
        return s.toks[s.i + k][1] if s.i + k < len(s.toks) else None

    def kind(s):
        return s.toks[s.i][0] if s.i < len(s.toks) else None

    def eat(s):
        s.i += 1
        return s.toks[s.i - 1]

    def alts(s, closer=None):
        out = []
        if s.peek() == "|":
            s.eat()
        out.append(s.alt(closer))
        while s.peek() == "|":
            s.eat()
            out.append(s.alt(closer))
        return out

    def alt(s, closer):
        items = []
        while s.peek() is not None and s.peek() not in ("|", closer):
            items.append(s.item())
        if not items:
            raise ConvError(f"rule {s.rule!r}: empty alternative")
        return items

    def item(s):
        v = s.peek()
        if s.kind() == "op" and v in ("&", "!", "&&"):
            s.eat()
            return ("look" if v != "&&" else "eager", s.postfix(), v != "!")
        if s.kind() == "op" and v == "~":
            s.eat()
            return ("cut",)
        return s.postfix()

    def postfix(s):
        a = s.atom()
        if s.peek() == "." and s.kind() == "op":            # gather  s.e+
            s.eat()
            e = s.atom()
            if s.peek() != "+":
                raise ConvError(f"rule {s.rule!r}: gather needs '+'")
            s.eat()
            return ("gather", a, e)
        while s.kind() == "op" and s.peek() in ("*", "+", "?"):
            a = ("rep", a, s.eat()[1])
        return a

    def atom(s):
        k, v = s.eat()
        if k == "str":
            return ("lit", v[1:-1].replace("\\'", "'").replace('\\"', '"'))
        if k == "name":
            return ("name", v)
        if v == "(":
            a = s.alts(")")
            s.eat()
            return ("group", a)
        if v == "[":
            a = s.alts("]")
            s.eat()
            return ("opt", a)
        raise ConvError(f"rule {s.rule!r}: unexpected {v!r}")


# ───────────────────────── pruning ─────────────────────────
class Pruner:
    def __init__(s, drop_names, drop_prefixes=("invalid_",)):
        s.drop, s.pre = set(drop_names), tuple(drop_prefixes)

    def dead_name(s, n):
        return n in s.drop or n.startswith(s.pre)

    def alts(s, alts):
        out = []
        for a in alts:
            items = []
            for it in a:
                r = s.item(it)
                if r is DEAD:
                    break
                if r is not GONE:
                    items.append(r)
            else:
                if items:
                    out.append(items)
        return out

    def item(s, it):
        k = it[0]
        if k == "name":
            return DEAD if s.dead_name(it[1]) else it
        if k in ("lit", "cut"):
            return it
        if k in ("look", "eager"):
            r = s.item(it[1])
            return GONE if r in (DEAD, GONE) else (k, r, it[2])
        if k == "gather":
            a, e = s.item(it[1]), s.item(it[2])
            return DEAD if DEAD in (a, e) or GONE in (a, e) else ("gather", a, e)
        if k == "group":
            alts = s.alts(it[1])
            return DEAD if not alts else ("group", alts)
        if k == "opt":
            alts = s.alts(it[1])
            return GONE if not alts else ("opt", alts)
        if k == "rep":
            r = s.item(it[1])
            if r in (DEAD, GONE):
                return DEAD if it[2] == "+" else GONE
            return ("rep", r, it[2])
        raise ConvError(f"unknown item {it!r}")


DEAD, GONE = object(), object()


def names_in(alts, out):
    for a in alts:
        for it in a:
            names_in_item(it, out)


def names_in_item(it, out):
    k = it[0]
    if k == "name":
        out.add(it[1])
    elif k in ("group", "opt"):
        names_in(it[1], out)
    elif k in ("rep", "look", "eager"):
        names_in_item(it[1], out)
    elif k == "gather":
        names_in_item(it[1], out)
        names_in_item(it[2], out)


# ───────────────────────── emitting ─────────────────────────
def q(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def emit_alts(alts, tm):
    return " | ".join(" ".join(emit_item(i, tm) for i in a) for a in alts)


def emit_atom(it, tm):
    s = emit_item(it, tm)
    return s if it[0] in ("name", "lit", "group", "opt") else f"({s})"


def emit_item(it, tm):
    k = it[0]
    if k == "name":
        return tm.get(it[1], it[1])
    if k == "lit":
        return q(it[1])
    if k == "cut":
        return "~"
    if k == "group":
        return f"({emit_alts(it[1], tm)})"
    if k == "opt":
        return f"({emit_alts(it[1], tm)})?"
    if k == "rep":
        base = it[1]
        b = emit_item(base, tm) if base[0] in ("name", "lit", "group") else f"({emit_item(base, tm)})"
        if base[0] == "opt":                              # (x)? followed by a quantifier
            b = f"({emit_item(base, tm)})"
        return b + it[2]
    if k == "look":
        return ("&" if it[2] else "!") + emit_atom(it[1], tm)
    if k == "eager":
        return emit_atom(it[1], tm)                      # best effort: plain match
    if k == "gather":
        return f"{emit_atom(it[2], tm)} % {emit_atom(it[1], tm)}"
    raise ConvError(f"cannot emit {it!r}")


# ───────────────────────── public API ─────────────────────────
def convert(text, start="file", layer="python", token_map=None, drop_tokens=(), drop_rules=(),
            fixes=None, flags="strict peg", report=None):
    """pegen grammar text -> parsekit grammar text (one peg layer).

    fixes   {old: new} plain substring repairs applied before parsing (for damaged sources)
    report  optional dict filled with: undefined (names with no rule/token), dropped, rules
    """
    tm = dict(token_map or {})
    for old, new in (fixes or {}).items():
        if old not in text:
            raise ConvError(f"fix target not found: {old!r}")
        text = text.replace(old, new)
    rules = {}
    order = []
    for name, body in split_rules(strip_comments(text)):
        alts = P(body, name).alts()
        rules[name] = alts
        order.append(name)
    pr = Pruner(set(drop_tokens) | set(drop_rules))
    for n in order:
        rules[n] = pr.alts(rules[n])
    if start not in rules:
        raise ConvError(f"start rule {start!r} not found")
    reach, todo = set(), [start]
    while todo:
        n = todo.pop()
        if n in reach or n not in rules:
            continue
        reach.add(n)
        refs = set()
        names_in(rules[n], refs)
        todo += [r for r in refs if r in rules]
    undefined = set()
    for n in reach:
        refs = set()
        names_in(rules[n], refs)
        undefined |= {r for r in refs if r not in rules and r not in tm and not r.isupper()}
    if report is not None:
        report.update(undefined=sorted(undefined), rules=len(reach),
                      dropped=sorted(set(order) - reach))
    out = [f"layer {layer} {flags}:"]
    for n in [start] + [x for x in order if x in reach and x != start]:
        alts = rules[n]
        for i, a in enumerate(alts, 1):
            label = f"/a{i}" if len(alts) > 1 else ""
            out.append(f"  {n}{label}: " + " ".join(emit_item(it, tm) for it in a))
    return "\n".join(out) + "\n"
