"""
lexer2: two-pass lexer (Python port of the principles in the Lexer Specification).

  pass 1  coarse():    source -> contiguous, context-free pieces
                       (IDENTIFIER, DIGITS, SPACE, COMMENTBEG, COMMENTEND, delimiters, ERROR)
  pass 2  tokenize():  pieces -> final stream; context decides what the pieces mean:
                       comments, strings (raw / hashes), f-strings, chars, numbers,
                       operators, keywords, brackets, NEWLINE / INDENT / DEDENT.

Guarantees (tested): pass 1 tiles the source exactly; pass 2 balances brackets,
INDENT/DEDENT and FSTRING_BEG/END; NEWLINE is never first and never doubled;
whitespace and comments never reach the output.

Differences from the C spec: tokens are `tokrewrite.Tok` objects, there is no
ownership / array-sentinel machinery, and two optional output modes exist:
  space_tokens=True   put a `_` token (text = the gap) between adjacent real tokens
                      so grammars can use `_` / `!_` (see tokrewrite)
  comments="trivia"   attach skipped comments to the next token's `.trivia`
  brackets="lenient"  unbalanced brackets are NOT lexer errors: they stay plain tokens and
                      the parser (PSP patterns + recovery layer) reports them. The default
                      "strict" gives the spec behavior (BRACKET_FAIL).
  layout=False        free-form mode: no NEWLINE / INDENT / DEDENT at all (newlines
                      stay visible only through the `nl` count of `_` gap tokens)
"""
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field

from .tokens import Tok

# ───────────────────────── configuration ─────────────────────────
DELIM = {
    "!": "BANG", '"': "DOUBLEQUOTE", "#": "HASH", "$": "DOLLAR", "%": "PERCENT",
    "&": "AMP", "'": "QUOTE", "(": "LPAREN", ")": "RPAREN", "*": "STAR", "+": "PLUS",
    ",": "COMMA", "-": "MINUS", ".": "DOT", "/": "SLASH", ":": "COLON", ";": "SEMICOLON",
    "<": "LESS", "=": "ASSIGN", ">": "GREATER", "?": "QUESTION", "@": "AT",
    "[": "LBRACKET", "\\": "BACKSLASH", "]": "RBRACKET", "^": "CARET", "{": "LBRACE",
    "|": "PIPE", "}": "RBRACE", "~": "TILDE",
}                                         # a backtick or control byte is TOKEN ERROR
DELIM_NAMES = frozenset(DELIM.values())

OPERATORS = [("...", "TRPLDOT"), ("..", "DBLDOT"), ("::", "DBLCOLON"), ("->", "ARROW"),
             ("=>", "DBLARROW"), ("++", "INCREMENT"), ("--", "DECREMENT"),
             ("+=", "ASSIGNADD"), ("-=", "ASSIGNSUB"), ("*=", "ASSIGNMUL"),
             ("/=", "ASSIGNDIV"), ("%=", "ASSIGNMOD"), ("==", "CMPEQ"), ("!=", "CMPNEQ"),
             (">=", "CMPGRTEQ"), ("<=", "CMPLESEQ")]          # longest first

KEYWORDS = ("if else while for return fn let const struct enum import true false nil "
            "break continue match pub priv prot static").split()

STRUCT = frozenset({"NEWLINE", "INDENT", "DEDENT", "EOF"})


@dataclass
class LexConfig:
    tab: int = 2                                   # columns per tab AND per indent level
    keywords: dict = field(default_factory=lambda: {k: "KW_" + k.upper() for k in KEYWORDS})
    operators: list = field(default_factory=lambda: list(OPERATORS))
    line_comment: str = "///"
    cont_tokens: tuple = ("BACKSLASH", "COMMA")    # last token that continues a line
    cont_paren: bool = True
    cont_bracket: bool = True
    cont_brace: bool = True
    str_prefixes: tuple = ("r", "f", "rf", "fr")
    raw_hash_max: int = 256
    max_nesting: int = 256
    comments: str = "drop"                         # "drop" | "trivia"
    layout: bool = True                            # emit NEWLINE / INDENT / DEDENT
    brackets: str = "strict"                       # "strict" | "lenient"
    space_tokens: bool = False
    rename: dict = field(default_factory=dict)     # e.g. {"IDENTIFIER": "ID"}


class LexError(SyntaxError):
    """Lexer failure with a spec error code and 1-based line/col."""

    def __init__(self, code, line, col, detail=""):
        super().__init__(f"{line}:{col}: {code}" + (f": {detail}" if detail else ""))
        self.code, self.line, self.col, self.detail = code, line, col, detail


# ───────────────────────── pass 1 ─────────────────────────
_DELIMS = re.escape("".join(DELIM))
COARSE = re.compile(
    r"(?P<IDENTIFIER>[A-Za-z_\u0080-\U0010ffff][A-Za-z0-9_\u0080-\U0010ffff]*)"
    r"|(?P<DIGITS>[0-9]+)"
    r"|(?P<SPACE>[ \t\r\n][ \t]*)"            # LF absorbs the next line's indentation
    r"|(?P<COMMENTBEG>/\.)"
    r"|(?P<COMMENTEND>\./)"
    r"|(?P<DELIM>[" + _DELIMS + r"])"
    r"|(?P<ERROR>.)", re.S)

OPEN = {"LPAREN": ")", "LBRACKET": "]", "LBRACE": "}"}
OPENER = {"LPAREN": "(", "LBRACKET": "[", "LBRACE": "{"}
CLOSE = {"RPAREN": ")", "RBRACKET": "]", "RBRACE": "}"}
CLOSER_OPENER = {")": "(", "]": "[", "}": "{"}
RX_STR = re.compile(r'\\.|"', re.S)
RX_F_ESC = re.compile(r'\\.|\$\{|"', re.S)
RX_F_RAW = re.compile(r'\$\{|"')


class TwoPassLexer:
    def __init__(self, config=None):
        self.cfg = config or LexConfig()
        self.opmap = {}
        for op, kind in self.cfg.operators:
            self.opmap.setdefault(op[0], []).append((op, kind))

    # pass 1 ---------------------------------------------------------
    def coarse(self, src):
        out = []
        for m in COARSE.finditer(src):
            kind, (b, e) = m.lastgroup, m.span()
            out.append((DELIM[src[b]] if kind == "DELIM" else kind, b, e))
        return out

    # pass 2 ---------------------------------------------------------
    def tokenize(self, src, _fragment=False):
        p = _Pass2(self, src, self.coarse(src), _fragment)
        toks = p.run()
        if self.cfg.space_tokens:
            toks = self._spaces(src, toks, p)
        if self.cfg.rename:
            for t in toks:
                t.type = self.cfg.rename.get(t.type, t.type)
        return toks

    def fragment(self, text):
        """Tokenize a snippet that need not be balanced (used by the recovery layer)."""
        return [t for t in self.tokenize(text, _fragment=True)
                if t.type not in STRUCT and t.type != "_"]

    def token_types(self):
        """Every token type this lexer can produce (used by the grammar linter)."""
        c = self.cfg
        ts = {"IDENTIFIER", "DIGITS", "STRING", "CHAR", "NEWLINE", "INDENT", "DEDENT", "EOF",
              "CONTLINE", "FSTRING_BEG", "FSTRING_CONTENT", "FSTRING_INTERPOL_BEG",
              "FSTRING_INTERPOL_END", "FSTRING_END", "_"}
        ts |= set(DELIM_NAMES) | {k for _, k in c.operators} | set(c.keywords.values())
        return {c.rename.get(t, t) for t in ts}

    @staticmethod
    def _spaces(src, toks, p):
        out, prev = [], None
        for t in toks:
            if prev is not None and prev.type not in STRUCT and t.type not in STRUCT:
                end = prev.pos + len(prev.text)
                if end < t.pos:
                    gap = src[end:t.pos]
                    ln, col = p.pos(end)
                    ws = Tok("_", gap, end, ln, col)
                    ws.nl = gap.count("\n")
                    if ws.nl:
                        lead = re.match(r"[ \t]*", gap.rsplit("\n", 1)[-1]).group()
                        ws.w = sum(p.cfg.tab if ch == "\t" else 1 for ch in lead)
                    out.append(ws)
            out.append(t)
            prev = t
        return out


class _Pass2:
    def __init__(s, lx, src, ct, fragment):
        s.lx, s.cfg, s.src, s.ct, s.n, s.fragment = lx, lx.cfg, src, ct, len(ct), fragment
        s.starts = [c[1] for c in ct]
        s.ls = [0] + [i + 1 for i, ch in enumerate(src) if ch == "\n"]
        s.tabs = [i for i, ch in enumerate(src) if ch == "\t"]   # for O(log n) column maths
        s.out, s.i = [], 0
        s.first_nl = s.last_nl = None             # pending newline tokens (indices into ct)
        s.stack = []                              # expected closers, or ("I", fstring frame)
        s.cnt = {"(": 0, "[": 0, "{": 0, "I": 0}
        s.level, s.fs, s.trivia = 0, [], []
        s.lenient = fragment or s.cfg.brackets == "lenient"

    # ---- helpers ----
    def pos(s, off):
        ln = bisect_right(s.ls, off)
        ls = s.ls[ln - 1]
        ntab = bisect_left(s.tabs, off) - bisect_left(s.tabs, ls) if s.tabs else 0
        return ln, off - ls + 1 + (s.cfg.tab - 1) * ntab

    def fail(s, code, off, detail=""):
        raise LexError(code, *s.pos(off), detail)

    def emit(s, typ, b, e):
        ln, col = s.pos(b)
        t = Tok(typ, s.src[b:e], b, ln, col)
        if s.trivia:
            t.trivia, s.trivia = s.trivia, []
        s.out.append(t)
        return t

    def seek(s, off):
        s.i = bisect_left(s.starts, off)

    # ---- main loop ----
    def run(s):
        ct, src, cfg = s.ct, s.src, s.cfg
        while s.i < s.n:
            kind, b, e = ct[s.i]
            if kind == "SPACE":
                if src[b] == "\n":
                    if s.first_nl is None:
                        s.first_nl = s.i
                    s.last_nl = s.i
                s.i += 1
            elif kind == "COMMENTBEG":
                s.block_comment()
            elif kind == "COMMENTEND":
                s.fail("COMMENT_FAIL", b, "stray './'")
            elif kind == "SLASH" and src.startswith(cfg.line_comment, b):
                s.line_comment()
            else:
                s.resolve(b)
                s.significant()
        s.finish()
        return s.out

    def finish(s):
        src, end = s.src, len(s.src)
        if not s.fragment and s.fs:
            s.fail("STRING_FAIL", s.fs[0]["start"], "unterminated format string")
        if s.stack and not s.lenient:
            s.fail("BRACKET_FAIL", end, "unclosed bracket")
        if s.out and s.cfg.layout and s.out[-1].type != "NEWLINE":
            b = s.ct[s.first_nl][1] if s.first_nl is not None else end
            s.emit("NEWLINE", b, min(b + 1, end))
        s.first_nl = s.last_nl = None
        while s.level > 0:
            s.level -= 1
            s.emit("DEDENT", end, end)
        s.emit("EOF", end, end)

    # ---- comments ----
    def block_comment(s):
        start, depth = s.i, 0
        while s.i < s.n:
            k = s.ct[s.i][0]
            if k == "COMMENTBEG":
                depth += 1
            elif k == "COMMENTEND":
                depth -= 1
                if depth == 0:
                    s.i += 1
                    s.note(s.ct[start][1], s.ct[s.i - 1][2])
                    return
            s.i += 1
        s.fail("COMMENT_FAIL", s.ct[start][1], "unclosed '/.'")

    def line_comment(s):
        ct, src = s.ct, s.src
        b = ct[s.i][1]
        while s.i < s.n and not (ct[s.i][0] == "SPACE" and src[ct[s.i][1]] == "\n"):
            s.i += 1
        s.note(b, ct[s.i][1] if s.i < s.n else len(src))   # newline stays pending

    def note(s, b, e):
        if s.cfg.comments == "trivia":
            s.trivia.append(("COMMENT", s.src[b:e]))

    # ---- newline / indentation resolution (spec 4.4) ----
    def resolve(s, nb):
        if s.first_nl is None:
            return
        f, l = s.first_nl, s.last_nl
        s.first_nl = s.last_nl = None
        cfg, c = s.cfg, s.cnt
        if not s.out or not cfg.layout:                        # leading blank lines / free-form
            return
        if (c["I"] or (c["("] and cfg.cont_paren) or (c["["] and cfg.cont_bracket)
                or (c["{"] and cfg.cont_brace)):               # inside brackets
            return
        last = s.out[-1]
        if last.type in cfg.cont_tokens:                       # continuation token
            if last.type == "BACKSLASH":
                last.type = "CONTLINE"
            return
        lb, le = s.ct[l][1], s.ct[l][2]
        cols = sum(cfg.tab if ch == "\t" else 1 for ch in s.src[lb + 1:le])
        if cols % cfg.tab:
            s.fail("INDENT_FAIL", nb, f"indentation {cols} is not a multiple of {cfg.tab}")
        fb = s.ct[f][1]
        s.emit("NEWLINE", fb, fb + 1)
        lv = cols // cfg.tab
        while s.level < lv:
            s.level += 1
            s.emit("INDENT", nb, nb)
        while s.level > lv:
            s.level -= 1
            s.emit("DEDENT", nb, nb)

    # ---- significant tokens ----
    def significant(s):
        ct, src, i = s.ct, s.src, s.i
        kind, b, e = ct[i]
        if kind == "IDENTIFIER":
            return s.ident()
        if kind == "DIGITS":
            return s.number()
        if kind == "ERROR":
            s.fail("TOKENIZATION_FAIL", b, f"unexpected {src[b]!r}")
        if kind == "DOUBLEQUOTE":
            return s.string(i, "", 0, i)
        if kind == "QUOTE":
            return s.char()
        if kind in OPEN:
            if len(s.stack) >= s.cfg.max_nesting:
                s.fail("BRACKET_FAIL", b, "nesting too deep")
            s.stack.append(OPEN[kind])
            s.cnt[OPENER[kind]] += 1
            s.emit(kind, b, e)
            s.i += 1
            return
        if kind in CLOSE:
            want = CLOSE[kind]
            top = s.stack[-1] if s.stack else None
            if top is None or (not isinstance(top, tuple) and top != want) \
                    or (isinstance(top, tuple) and kind != "RBRACE"):
                if not s.lenient:
                    s.fail("BRACKET_FAIL", b, f"unmatched or mismatched {want!r}")
                s.emit(kind, b, e)                             # lenient: plain token
                s.i += 1
                return
            if isinstance(top, tuple):                         # closes an ${ interpolation
                s.stack.pop()
                s.cnt["I"] -= 1
                s.emit("FSTRING_INTERPOL_END", b, e)
                return s.fcontent(top[1], e)
            s.stack.pop()
            s.cnt[CLOSER_OPENER[want]] -= 1
            s.emit(kind, b, e)
            s.i += 1
            return
        for op, okind in s.lx.opmap.get(src[b], ()):           # operators: longest first
            L = len(op)
            if i + L <= s.n and all(
                    ct[i + k][0] in DELIM_NAMES and ct[i + k][2] - ct[i + k][1] == 1
                    and src[ct[i + k][1]] == op[k] for k in range(L)):
                s.emit(okind, b, ct[i + L - 1][2])
                s.i += L
                return
        s.emit(kind, b, e)
        s.i += 1

    def ident(s):
        ct, src, cfg, i = s.ct, s.src, s.cfg, s.i
        _, b, e = ct[i]
        text = src[b:e]
        if text in cfg.str_prefixes:
            j, h = i + 1, 0
            if "r" in text:                                    # hashes only after raw prefixes
                while j < s.n and ct[j][0] == "HASH" and h < cfg.raw_hash_max:
                    h += 1
                    j += 1
            if j < s.n and ct[j][0] == "DOUBLEQUOTE":
                return s.string(i, text, h, j)
        s.emit(cfg.keywords.get(text, "IDENTIFIER"), b, e)
        s.i += 1

    def number(s):
        ct, i = s.ct, s.i
        _, b, e = ct[i]
        j = i + 1
        if j < s.n and ct[j][0] == "IDENTIFIER":               # 0xFF 10u8 1e5 1_000
            e, j = ct[j][2], j + 1
        elif (j + 1 < s.n and ct[j][0] == "DOT" and ct[j + 1][0] == "DIGITS"
              and not (s.out and s.out[-1].type == "DOT" and s.out[-1].pos + 1 == b)):
            e, j = ct[j + 1][2], j + 2                         # 1.5   (but t.0.1 is t . 0 . 1)
            if j < s.n and ct[j][0] == "IDENTIFIER":           # 1.5e3 1.5f
                e, j = ct[j][2], j + 1
        s.emit("DIGITS", b, e)
        s.i = j

    # ---- strings ----
    def string(s, i0, prefix, hashes, qidx):
        ct, src = s.ct, s.src
        b, qe = ct[i0][1], ct[qidx][2]
        raw, fmt = "r" in prefix, "f" in prefix
        if fmt:
            fr = {"start": b, "raw": raw, "hashes": hashes}
            s.emit("FSTRING_BEG", b, qe)
            s.fs.append(fr)
            return s.fcontent(fr, qe)
        close, pos = '"' + "#" * hashes, qe
        if raw:
            k = src.find(close, pos)
            if k < 0:
                s.fail("STRING_FAIL", b, "unterminated string")
            endo = k + len(close)
        else:
            while True:
                m = RX_STR.search(src, pos)
                if not m:
                    s.fail("STRING_FAIL", b, "unterminated string")
                pos = m.end()
                if m.group() == '"' and src.startswith("#" * hashes, pos):
                    endo = pos + hashes
                    break
        s.emit("STRING", b, endo)
        s.seek(endo)

    def fcontent(s, fr, pos):
        """Scan f-string literal text from `pos` up to `${` or the closing quote."""
        src = s.src
        rx = RX_F_RAW if fr["raw"] else RX_F_ESC
        h, cstart = "#" * fr["hashes"], pos
        while True:
            m = rx.search(src, pos)
            if not m:
                s.fail("STRING_FAIL", s.fs[0]["start"], "unterminated format string")
            g = m.group()
            if g[0] == "\\":                                   # escape (also \$)
                pos = m.end()
                continue
            if g == "${":
                if m.start() > cstart:
                    s.emit("FSTRING_CONTENT", cstart, m.start())
                if len(s.stack) >= s.cfg.max_nesting:
                    s.fail("BRACKET_FAIL", m.start(), "nesting too deep")
                s.emit("FSTRING_INTERPOL_BEG", m.start(), m.end())
                s.stack.append(("I", fr))
                s.cnt["I"] += 1
                return s.seek(m.end())
            if src.startswith(h, m.end()):                     # closing quote (+ hashes)
                if m.start() > cstart:
                    s.emit("FSTRING_CONTENT", cstart, m.start())
                endo = m.end() + len(h)
                s.emit("FSTRING_END", m.start(), endo)
                s.fs.pop()
                return s.seek(endo)
            pos = m.end()

    def char(s):
        src, b = s.src, s.ct[s.i][1]
        p = b + 1
        while True:
            if p >= len(src) or src[p] in "\r\n":
                s.fail("CHAR_FAIL", b, "unterminated char literal")
            if src[p] == "\\":
                if src[p + 1:p + 2] in ("", "\r", "\n"):
                    s.fail("CHAR_FAIL", b, "unterminated char literal")
                p += 2
            elif src[p] == "'":
                break
            else:
                p += 1
        if p == b + 1:
            s.fail("CHAR_FAIL", b, "empty char literal")
        s.emit("CHAR", b, p + 1)
        s.seek(p + 1)
