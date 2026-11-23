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
    # ---- options for syntaxes other than the one in the spec (see python_config()) ----
    block_comment: tuple = ("/.", "./")            # None disables block comments
    quote_strings: bool = False                    # ' starts a string (else a char literal)
    triple_quotes: bool = False                    # """...""" and '''...'''
    prefix_ci: bool = False                        # string prefixes are case-insensitive
    indent: str = "fixed"                          # "fixed" (spec) | "stack" (arbitrary widths)
    indent_tab: int = 0                            # tab stop for indentation (0 = `tab`)
    py_numbers: bool = False                       # .5  1.  1e-5  0b1_0  1j
    fstring_braces: bool = False                   # {expr} fields, {{ }} escapes, format specs
    template_prefix: str = ""                      # letter that makes a string a t-string
    drop_continuation: bool = False                # remove the `\\` token of a continued line
    # ---- options used by the Light preset (see examples/light/light.py) ----
    block_nested: bool = True                      # block comments nest (spec lexer) or end at the first closer
    unicode_idents: bool = True                    # identifiers may contain non-ASCII letters
    symbols: dict = field(default_factory=dict)    # single non-ASCII symbols -> token type, e.g. {"->": ...}
    compound_keywords: dict = field(default_factory=dict)   # {"as": ("?", "!", "!!")}: `as?` is one token
    number_style: str = "spec"                     # "spec" | "python" | "light" (suffix literals, 0b/0o/0x)
    any_string_prefix: bool = False                # any identifier glued to a quote is a string prefix
    string_suffix: bool = False                    # identifier glued after the closing quote joins the token
    fstrings: bool = True                          # f / t prefixes switch on interpolation tokens
    raw_prefixes: tuple = None                     # prefixes meaning "raw" (default: any containing r)
    lifetimes: bool = False                        # 'name (not 'x') is a LIFETIME token
    asi: dict = None                               # automatic semicolon insertion rules
    raw_blocks: tuple = ()                         # (("extern", ("C", "ASM")),): opaque { ... } bodies
    raw_block_strings: tuple = ("C",)              # raw blocks whose strings/comments are skipped


PY_KEYWORDS = ("False None True and as assert async await break class continue def del elif else "
               "except finally for from global if import in is lambda nonlocal not or pass raise "
               "return try while with yield").split()
PY_OPERATORS = [(o, n) for o, n in [
    ("**=", "ASSIGNPOW"), ("//=", "ASSIGNFLOORDIV"), (">>=", "ASSIGNRSHIFT"), ("<<=", "ASSIGNLSHIFT"),
    ("...", "ELLIPSIS"), ("->", "ARROW"), (":=", "WALRUS"), ("**", "POW"), ("//", "FLOORDIV"),
    (">>", "RSHIFT"), ("<<", "LSHIFT"), ("<=", "CMPLESEQ"), (">=", "CMPGRTEQ"), ("==", "CMPEQ"),
    ("!=", "CMPNEQ"), ("+=", "ASSIGNADD"), ("-=", "ASSIGNSUB"), ("*=", "ASSIGNMUL"),
    ("/=", "ASSIGNDIV"), ("%=", "ASSIGNMOD"), ("@=", "ASSIGNMATMUL"), ("&=", "ASSIGNAND"),
    ("|=", "ASSIGNOR"), ("^=", "ASSIGNXOR")]]


def python_config(**overrides):
    """LexConfig for Python source: `#` comments, ' and " strings (triple, prefixed, f/t with
    {expr} fields), indentation stack, python numbers, python keywords and operators.
    Token types follow python.gram: NAME NUMBER STRING NEWLINE INDENT DEDENT FSTRING_START ..."""
    cfg = dict(
        tab=8, indent="stack", indent_tab=8, line_comment="#", block_comment=None,
        keywords={k: "KW_" + k.upper() for k in PY_KEYWORDS}, operators=PY_OPERATORS,
        cont_tokens=("BACKSLASH",), str_prefixes=("r", "u", "b", "br", "rb", "f", "fr", "rf", "t", "tr", "rt"),
        prefix_ci=True, raw_hash_max=0, quote_strings=True, triple_quotes=True, py_numbers=True,
        fstring_braces=True, template_prefix="t", brackets="lenient", drop_continuation=True,
        rename={"IDENTIFIER": "NAME", "DIGITS": "NUMBER", "FSTRING_BEG": "FSTRING_START",
                "FSTRING_CONTENT": "FSTRING_MIDDLE", "TSTRING_BEG": "TSTRING_START",
                "TSTRING_CONTENT": "TSTRING_MIDDLE"})
    cfg.update(overrides)
    return LexConfig(**cfg)


class LexError(SyntaxError):
    """Lexer failure with a spec error code and 1-based line/col."""

    def __init__(self, code, line, col, detail=""):
        super().__init__(f"{line}:{col}: {code}" + (f": {detail}" if detail else ""))
        self.code, self.line, self.col, self.detail = code, line, col, detail


# ───────────────────────── pass 1 ─────────────────────────
_DELIMS = re.escape("".join(DELIM))


def _coarse_rx(block, unicode_idents=True, symbols=None):
    hi = r"\u0080-\U0010ffff" if unicode_idents else ""
    parts = [rf"(?P<IDENTIFIER>[A-Za-z_{hi}][A-Za-z0-9_{hi}]*)",
             r"(?P<DIGITS>[0-9]+)",
             r"(?P<SPACE>[ \t\r\n][ \t]*)"]            # LF absorbs the next line's indentation
    if block:
        parts += [f"(?P<COMMENTBEG>{re.escape(block[0])})", f"(?P<COMMENTEND>{re.escape(block[1])})"]
    parts += [r"(?P<DELIM>[" + _DELIMS + r"])"]
    if symbols:
        parts.append("(?P<SYMBOL>[" + re.escape("".join(symbols)) + "])")
    parts.append(r"(?P<ERROR>.)")
    return re.compile("|".join(parts), re.S)


COARSE = _coarse_rx(("/.", "./"))
_D = r"[0-9](?:_?[0-9])*"
PYNUM = re.compile(r"0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+"
                   rf"|(?:{_D}\.(?:{_D})?|\.{_D})(?:[eE][+-]?{_D})?[jJ]?"
                   rf"|{_D}[eE][+-]?{_D}[jJ]?|{_D}[jJ]?")
RX_SPEC = re.compile(r"[{}]")
LNUM = re.compile(r"0x[0-9a-fA-F_]+|0o[0-7_]+|0b[01_]+"
                  r"|(?P<f>[0-9][0-9_]*\.[0-9][0-9_]*(?:[eE][+-]?[0-9]+)?|[0-9][0-9_]*[eE][+-]?[0-9]+)"
                  r"|[0-9][0-9_]*")
LSUFFIX = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)?")
RX_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
RX_PLAININT = re.compile(r"[0-9]+")

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
        self.opmap, self.rx_cache = {}, {}
        for op, kind in sorted(self.cfg.operators, key=lambda x: -len(x[0])):
            self.opmap.setdefault(op[0], []).append((op, kind))
        c = self.cfg
        self.coarse_rx = (COARSE if (c.block_comment == ("/.", "./") and c.unicode_idents and not c.symbols)
                          else _coarse_rx(c.block_comment, c.unicode_idents, c.symbols))

    # pass 1 ---------------------------------------------------------
    def coarse(self, src):
        return self.coarse_range(src, 0, len(src))

    def coarse_range(self, src, pos, end):
        out = []
        for m in self.coarse_rx.finditer(src, pos, end):
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
        if c.template_prefix:
            ts |= {"TSTRING_BEG", "TSTRING_CONTENT", "TSTRING_INTERPOL_BEG", "TSTRING_INTERPOL_END",
                   "TSTRING_END"}
        ts |= set(DELIM_NAMES) | {k for _, k in c.operators} | set(c.keywords.values())
        if c.number_style == "light":
            ts |= {"INT", "FLOAT", "INT_SUFFIX", "FLOAT_SUFFIX"}
        if c.lifetimes:
            ts.add("LIFETIME")
        if c.raw_blocks:
            ts.add("RAW_BLOCK")
        ts |= set(c.symbols.values())
        ts |= {f"KW_{k.upper()}{x}" for k, sufs in c.compound_keywords.items() for x in sufs}
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
        s.level, s.fs, s.trivia, s.istack = 0, [], [], [0]
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
        """Continue at source offset `off`. If that falls inside a pass-1 piece (a number
        regex may stop mid-identifier, as in `1e5x`), the rest of the piece is re-split."""
        i = bisect_left(s.starts, off)
        if i < s.n and s.starts[i] == off:
            s.i = i
            return
        if i > 0 and s.ct[i - 1][1] < off < s.ct[i - 1][2]:
            _, b, e = s.ct[i - 1]
            rest = s.lx.coarse_range(s.src, off, e)
            s.ct[i - 1:i] = rest
            s.starts[i - 1:i] = [r[1] for r in rest]
            s.n = len(s.ct)
            s.i = i - 1
            return
        s.i = i

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
            elif (cfg.line_comment and src[b] == cfg.line_comment[0] and kind in DELIM_NAMES
                  and src.startswith(cfg.line_comment, b)):
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
        if (s.out and s.cfg.asi and not s.fragment and not s.stack and s.asi_enders()
                and not (s.cfg.asi.get("decorators") and s.decorator_line())):
            s.emit("SEMICOLON", end, end).text = ";"              # EOF ends a statement too
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
                if s.cfg.block_nested or depth == 0:
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
    # ---- automatic semicolon insertion ----
    def asi_enders(s):
        a, last = s.cfg.asi, s.out[-1]
        return last.type in a["enders"] or (not last.kids and last.text in a["enders_text"])

    def decorator_line(s):
        """Does the line end with `@name` or `@name(...)`? (a decorator is not a statement)"""
        out, i = s.out, len(s.out) - 1
        if out[i].type == "RPAREN":
            depth = 0
            while i >= 0:
                if out[i].type == "RPAREN":
                    depth += 1
                elif out[i].type == "LPAREN":
                    depth -= 1
                    if depth == 0:
                        break
                i -= 1
            i -= 1
        return i >= 1 and out[i - 1].type == "AT"

    def asi_ok(s):
        a, c = s.cfg.asi, s.cnt
        if c["("] or c["["] or c["I"] or not s.asi_enders():
            return False
        if a.get("decorators") and s.decorator_line():
            return False
        kind, b, e = s.ct[s.i]
        if kind == "IDENTIFIER" and s.src[b:e] in a["continuers_words"]:
            return False
        text = s.src[b:e]
        for op, _k in s.lx.opmap.get(s.src[b], ()):
            if s.src.startswith(op, b):
                text = op
                break
        return text not in a["continuers_ops"]

    def cols_of(s, text):
        cfg = s.cfg
        tab = cfg.indent_tab or cfg.tab
        cols = 0
        for ch in text:
            if ch == "\t":
                cols = (cols // tab + 1) * tab if cfg.indent == "stack" else cols + cfg.tab
            else:
                cols += 1
        return cols

    def resolve(s, nb):
        cfg = s.cfg
        if not s.out and cfg.layout and cfg.indent == "stack":      # first token already indented
            ls = s.src.rfind("\n", 0, nb) + 1
            seg = s.src[ls:nb]
            if not seg.strip() and (cols := s.cols_of(seg)) > 0:
                s.istack.append(cols)
                s.level = len(s.istack) - 1
                s.emit("INDENT", nb, nb)
        if s.first_nl is None:
            return
        f, l = s.first_nl, s.last_nl
        s.first_nl = s.last_nl = None
        cfg, c = s.cfg, s.cnt
        if not s.out or not cfg.layout:                        # leading blank lines / free-form
            if s.out and cfg.asi and not s.fragment and s.asi_ok():
                fb = s.ct[f][1]
                s.emit("SEMICOLON", fb, fb).text = ";"
            return
        last = s.out[-1]
        if (c["I"] or (c["("] and cfg.cont_paren) or (c["["] and cfg.cont_bracket)
                or (c["{"] and cfg.cont_brace)):               # inside brackets
            if last.type == "BACKSLASH" and cfg.drop_continuation:
                s.out.pop()
            return
        if last.type in cfg.cont_tokens:                       # continuation token
            if last.type == "BACKSLASH":
                if cfg.drop_continuation:
                    s.out.pop()
                else:
                    last.type = "CONTLINE"
            return
        lb, le = s.ct[l][1], s.ct[l][2]
        cols = s.cols_of(s.src[lb + 1:le])
        if cfg.indent == "stack":
            s.emit("NEWLINE", s.ct[f][1], s.ct[f][1] + 1)
            st = s.istack
            if cols > st[-1]:
                st.append(cols)
                s.emit("INDENT", nb, nb)
            else:
                while cols < st[-1]:
                    st.pop()
                    s.emit("DEDENT", nb, nb)
                if cols != st[-1]:
                    s.fail("INDENT_FAIL", nb, "unindent does not match any outer indentation level")
            s.level = len(st) - 1
            return
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
        if kind == "DIGITS" or (kind == "DOT" and s.cfg.py_numbers and src[b + 1:b + 2].isdigit()):
            if s.cfg.number_style == "light":
                return s.number_light()
            return s.number_py() if (s.cfg.py_numbers or s.cfg.number_style == "python") else s.number()
        if kind == "SYMBOL":
            s.emit(s.cfg.symbols[src[b]], b, e)
            s.i += 1
            return
        if kind == "LBRACE" and s.cfg.raw_blocks and len(s.out) >= 2 and not s.out[-1].kids:
            for kw, names in s.cfg.raw_blocks:
                if s.out[-2].text == kw and s.out[-1].text in names:
                    return s.raw_block(s.out[-1].text, b, e)
        if kind == "ERROR":
            s.fail("TOKENIZATION_FAIL", b, f"unexpected {src[b]!r}")
        if kind == "DOUBLEQUOTE":
            return s.string(i, "", 0, i)
        if kind == "QUOTE":
            return s.string(i, "", 0, i) if s.cfg.quote_strings else s.char()
        if (kind == "COLON" and s.cfg.fstring_braces and s.stack
                and isinstance(s.stack[-1], tuple)):           # start of a format spec
            s.emit(kind, b, e)
            return s.fspec(s.stack[-1][1], e)
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
                s.emit(top[1]["kind"] + "STRING_INTERPOL_END", b, e)
                return s.fspec(top[1], e) if top[2] else s.fcontent(top[1], e)
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
        if text in cfg.compound_keywords:                      # `as?`  `let!`  `where!!`
            for suf in sorted(cfg.compound_keywords[text], key=len, reverse=True):
                if src.startswith(suf, e):
                    s.emit(f"KW_{text.upper()}{suf}", b, e + len(suf))
                    return s.seek(e + len(suf))
        if (cfg.any_string_prefix and text not in cfg.keywords and i + 1 < s.n
                and ct[i + 1][0] == "DOUBLEQUOTE"):
            return s.string(i, text, 0, i + 1)
        low = text.lower() if cfg.prefix_ci else text
        if low in cfg.str_prefixes:
            j, h = i + 1, 0
            if "r" in low:                                     # hashes only after raw prefixes
                while j < s.n and ct[j][0] == "HASH" and h < cfg.raw_hash_max:
                    h += 1
                    j += 1
            if j < s.n and (ct[j][0] == "DOUBLEQUOTE" or (ct[j][0] == "QUOTE" and cfg.quote_strings)):
                return s.string(i, low, h, j)
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

    def number_light(s):
        """0x 0o 0b ints, floats, and suffix literals: 5ms  1.5km  123std::bigi"""
        src = s.src
        _, b, _e = s.ct[s.i]
        if s.out and s.out[-1].type == "DOT" and s.out[-1].pos + 1 == b:      # t.0.1: plain integers
            end, typ = RX_PLAININT.match(src, b).end(), "INT"
        else:
            m = LNUM.match(src, b)
            end, typ = m.end(), ("FLOAT" if m.lastgroup == "f" else "INT")
            sm = LSUFFIX.match(src, end)
            if sm:
                end, typ = sm.end(), typ + "_SUFFIX"
        s.emit(typ, b, end)
        s.seek(end)

    def raw_block(s, name, b, e):
        """`extern C { ... }`: the body is opaque foreign code, emitted as one RAW_BLOCK token."""
        src, n = s.src, len(s.src)
        skip_lits = name in s.cfg.raw_block_strings
        depth, p = 1, e
        while p < n and depth:
            ch = src[p]
            if skip_lits and src.startswith("//", p):
                q = src.find("\n", p)
                p = n if q < 0 else q
                continue
            if skip_lits and src.startswith("/*", p):
                q = src.find("*/", p + 2)
                p = n if q < 0 else q + 2
                continue
            if skip_lits and ch in "\"'":
                q = p + 1
                while q < n and src[q] != ch:
                    q += 2 if src[q] == "\\" else 1
                p = q + 1
                continue
            depth += (ch == "{") - (ch == "}")
            p += 1
        if depth:
            s.fail("BRACKET_FAIL", b, "unclosed foreign block")
        close = p - 1
        s.emit("LBRACE", b, e)
        if close > e:
            s.emit("RAW_BLOCK", e, close)
        s.emit("RBRACE", close, close + 1)
        s.seek(close + 1)

    def number_py(s):
        _, b, _e = s.ct[s.i]
        m = PYNUM.match(s.src, b)
        s.emit("DIGITS", b, m.end())
        s.seek(m.end())

    # ---- strings ----
    def string(s, i0, prefix, hashes, qidx):
        ct, src, cfg = s.ct, s.src, s.cfg
        b, qk, qe = ct[i0][1], ct[qidx][0], ct[qidx][2]
        q = src[ct[qidx][1]]
        triple = False
        if cfg.triple_quotes and qidx + 2 < s.n and ct[qidx + 1][0] == qk and ct[qidx + 2][0] == qk:
            triple, qe = True, ct[qidx + 2][2]
        raw = (prefix in cfg.raw_prefixes) if cfg.raw_prefixes is not None else "r" in prefix
        fmt = cfg.fstrings and "f" in prefix
        tmpl = cfg.fstrings and bool(cfg.template_prefix) and cfg.template_prefix in prefix
        if fmt or tmpl:
            fr = {"start": b, "raw": raw, "hashes": hashes, "q": q * (3 if triple else 1),
                  "triple": triple, "kind": "T" if tmpl else "F"}
            s.emit(fr["kind"] + "STRING_BEG", b, qe)
            s.fs.append(fr)
            return s.fcontent(fr, qe)
        close, pos = q * (3 if triple else 1) + "#" * hashes, qe
        if raw and not cfg.quote_strings:                      # spec raw strings: no escapes at all
            k = src.find(close, pos)
            if k < 0:
                s.fail("STRING_FAIL", b, "unterminated string")
            endo = k + len(close)
        else:
            single = cfg.quote_strings and not triple          # no raw newline in 'x' / "x"
            key = ("str", q, triple, single)
            rx = s.lx.rx_cache.get(key)
            if rx is None:
                rx = s.lx.rx_cache[key] = re.compile(
                    r"\\.|" + re.escape(q * (3 if triple else 1)) + (r"|\n" if single else ""), re.S)
            while True:
                m = rx.search(src, pos)
                if not m or m.group() == "\n":
                    s.fail("STRING_FAIL", b, "unterminated string")
                pos = m.end()
                if m.group()[0] != "\\" and src.startswith("#" * hashes, pos):
                    endo = pos + hashes
                    break
        if cfg.string_suffix:                                  # "123"BigInt
            sm = RX_IDENT.match(src, endo)
            if sm:
                endo = sm.end()
        s.emit("STRING", b, endo)
        s.seek(endo)

    def frx(s, fr):
        key = ("f", fr["q"], fr["triple"], fr["raw"])
        rx = s.lx.rx_cache.get(key)
        if rx is None:
            parts = [] if fr["raw"] else [r"\\N\{[^}\n]*\}"]
            parts += [r"\{\{", r"\}\}", r"\{", r"\\[^{}]", r"\\(?=[{}])", re.escape(fr["q"])]
            if not fr["triple"]:
                parts.append(r"\n")
            rx = s.lx.rx_cache[key] = re.compile("|".join(parts), re.S)
        return rx

    def fcontent_py(s, fr, pos):
        """PEP 701 style: literal text up to `{` (field), `{{`/`}}` are literal braces."""
        src, kind = s.src, fr["kind"]
        rx, cstart = s.frx(fr), pos
        while True:
            m = rx.search(src, pos)
            if not m or m.group() == "\n":
                s.fail("STRING_FAIL", s.fs[0]["start"], "unterminated format string")
            g = m.group()
            if g in ("{{", "}}") or g[0] == "\\":
                pos = m.end()
                continue
            if m.start() > cstart:
                s.emit(kind + "STRING_CONTENT", cstart, m.start())
            if g == "{":
                if len(s.stack) >= s.cfg.max_nesting:
                    s.fail("BRACKET_FAIL", m.start(), "nesting too deep")
                s.emit(kind + "STRING_INTERPOL_BEG", m.start(), m.end())
                s.stack.append(("I", fr, False))
                s.cnt["I"] += 1
                return s.seek(m.end())
            s.emit(kind + "STRING_END", m.start(), m.end())     # closing quote
            s.fs.pop()
            return s.seek(m.end())

    def fspec(s, fr, pos):
        """Format-spec text after the top-level `:` of a field: up to `}` or a nested `{`."""
        src, kind = s.src, fr["kind"]
        m = RX_SPEC.search(src, pos)
        if not m:
            s.fail("STRING_FAIL", s.fs[0]["start"], "unterminated format string")
        if m.start() > pos:
            s.emit(kind + "STRING_CONTENT", pos, m.start())
        if m.group() == "{":
            s.emit(kind + "STRING_INTERPOL_BEG", m.start(), m.end())
            s.stack.append(("I", fr, True))
            s.cnt["I"] += 1
            return s.seek(m.end())
        top = s.stack.pop()                                     # `}` ends the current field
        s.cnt["I"] -= 1
        s.emit(kind + "STRING_INTERPOL_END", m.start(), m.end())
        return s.fspec(fr, m.end()) if top[2] else s.fcontent(fr, m.end())

    def fcontent(s, fr, pos):
        """Scan f-string literal text from `pos` up to `${` or the closing quote."""
        if s.cfg.fstring_braces:
            return s.fcontent_py(fr, pos)
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
                s.stack.append(("I", fr, False))
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
        if s.cfg.lifetimes and p < len(src):                   # 'static  'a   (but 'x' is a char)
            c0 = src[p]
            if c0 != "\\" and src[p + 1:p + 2] != "'" and c0.isascii() and (c0.isalpha() or c0 == "_"):
                m = RX_IDENT.match(src, p)
                s.emit("LIFETIME", b, m.end())
                return s.seek(m.end())
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
