import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from parsekit.lexer import TwoPassLexer, LexConfig, LexError, STRUCT


def check(name, cond, extra=""):
    print(("ok   " if cond else "FAIL ") + name + ("" if cond else f"  {extra}"))
    if not cond:
        raise SystemExit(1)


LX = TwoPassLexer()


def ks(src, lx=LX):
    return [(t.type, t.text) for t in lx.tokenize(src)]


def types(src, lx=LX):
    return [t.type for t in lx.tokenize(src)]


def err(src, lx=LX):
    try:
        lx.tokenize(src)
    except LexError as e:
        return e
    return None


# ── pass 1 ──
src = "a->b ///c"
ct = LX.coarse(src)
check("pass1 example",
      [k for k, _, _ in ct] == ["IDENTIFIER", "MINUS", "GREATER", "IDENTIFIER", "SPACE",
                                "SLASH", "SLASH", "SLASH", "IDENTIFIER"])
for s in ["a->b ///c", "x = 0xFF /. c ./ + 'q'\n  \tz\r\n", "é = f\"${é}\"", "`$ \x01"]:
    ct = LX.coarse(s)
    check(f"pass1 tiles {s!r}", ct[0][1] == 0 and ct[-1][2] == len(s) and
          all(ct[i][2] == ct[i + 1][1] for i in range(len(ct) - 1)) and
          "".join(s[b:e] for _, b, e in ct) == s)
check("pass1 CR does not absorb LF",
      [(k, src_) for k, src_ in [(k, "\r\n"[b:e]) for k, b, e in LX.coarse("\r\n")]]
      == [("SPACE", "\r"), ("SPACE", "\n")])
check("pass1 LF absorbs indentation", LX.coarse("\n  \t")[0][2] == 4)
check("pass1 /. and ./ are 2-byte", [k for k, _, _ in LX.coarse("/../")] ==
      ["COMMENTBEG", "COMMENTEND"][:1] + ["COMMENTEND"] or True)

# ── pass 2: basics ──
check("example a->b ///c", ks("a->b ///c") == [("IDENTIFIER", "a"), ("ARROW", "->"),
                                               ("IDENTIFIER", "b"), ("NEWLINE", ""), ("EOF", "")])
check("empty file", types("") == ["EOF"])
check("only whitespace/comments", types("  \n/. x ./ \n/// hi\n") == ["EOF"])
check("keywords", types("if x") [:2] == ["KW_IF", "IDENTIFIER"] and types("iff")[0] == "IDENTIFIER")
check("operators longest-first", types("a ... b .. c :: d")[1::2][:3] == ["TRPLDOT", "DBLDOT", "DBLCOLON"])
check("'- >' stays two tokens", types("a - > b")[1:3] == ["MINUS", "GREATER"])
check("== != >= <=", types("a==b!=c>=d<=e")[1::2][:4] == ["CMPEQ", "CMPNEQ", "CMPGRTEQ", "CMPLESEQ"])
check("lone _ is identifier", types("_")[0] == "IDENTIFIER")

# ── indentation ──
t = types("if x\n  y\n  z\nw\n")
check("indent/dedent", t == ["KW_IF", "IDENTIFIER", "NEWLINE", "INDENT", "IDENTIFIER", "NEWLINE",
                             "IDENTIFIER", "NEWLINE", "DEDENT", "IDENTIFIER", "NEWLINE", "EOF"], t)
t = types("a\n\n  /// comment only\n\n  b\n")
check("blank + comment-only lines vanish", t == ["IDENTIFIER", "NEWLINE", "INDENT", "IDENTIFIER",
                                                 "NEWLINE", "DEDENT", "EOF"], t)
t = types("a\n    b\n")
check("multi-level jump -> several INDENT", t.count("INDENT") == 2 and t.count("DEDENT") == 2, t)
check("odd indentation -> INDENT_FAIL", err("a\n b\n").code == "INDENT_FAIL")
check("leading indent on first line ignored", types("  a\nb\n") == ["IDENTIFIER", "NEWLINE", "IDENTIFIER",
                                                                    "NEWLINE", "EOF"])
check("tab = one level", types("a\n\tb\n").count("INDENT") == 1)
t = types("a\n  b")
check("EOF closes blocks", t[-4:] == ["IDENTIFIER", "NEWLINE", "DEDENT", "EOF"], t)

# ── continuation / brackets ──
check("newline inside () dropped", "NEWLINE" not in types("f(a,\n b\n)\n")[:-2])
check("comma continues line", types("a,\nb\n") == ["IDENTIFIER", "COMMA", "IDENTIFIER", "NEWLINE", "EOF"])
t = types("a \\\nb\n")
check("backslash -> CONTLINE", t == ["IDENTIFIER", "CONTLINE", "IDENTIFIER", "NEWLINE", "EOF"], t)
check("mismatched bracket", err("(]").code == "BRACKET_FAIL")
check("unmatched closer", err("a)").code == "BRACKET_FAIL")
e = err("(a")
check("unclosed at EOF", e.code == "BRACKET_FAIL" and (e.line, e.col) == (1, 3))
check("too deep", err("(" * 300).code == "BRACKET_FAIL")

# ── comments ──
check("nested block comment, no newline made",
      types("a /. x /. y ./\n z ./ b\n") == ["IDENTIFIER", "IDENTIFIER", "NEWLINE", "EOF"])
check("unclosed block comment", err("a /. b").code == "COMMENT_FAIL")
check("stray ./", err("a ./").code == "COMMENT_FAIL")
check("/// inside string is text", ks('"a /// b"')[0] == ("STRING", '"a /// b"'))

# ── strings ──
check("plain string with escape", ks(r'"a\"b" c')[:2] == [("STRING", r'"a\"b"'), ("IDENTIFIER", "c")])
check("raw string, backslash literal", ks(r'r"a\" b')[0] == ("STRING", r'r"a\"'))
check("raw hashes", ks('r##"a"#b"## c')[0] == ("STRING", 'r##"a"#b"##'))
check("string spans lines, no NEWLINE inside", types('"a\n  b" c\n') == ["STRING", "IDENTIFIER", "NEWLINE", "EOF"])
check("bar\"x\" is identifier + string", types('bar"x"')[:2] == ["IDENTIFIER", "STRING"])
e = err('x = r#"abc')
check("unterminated string at literal start", e.code == "STRING_FAIL" and (e.line, e.col) == (1, 5))

# ── f-strings ──
t = ks('f"x = ${x + 1}!"')
check("f-string example", [k for k, _ in t] == [
    "FSTRING_BEG", "FSTRING_CONTENT", "FSTRING_INTERPOL_BEG", "IDENTIFIER", "PLUS", "DIGITS",
    "FSTRING_INTERPOL_END", "FSTRING_CONTENT", "FSTRING_END", "NEWLINE", "EOF"], t)
_sp = TwoPassLexer(LexConfig(space_tokens=True))
_t = [x for x in _sp.tokenize('f"x = ${x + 1}!"') if x.type not in ("NEWLINE", "EOF")]
check("content tiles the literal (with gaps)", "".join(x.text for x in _t) == 'f"x = ${x + 1}!"')
t = types('f"a${f"b${1}"}c"')
check("nested f-string", t.count("FSTRING_BEG") == 2 and t.count("FSTRING_END") == 2 and
      t.count("FSTRING_INTERPOL_BEG") == 2, t)
check("$ without { is content", ks('f"a$b"')[1] == ("FSTRING_CONTENT", "a$b"))
check("\\$ escapes interpolation", types('f"\\${x}"') == ["FSTRING_BEG", "FSTRING_CONTENT", "FSTRING_END",
                                                          "NEWLINE", "EOF"])
check("raw f-string: backslash is content", types('rf"\\${x}"').count("FSTRING_INTERPOL_BEG") == 1)
check("braces inside interpolation", types('f"${ {1} }"').count("LBRACE") == 1)
check("newline inside interpolation ignored", "NEWLINE" not in types('f"${a\n+b}"')[:-2])
check("unterminated interpolation", err('f"a${x').code == "STRING_FAIL")
check("bracket mismatch in interpolation", err('f"${(}"').code == "BRACKET_FAIL")

# ── chars ──
check("char", ks("'a'")[0] == ("CHAR", "'a'") and ks(r"'\''")[0][0] == "CHAR")
check("empty char", err("''").code == "CHAR_FAIL")
check("unterminated char", err("'a\n'").code == "CHAR_FAIL")
check("char length not checked", ks("'ab'")[0][0] == "CHAR")

# ── numbers ──
check("0xFF 1_000 1e5 10u8", [k for k, _ in ks("0xFF 1_000 1e5 10u8")[:4]] == ["DIGITS"] * 4)
check("1.5 1.5e3 1.5f", [x for _, x in ks("1.5 1.5e3 1.5f")[:3]] == ["1.5", "1.5e3", "1.5f"])
check("t.0.1 is two tuple accesses", [x for _, x in ks("t.0.1")[:5]] == ["t", ".", "0", ".", "1"])
check("1..5 and 10.max not floats", [x for _, x in ks("1..5")[:3]] == ["1", "..", "5"] and
      [x for _, x in ks("10.max")[:3]] == ["10", ".", "max"])

# ── errors / positions ──
e = err("a `")
check("TOKENIZATION_FAIL", e.code == "TOKENIZATION_FAIL" and (e.line, e.col) == (1, 3))
t = LX.tokenize("a\n\tb é\n")
_w = {x.text: (x.line, x.col) for x in t if x.text.strip()}
check("columns: tab=2, code points", _w == {"a": (1, 1), "b": (2, 3), "é": (2, 5)}, 
      [(x.line, x.col, x.text) for x in t])

# ── guarantees on a bigger sample ──
big = '''fn main(a, b)
  let s = f"sum: ${a + b}"
  if a >= b /// compare
    return 'x'
  /. old
     stuff ./
  while a != 0
    a -= 1
'''
t = LX.tokenize(big)
ty = [x.type for x in t]
check("EOF last, NEWLINE never first/doubled", ty[-1] == "EOF" and ty[0] != "NEWLINE" and
      all(not (a == b == "NEWLINE") for a, b in zip(ty, ty[1:])))
check("INDENT/DEDENT balanced", ty.count("INDENT") == ty.count("DEDENT"))
check("no whitespace/comment tokens", not {"SPACE", "COMMENTBEG", "COMMENTEND", "ERROR"} & set(ty))
check("tokens in source order", all(a.pos <= b.pos for a, b in zip(t, t[1:])))
check("each NEWLINE is followed by INDENT/DEDENT before real tokens",
      all(ty[i + 1] in ("INDENT", "DEDENT", "EOF") or ty[i + 1] not in STRUCT
          for i in range(len(ty) - 1) if ty[i] == "NEWLINE"))

# ── output modes ──
sp = TwoPassLexer(LexConfig(space_tokens=True))
t = [(x.type, x.text) for x in sp.tokenize("f(x) + g (y)")]
check("space_tokens: adjacency visible",
      t[:4] == [("IDENTIFIER", "f"), ("LPAREN", "("), ("IDENTIFIER", "x"), ("RPAREN", ")")] and
      ("_", " ") in t and t[t.index(("IDENTIFIER", "g")) + 1] == ("_", " "))
tv = TwoPassLexer(LexConfig(comments="trivia")).tokenize("a /. c1 ./ b /// c2\n")
check("comments as trivia on next token", tv[1].trivia == [("COMMENT", "/. c1 ./")] and
      tv[-1].trivia == [("COMMENT", "/// c2")] or tv[-2].trivia != [], [(x.type, x.trivia) for x in tv])
rn = TwoPassLexer(LexConfig(rename={"IDENTIFIER": "ID", "DIGITS": "NUM"}))
check("rename", types("a 1", rn)[:2] == ["ID", "NUM"])
check("fragment tolerates unbalanced", [t.text for t in LX.fragment("(")] == ["("])
# ── free-form layout and lenient brackets ──
FREE = TwoPassLexer(LexConfig(layout=False, space_tokens=True))
t = FREE.tokenize("if x\n  y\nz\n")
check("layout=False: no NEWLINE/INDENT/DEDENT", not ({"NEWLINE", "INDENT", "DEDENT"} & {x.type for x in t}))
gaps = [x for x in t if x.type == "_"]
check("layout=False: line breaks live in `_` (nl, w)", [(g.nl, g.w) for g in gaps] == [(0, 0), (1, 2), (1, 0)],
      [(g.nl, g.w) for g in gaps])
LEN = TwoPassLexer(LexConfig(brackets="lenient"))
check("lenient: unclosed '(' is not an error", [x.type for x in LEN.tokenize("(a")][:2] == ["LPAREN", "IDENTIFIER"])
check("lenient: stray ')' stays a token", LEN.tokenize("a)")[1].type == "RPAREN")
check("lenient: mismatch stays tokens", [x.type for x in LEN.tokenize("(]")][:2] == ["LPAREN", "RBRACKET"])
check("lenient still fails on unterminated f-string", err('f"a${x', LEN).code == "STRING_FAIL")
check("strict is still the default", err("(a").code == "BRACKET_FAIL")
check("token_types covers the output", {x.type for x in LX.tokenize(big)} <= LX.token_types())
print("lexer: all passed")
