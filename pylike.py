"""Tiny indentation-based language: the lexer's NEWLINE / INDENT / DEDENT drive blocks."""
from parsekit import Parser, TwoPassLexer, LexConfig

GRAMMAR = r'''
names IDENTIFIER
layer exprs:
  operators expr over DIGITS IDENTIFIER
    chain "<" ">" "==",
    left "+" "-",
    left "*" "/"
layer simple:
  assign: target=IDENTIFIER "=" value=(expr | IDENTIFIER | DIGITS) NEWLINE
  exprstmt: e=expr NEWLINE
layer blocks:
  suite: INDENT body=(assign | exprstmt | ifstmt)+ DEDENT
  ifstmt: "if" cond=(expr | IDENTIFIER | DIGITS) NEWLINE body=suite
layer top:
  module: body=(assign | exprstmt | ifstmt)+

recover:
  empty_block: "if" cond=* NEWLINE EOF
      => "the 'if' on line {line} has no indented body"
'''


def make_parser():
    return Parser(GRAMMAR, "module", lexer=TwoPassLexer(LexConfig(space_tokens=True, brackets="lenient")))
