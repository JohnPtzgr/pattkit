"""C-like subset: typedef / templates fill the name table, which decides `T * x;`
(declaration) vs `a * x;` (multiplication) and `List<int>` (generic) vs `a < b`."""
from parsekit import Parser, TwoPassLexer, LexConfig

GRAMMAR = r'''
names IDENTIFIER
stops ";" "{" "}"

layer decls:
  decl/typedef:  "typedef" IDENTIFIER name=IDENTIFIER as TYPENAME ";"
  decl/template: "template" name=IDENTIFIER as GENERIC_ID ";"
layer types:
  type/generic: GENERIC_ID "<" args=*?{kind = type} ">"
  type/plain:   TYPENAME
layer vars:
  vardecl: ty=type ptr="*"? name=IDENTIFIER ";"
layer expr:
  operators expr over DIGITS IDENTIFIER
    chain "<" ">" "<=" ">=",
    left "+" "-",
    left "*" "/"
layer stmts:
  exprstmt: e=(expr | IDENTIFIER | DIGITS) ";"
layer top:
  program: items=(decl | vardecl | exprstmt)+

recover:
  missing_semi: ty=type ptr="*"? name=IDENTIFIER EOF
      => "declaration of '{name}' is missing a terminating ';'"
'''


def make_parser():
    lx = TwoPassLexer(LexConfig(layout=False, space_tokens=True, brackets="lenient"))
    return Parser(GRAMMAR, "program", lexer=lx, builtins={"int": "TYPENAME"})
