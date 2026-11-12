# parsekit

A universal parser "source + grammar → AST": a two-pass lexer, bottom-up rewriting of the token
list by layers, a name table, an operator table, and a separate error-recovery layer.

```
Text  ─► Pass 1 (coarse tokens) ─► Pass 2 (context: comments, string, f-strings,
                                    numbers, operators, parens, NEWLINE/INDENT/DEDENT)
      ─► rewrite layers ─► single root token = AST
      ─► (if unsuccessful) recovery layer: error patterns + bounded edit search
```

## Quickstart

```python
from parsekit import Parser, sexp
from examples.json_parser import make_parser, to_python

p = make_parser()
print(to_python(p.parse('{"a": [1, 2.5, -3], "b": null}')))
p.parse("[1 2]")        # ParseError: ... possible fixes: 1. insert ',' after '1' ...
```

Custom grammar: `Parser(grammar_text, "start_rule", lexer=None)`. Lexer by default uses 2 passes,
with `_` tokens inbetween non-contiguous tokens (pattern can use `_` or `!_`).
`parse()` returns AST or raises `ParseError` (with `.diagnostic`); `parse_recover()` returns
`(ast, diagnostic)`, where AST was built using fixed tokens, if the bug was found.

## Lexer (`parsekit/lexer.py`)

Specs: pass 1 is CFG and splits text without whitespaces; pass 2 gives a nuance
(nested `/. .../`, `///`, strings with prefixes `r f rf fr` and `#`-guards, f-strings use a flat sequence
 `FSTRING_*`, `${ }` with nesting, symbols use `'a'`, numbers - `0xFF 1.5e3 t.0.1`,
operators «longest first», fixed indent/dedent steps, line continues with `,` and `\`).

| `LexConfig` | Meaning |
|---|---|
| `layout=False` | free syntax: no `NEWLINE/INDENT/DEDENT`; newlines use `_` (with `nl` > 0, `w`) |
| `brackets="lenient"` | unbalanced brackets are not erroneous: balance is checked strictly by BBS (`< ... >`), while a missing bracket can reach recovery layer. Default is  `"strict"` (raises `BRACKET_FAIL`); `Parser` uses `lenient` by default |
| `space_tokens=True` | a `_` token carrying the gap text between adjacent meaningful tokens |
| `comments="trivia"` | puts commnets into `.trivia` of the following token |
| `operators`, `keywords`, `rename` | extends tables, can rename tokens (`{"IDENTIFIER": "ID"}`) |

## Grammar Language

The full list of all pattern elements and directives is in the `parsekit/engine.py`. TLDR:

| Entry | Value |
|---|---|
| `NAME`, `NAME/label`, `"text"` | token by type/class, by label, by leaf text |
| `_`, `_{nl}`, `_{w=n}`, `!_` | whitespace token and conditions; in non-strict layers `_?` is implicitly present between elements |
| `A B`, `A\|B`, `A* A+ A? A*? A+?`, `X{n}`, `X % S` | sequence, choice, repetitions (greedy/lazy), exactly n times, list separated by S |
| `*`, `+`, `*?`, `+?`, `*{kind = T}` | substitution: arbitrary tokens, the slice must fold into `T` |
| `<OPEN … CLOSE>` | bracket pair (OPEN/CLOSE may be alternatives); contents fold in isolation |
| `name=elem`, `name=ID as CLASS` | capture; capture that declares a name in the current scope |
| `operators E over …` | `left right nonassoc prefix postfix chain apply`, levels from weak to strong |
| `names`, `unknown`, `stops` | classified types, class of undeclared names, substitution boundary tokens |
| `recover:` | `name: pattern => "message {capture}"` |

Examples: `examples/json_parser.py`, `examples/clike.py` (typedef, `T * x`, generics),
`examples/pylike.py` (grouped by indentation).

## How it works

* **Layers** are applied in order, each to a fixed point; within a layer the longest match wins
(ties go to the earlier-written rule), and for nested choice, first success wins.
* **Captures with a goal and bracket matching** fold the slice with a separate run through all
layers with a child scope.
* **Name table**: `as CLASS` writes to the scope; the entry is applied at the layer boundary and
reclassifies tokens of `names` types, so `T * x` and `List<int>` are resolved by token class,
without backtracking.
* **Repetitions** use an explicit stack (no recursion limit); captures/consumption use persistent lists.

## Error recovery (`parsekit/recover.py`)

1. Handwritten patterns from `recover:`: matched against what remains after folding; the leftmost match
wins. In the message you can use `{захват}`, `{захват.line}`, `{line}`, `{col}`, `{text}`, `{found}`.
2. Fix search: up to 2 token insertions/replacements (deletion is enabled with `ops=`) near the
failure site; beam selection, budgets on the number of trials and on time. Hint order: fewer edits,
then tokens already seen in the input, then further right (as an LR parser would notice).
Settings: `Parser(..., window=, beam=, max_edits=, budget=)`.

