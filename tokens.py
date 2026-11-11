"""Token object shared by the lexer, the engine and the recovery layer."""


class Tok:
    """A leaf (no kids) or a node produced by a reduction (kids, label, fields)."""
    __slots__ = ("type", "text", "pos", "line", "col", "w", "nl", "kids",
                 "label", "fields", "cls", "trivia", "params")

    def __init__(s, type, text="", pos=0, line=1, col=1, kids=(), label=None,
                 fields=None, w=0, nl=0):
        s.type, s.text, s.pos, s.line, s.col = type, text, pos, line, col
        s.kids, s.label, s.fields, s.w, s.nl = tuple(kids), label, fields or {}, w, nl
        s.cls, s.trivia, s.params = None, [], {}

    def __repr__(s):
        return f"{s.type}{'/' + s.label if s.label else ''}({s.text!r})"

    def __str__(s):
        return s.text
