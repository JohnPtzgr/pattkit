import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def check(name, cond, extra=""):
    print(("ok   " if cond else "FAIL ") + name + ("" if cond else f"  {extra}"))
    if not cond:
        raise SystemExit(1)


def diag(p, src):
    from parsekit import ParseError
    try:
        p.parse(src)
    except ParseError as e:
        return e.diagnostic
    return None


def fails(p, src):
    try:
        p.parse(src)
    except SyntaxError as e:
        return str(e)
    return None
