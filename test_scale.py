import time
from _util import check
from json_parser import make_parser
from clike import make_parser as clike

p, c = make_parser(), clike()


def timed(f, *a):
    t = time.time()
    f(*a)
    return time.time() - t


ts = {n: timed(p.parse, "[" + ", ".join(str(i) for i in range(n)) + "]") for n in (4000, 8000, 16000)}
print("  JSON array:", {n: round(t, 2) for n, t in ts.items()})
check("JSON array scales ~linearly (16000 / 8000 < 3.5)", ts[16000] / ts[8000] < 3.5, ts)
src = lambda n: "typedef int T;\n" + "T * x;\na * b + c;\n" * n
tc = {n: timed(c.parse, src(n)) for n in (400, 800, 1600)}
print("  C-like stmts:", {2 * n: round(t, 2) for n, t in tc.items()})
check("C-like scales ~linearly (1600 / 800 < 3.5)", tc[1600] / tc[800] < 3.5, tc)
big = "[" * 100 + "1" + "]" * 100
check("nesting depth 100", p.parse(big) is not None)
print("scale: all passed")
