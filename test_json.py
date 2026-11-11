import json
import time
from _util import check, diag, fails
from json_parser import make_parser, to_python

p = make_parser()
check("lint: clean", p.warnings == [], p.warnings)
for src in ['1', '-12', '"hi"', 'true', 'null', '[]', '{}', '[1, 2, 3]', '[[1, [2]], []]',
            '{"a": 1}', '{"a": [1, 2.5, -3], "b": {"c": null}, "d": "x\\ny", "e": true}',
            ' [ 1 ,\n 2 ] ']:
    check(f"round trip {src.strip()[:40]!r}", to_python(p.parse(src)) == json.loads(src))
d = diag(p, "[1 2]")
check("missing comma -> suggestion", d.suggestions and "insert ','" in d.suggestions[0].describe(), d.format())
d = diag(p, '{"a": 1')
check("missing '}' -> suggestion", d.suggestions and "insert '}'" in d.suggestions[0].describe(), d.format())
print(d.format())
check("trailing comma rejected", fails(p, "[1, 2,]") is not None)

# long lists must not hit the Python recursion limit (explicit-stack repetition)
for n in (1000, 5000):
    src = "[" + ", ".join(str(i) for i in range(n)) + "]"
    t = time.time()
    v = to_python(p.parse(src))
    dt = time.time() - t
    check(f"list of {n} items parses ({dt:.2f}s)", v == list(range(n)))
deep = "[" * 60 + "1" + "]" * 60
check("nesting depth 60", to_python(p.parse(deep)) is not None)
print("json: all passed")
