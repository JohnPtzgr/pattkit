"""
buildkit.toml: a small, stdlib-only TOML-subset parser.

Supported: [table] / [a.b.c], key = value (dotted keys ok), basic + multiline
strings, literal strings, integers (0x/0o/0b), floats, booleans, arrays
(may span lines, trailing comma ok), inline tables, # comments.
Not supported (yet): dates, [[array-of-tables]].
"""
from pathlib import Path


class TomlError(SyntaxError):
    def __init__(self, msg, line):
        super().__init__(f"toml line {line}: {msg}")
        self.msg, self.line = msg, line


class _Parser:
    def __init__(self, text):
        self.s, self.n, self.i, self.line = text, len(text), 0, 1

    # ---- cursor ----
    def eof(self):
        return self.i >= self.n

    def peek(self, k=0):
        j = self.i + k
        return self.s[j] if j < self.n else ""

    def advance(self):
        ch = self.s[self.i]
        self.i += 1
        if ch == "\n":
            self.line += 1
        return ch

    def error(self, msg):
        raise TomlError(msg, self.line)

    # ---- whitespace / comments ----
    def skip_ws(self, newlines=False):
        while not self.eof():
            ch = self.peek()
            if ch in " \t\r" or (ch == "\n" and newlines):
                self.advance()
            else:
                break

    def skip_comment(self):
        if self.peek() == "#":
            while not self.eof() and self.peek() != "\n":
                self.advance()

    def end_of_statement(self):
        self.skip_ws()
        self.skip_comment()
        if self.eof():
            return
        if self.peek() == "\n":
            self.advance()
        else:
            self.error("expected end of line")

    # ---- entry point ----
    def parse(self):
        root, current = {}, {}
        root = current = {}
        while True:
            self.skip_ws(newlines=True)
            if self.eof():
                break
            if self.peek() == "#":
                self.skip_comment()
                continue
            if self.peek() == "\n":
                self.advance()
                continue
            if self.peek() == "[":
                current = self.table_header(root)
                self.end_of_statement()
            else:
                self.keyval(current)
                self.end_of_statement()
        return root

    def table_header(self, root):
        self.advance()                                   # '['
        path = self.key_path()
        self.skip_ws()
        if self.peek() != "]":
            self.error("expected ']'")
        self.advance()
        node = root
        for part in path:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                node[part] = {}
            node = node[part]
        return node

    def key_path(self):
        parts = [self.simple_key()]
        self.skip_ws()
        while self.peek() == ".":
            self.advance()
            self.skip_ws()
            parts.append(self.simple_key())
            self.skip_ws()
        return parts

    def simple_key(self):
        self.skip_ws()
        if self.peek() in "\"'":
            return self.string()
        start = self.i
        while not self.eof() and (self.peek().isalnum() or self.peek() in "_-"):
            self.advance()
        if self.i == start:
            self.error("expected a key")
        return self.s[start:self.i]

    def keyval(self, table):
        path = self.key_path()
        self.skip_ws()
        if self.peek() != "=":
            self.error("expected '='")
        self.advance()
        self.skip_ws()
        value = self.value()
        node = table
        for part in path[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[path[-1]] = value

    # ---- values ----
    def value(self):
        self.skip_ws()
        ch = self.peek()
        if ch in "\"'":
            return self.string()
        if ch == "[":
            return self.array()
        if ch == "{":
            return self.inline_table()
        return self.scalar()

    def string(self):
        quote = self.advance()
        if self.peek(0) == quote and self.peek(1) == quote:
            self.advance(); self.advance()
            return self.multiline_string(quote)
        out, literal = [], quote == "'"
        while True:
            if self.eof() or self.peek() == "\n":
                self.error("unterminated string")
            ch = self.advance()
            if ch == quote:
                return "".join(out)
            if ch == "\\" and not literal:
                out.append(self.escape())
            else:
                out.append(ch)

    def multiline_string(self, quote):
        if self.peek() == "\n":
            self.advance()                               # trim first newline
        out, literal = [], quote == "'"
        while True:
            if self.eof():
                self.error("unterminated multiline string")
            if self.peek() == quote and self.peek(1) == quote and self.peek(2) == quote:
                self.advance(); self.advance(); self.advance()
                return "".join(out)
            ch = self.advance()
            if ch == "\\" and not literal:
                if self.peek() == "\n":                  # line-ending backslash
                    self.advance()
                    self.skip_ws(newlines=True)
                    continue
                out.append(self.escape())
            else:
                out.append(ch)

    def escape(self):
        ch = self.advance()
        simple = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\",
                  '"': '"', "b": "\b", "f": "\f"}
        if ch in simple:
            return simple[ch]
        if ch in "uU":
            width = 4 if ch == "u" else 8
            hx = self.s[self.i:self.i + width]
            self.i += width
            return chr(int(hx, 16))
        self.error(f"bad escape \\{ch}")

    def array(self):
        self.advance()                                   # '['
        items = []
        while True:
            self.skip_ws(newlines=True)
            self.skip_comment()
            if self.eof():
                self.error("unterminated array")
            if self.peek() == "]":
                self.advance()
                return items
            items.append(self.value())
            self.skip_ws(newlines=True)
            self.skip_comment()
            if self.peek() == ",":
                self.advance()
            elif self.peek() == "]":
                self.advance()
                return items
            else:
                self.error("expected ',' or ']' in array")

    def inline_table(self):
        self.advance()                                   # '{'
        table = {}
        self.skip_ws()
        if self.peek() == "}":
            self.advance()
            return table
        while True:
            self.skip_ws()
            self.keyval(table)
            self.skip_ws()
            if self.peek() == ",":
                self.advance()
            elif self.peek() == "}":
                self.advance()
                return table
            else:
                self.error("expected ',' or '}' in inline table")

    def scalar(self):
        start = self.i
        while not self.eof() and self.peek() not in "\n\r,]}#":
            self.advance()
        raw = self.s[start:self.i].strip()
        if raw == "true":
            return True
        if raw == "false":
            return False
        num = raw.replace("_", "")
        try:
            return int(num, 0)
        except ValueError:
            pass
        try:
            return float(num)
        except ValueError:
            self.error(f"invalid value {raw!r}")


def loads(text):
    return _Parser(text).parse()


def load(path):
    return loads(Path(path).read_text(encoding="utf-8"))
