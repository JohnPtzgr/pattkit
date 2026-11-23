"""
pattkit.astkit.visitors: Traversal and transformation infrastructure
for the Boundary AST.

Three idioms are demonstrated:

  1. ASTWalker      — the base dispatch machinery (read-only).
  2. ASTDumper      — a read-only visitor that pretty-prints the tree.
  3. NameCollector  — a read-only visitor that *accumulates* information.
  4. ConstantFolder — a Transformer that *rewrites* nodes in place.

Dispatch rules
--------------
visit(node) resolves in this order:

  a) BCustomNode  →  visit_<sanitised_tag>(node)   e.g. visit_light_shared_let
                 →  visit_BCustomNode(node)
                 →  generic_visit(node)

  b) Any other    →  visit_<ClassName>(node)       e.g. visit_BBinaryOp
                 →  generic_visit(node)

generic_visit walks every field that holds a BoundaryNode (or a list of
them), so adding a new node type to nodes.py never requires touching the
visitor base class.
"""
from __future__ import annotations

import re
from dataclasses import fields as dc_fields
from itertools import pairwise
from typing import Any, Dict, List, Optional, Set

from . import nodes


# ---------------------------------------------------------------------------
# Base machinery
# ---------------------------------------------------------------------------

class Visitor:
    """Read-only AST traversal.

    Subclass and override ``visit_<NodeClass>`` methods.  Call
    ``self.visit(node)`` to start.  The return value of each visit
    method is passed back to the caller, so visitors can accumulate
    and return information.
    """

    def visit(self, node: Any) -> Any:
        if node is None:
            return None

        # BCustomNode: try tag-specific handler first
        if isinstance(node, nodes.BCustomNode):
            handler = self._custom_handler(node)
            if handler is not None:
                return handler(node)
            handler = getattr(self, "visit_BCustomNode", None)
            if handler is not None:
                return handler(node)
            return self.generic_visit(node)

        # Regular node: dispatch on class name
        handler = getattr(self, f"visit_{type(node).__name__}", None)
        if handler is not None:
            return handler(node)
        return self.generic_visit(node)

    def generic_visit(self, node: Any) -> None:
        """Walk every child field.  Override only if you need custom traversal."""
        for child in self.iter_children(node):
            self.visit(child)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def iter_children(node: Any):
        """Yield every BoundaryNode (or list thereof) stored on *node*."""
        if not hasattr(node, "__dataclass_fields__"):
            return
        for f in dc_fields(node):
            if f.name == "tok":          # never traverse the raw parsekit token
                continue
            val = getattr(node, f.name)
            if isinstance(val, nodes.BoundaryNode):
                yield val
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, nodes.BoundaryNode):
                        yield item

    def _custom_handler(self, node: nodes.BCustomNode):
        """Resolve a ``visit_<tag>`` handler for a BCustomNode.

        Tags may contain characters that aren't valid in identifiers
        (e.g. ``"light:shared_let"``), so we sanitise them.
        """
        tag = getattr(node, "tag", "") or ""
        safe = re.sub(r"\W", "_", tag)
        return getattr(self, f"visit_{safe}", None)


class Transformer(Visitor):
    """AST *rewriting* traversal.

    Each ``transform_<NodeClass>`` method returns a (possibly new) node.
    The default behaviour rebuilds the node with transformed children,
    so you only override the cases you care about.

    Usage::

        class MyRewriter(Transformer):
            def transform_BBinaryOp(self, node):
                # ... maybe fold, maybe rewrite ...
                return node  # or a new node

        new_tree = MyRewriter().visit(old_tree)
    """

    def generic_visit(self, node: Any) -> Any:
        """Rebuild *node* with transformed children."""
        if not hasattr(node, "__dataclass_fields__"):
            return node

        changed = False
        new_vals: Dict[str, Any] = {}

        for f in dc_fields(node):
            if f.name == "tok":
                new_vals[f.name] = getattr(node, f.name)
                continue
            val = getattr(node, f.name)
            if isinstance(val, nodes.BoundaryNode):
                new_val = self.visit(val)
                new_vals[f.name] = new_val
                changed = changed or (new_val is not val)
            elif isinstance(val, list):
                new_list = []
                for item in val:
                    if isinstance(item, nodes.BoundaryNode):
                        new_item = self.visit(item)
                        new_list.append(new_item)
                        changed = changed or (new_item is not item)
                    else:
                        new_list.append(item)
                new_vals[f.name] = new_list
            else:
                new_vals[f.name] = val

        if not changed:
            return node
        return type(node)(**new_vals)


# ---------------------------------------------------------------------------
# 1. ASTDumper — pretty-print the tree (read-only, no accumulation)
# ---------------------------------------------------------------------------

class ASTDumper(Visitor):
    """Render the AST as an indented tree.

    Replaces the ad-hoc ``_walk_ast`` in test scripts::

        print(ASTDumper().dump(tree))
    """

    def dump(self, node: Any) -> str:
        self._lines: List[str] = []
        self.visit(node)
        return "\n".join(self._lines)

    def generic_visit(self, node: Any) -> None:
        depth = len(self._stack) if hasattr(self, "_stack") else 0
        indent = "  " * depth
        name = type(node).__name__

        # Collect scalar fields for the header line
        scalars = []
        for f in dc_fields(node):
            if f.name == "tok":
                continue
            val = getattr(node, f.name)
            if isinstance(val, (str, int, float, bool, type(None))):
                if val is not None and val != "" and val != [] and val is not False:
                    scalars.append(f"{f.name}={val!r}")

        header = f"{name}({', '.join(scalars)})" if scalars else name
        self._lines.append(f"{indent}{header}")

        # Recurse into children
        if not hasattr(self, "_stack"):
            self._stack: List[Any] = []
        self._stack.append(node)
        for child in self.iter_children(node):
            self.visit(child)
        self._stack.pop()


# ---------------------------------------------------------------------------
# 2. NameCollector — accumulate information (read-only, returns a result)
# ---------------------------------------------------------------------------

class NameCollector(Visitor):
    """Collect every name declared and every name referenced.

    Returns a ``NameReport``.  This is the first building block for
    symbol resolution, which Light's type checker will need::

        report = NameCollector().collect(tree)
        print(report.declared)   # {'x', 'main', 'add'}
        print(report.referenced) # {'x', 'add'}
    """

    def __init__(self):
        self.declared: Set[str] = set()
        self.referenced: Set[str] = set()

    def collect(self, node: Any) -> "NameCollector":
        self.visit(node)
        return self

    # Declarations introduce names
    def visit_BVarDecl(self, node: nodes.BVarDecl):
        self.declared.add(node.name)
        self.generic_visit(node)          # still walk the initializer

    def visit_BFuncDecl(self, node: nodes.BFuncDecl):
        self.declared.add(node.name)
        self.generic_visit(node)

    def visit_BParamDecl(self, node: nodes.BParamDecl):
        self.declared.add(node.name)
        # params have no children worth visiting, but be safe
        self.generic_visit(node)

    # Identifiers reference names
    def visit_BIdentifier(self, node: nodes.BIdentifier):
        self.referenced.add(node.name)


# ---------------------------------------------------------------------------
# 3. ConstantFolder — rewrite the tree (Transformer)
# ---------------------------------------------------------------------------

class ConstantFolder(Transformer):
    """Fold constant arithmetic at compile time.

    ``40 + 2`` becomes ``42``.  Demonstrates the Transformer idiom:
    override only the node types you want to rewrite; everything else
    is rebuilt automatically by ``generic_visit``::

        folded = ConstantFolder().visit(tree)
    """

    # Only handle int and float literals; leave strings/bools alone.
    _NUMERIC = {"digits", "int", "float", "number"}

    def transform_BBinaryOp(self, node: nodes.BBinaryOp) -> Any:
        # First, recursively fold children
        node = self.generic_visit(node)

        left, right = node.left, node.right
        if not (self._is_num(left) and self._is_num(right)):
            return node

        lv = float(left.value) if "." in str(left.value) else int(left.value)
        rv = float(right.value) if "." in str(right.value) else int(right.value)

        result = self._apply(node.op, lv, rv)
        if result is None:
            return node

        return nodes.BLiteral(
            tok=node.tok,
            value=str(result),
            literal_type="float" if isinstance(result, float) else "int",
        )

    def transform_BUnaryOp(self, node: nodes.BUnaryOp) -> Any:
        node = self.generic_visit(node)
        if not self._is_num(node.operand):
            return node
        v = float(node.operand.value) if "." in str(node.operand.value) else int(node.operand.value)
        if node.op == "-":
            v = -v
        elif node.op == "+":
            pass
        else:
            return node
        return nodes.BLiteral(
            tok=node.tok,
            value=str(v),
            literal_type="float" if isinstance(v, float) else "int",
        )

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _is_num(node: Any) -> bool:
        return (
            isinstance(node, nodes.BLiteral)
            and getattr(node, "literal_type", "").lower() in ConstantFolder._NUMERIC
        )

    @staticmethod
    def _apply(op: str, lv, rv):
        try:
            if op == "+": return lv + rv
            if op == "-": return lv - rv
            if op == "*": return lv * rv
            if op == "/" and rv != 0: return lv / rv
            if op == "%" and rv != 0: return lv % rv
        except Exception:
            pass
        return None


# class ImportCollector(Visitor):
#     def __init__(self):
#         self.imports: List[str] = []      # module names / paths
#
#     def collect(self, root):
#         self.visit(root)
#         return self.imports
#
#     def visit_BImport(self, node: nodes.BImport):
#         self.imports.append(node.module)
#         self.generic_visit(node)
#
#
# class PublicSymbolExtractor(Visitor):
#     def __init__(self):
#         self.symbols: Dict[str, nodes.BoundaryNode] = {}   # name -> decl node (signature only)
#
#     def extract(self, root):
#         self.visit(root)
#         return self.symbols
#
#     def visit_BFuncDecl(self, node: nodes.BFuncDecl):
#         if node.visibility in ("pub", None) and self._exported(node):
#             # store a body-less copy = a prototype
#             new_node = node
#             new_node.body = None
#             self.symbols[node.name] = new_node
#         self.generic_visit(node)
#
#     def visit_BVarDecl(self, node: nodes.BVarDecl):
#         if node.visibility == "pub":
#             self.symbols[node.name] = node
#         self.generic_visit(node)
#
#     def visit_BModuleDecl(self, node: nodes.BModuleDecl):
#         if node.visibility == "pub":
#             # store a body-less copy = a prototype
#             new_node = node
#             new_node.body = None
#             self.symbols[node.name] = new_node
#         self.generic_visit(node)
#
# class DependencyInjector(Transformer):
#     def __init__(self, dep_interfaces: Dict[str, Dict[str, nodes.BoundaryNode]]):
#         self.dep_interfaces = dep_interfaces   # dep_name -> {sym_name -> prototype}
#
#     def transform_BImport(self, node: nodes.BImport):
#         injected = []
#         for items in node.:                       # resolved import list
#             for proto in self.dep_interfaces[dep].values():
#                 injected.append(proto)                 # body-less prototypes
#         new_node = node
#         new_node.items = None
#         return replace(node, items=injected + node.items)
