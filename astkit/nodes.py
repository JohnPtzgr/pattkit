"""
pattkit.ast.nodes: The stable, language-agnostic Boundary AST.

Provides a rich set of universal nodes covering expressions, statements,
declarations, and types. Language-specific constructs fall back to BCustomNode,
which carries the raw parsekit.Tok as its payload for zero-copy extraction.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class BoundaryNode:
    """Base class for all Boundary AST nodes. Retains source location."""
    tok: Any = None


# ============================================================================
# Expressions
# ============================================================================

@dataclass
class BIdentifier(BoundaryNode):
    name: str = ""


@dataclass
class BLiteral(BoundaryNode):
    value: Any = None
    literal_type: str = ""  # e.g., "int", "str", "bool", "float"


@dataclass
class BBinaryOp(BoundaryNode):
    left: Any = None
    op: str = ""
    right: Any = None


@dataclass
class BUnaryOp(BoundaryNode):
    op: str = ""
    operand: Any = None
    is_prefix: bool = True


@dataclass
class BCallExpr(BoundaryNode):
    callee: Any = None
    args: List[Any] = field(default_factory=list)


@dataclass
class BMemberExpr(BoundaryNode):
    object: Any = None
    member: str = ""


@dataclass
class BIndexExpr(BoundaryNode):
    object: Any = None
    index: Any = None


# ============================================================================
# Statements
# ============================================================================

@dataclass
class BBlock(BoundaryNode):
    stmts: List[Any] = field(default_factory=list)


@dataclass
class BExprStmt(BoundaryNode):
    expr: Any = None


@dataclass
class BReturnStmt(BoundaryNode):
    value: Optional[Any] = None


@dataclass
class BIfStmt(BoundaryNode):
    cond: Any = None
    then_body: Any = None
    else_body: Optional[Any] = None


@dataclass
class BWhileStmt(BoundaryNode):
    cond: Any = None
    body: Any = None


@dataclass
class BForStmt(BoundaryNode):
    init: Optional[Any] = None
    cond: Optional[Any] = None
    step: Optional[Any] = None
    body: Any = None


# @dataclass
# class BImport(BoundaryNode):
#     module:  BM = ""
#     aliases: List[Any] = field(default_factory=list)
#     items:   List[str] = field(default_factory=List[str])


# ============================================================================
# Declarations
# ============================================================================

@dataclass
class BVarDecl(BoundaryNode):
    name: str = ""
    visibility: str = "pub"
    type_hint: Optional[Any] = None
    value: Optional[Any] = None
    is_const: bool = False


@dataclass
class BFuncDecl(BoundaryNode):
    name: str = ""
    params: List[Any] = field(default_factory=list)
    return_type: Optional[Any] = None
    body: Optional[BBlock] = None
    visibility: str = "pub"


@dataclass
class BParamDecl(BoundaryNode):
    name: str = ""
    type_hint: Optional[Any] = None
    visibility: str = "pub"


# @dataclass
# class BModuleDecl(BoundaryNode):
#     name: str = ""
#     body: Optional[BBlock] = None
#     visibility: str = "pub"


# ============================================================================
# Types
# ============================================================================

@dataclass
class BTypeRef(BoundaryNode):
    name: str = ""


@dataclass
class BGenericInst(BoundaryNode):
    base: Any = None
    args: List[Any] = field(default_factory=list)


# ============================================================================
# The Escape Hatch
# ============================================================================

@dataclass
class BCustomNode(BoundaryNode):
    """
    Opaque node for language-specific or complex constructs.
    The generic builder creates these when it encounters a parsekit label
    that isn't a universal primitive.

    Downstream tools (like a language-specific type checker or C codegen)
    MUST handle these via a visitor.
    """
    tag: str = ""
    payload: Any = None  # The raw parsekit.Tok tree
