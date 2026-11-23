"""
pattkit.astkit.builder: Converts a parsekit.Tok tree into the rich Boundary AST.

Maps common parsekit labels to standard Boundary nodes. Unknown labels fall
back to BCustomNode, passing the raw Tok as payload for zero-copy extraction.
"""
from . import nodes


class GenericASTBuilder:
    """
    Walks the parsekit AST and converts it to the Boundary AST.
    Maps common parsekit labels to standard Boundary nodes.
    """

    # Map of parsekit rule types/labels to builder methods.
    # Checked in order: first tok.label, then tok.type.
    HANDLERS = {
        # Expressions
        "IDENTIFIER": "_build_ident",
        "DIGITS": "_build_literal",
        "STRING": "_build_literal",
        "CHAR": "_build_literal",
        "binary": "_build_binary",
        "prefix": "_build_unary",
        "postfix": "_build_unary",
        "chain": "_build_chain",
        "call": "_build_call",
        "apply": "_build_call",
        "member": "_build_member",
        "index": "_build_index",

        # Statements
        "block": "_build_block",
        "suite": "_build_block",
        "expr_stmt": "_build_expr_stmt",
        "exprstmt": "_build_expr_stmt",
        "return": "_build_return",
        "return_stmt": "_build_return",
        "if": "_build_if",
        "ifstmt": "_build_if",
        "while": "_build_while",
        "for": "_build_for",

        # Declarations
        "var_decl": "_build_var_decl",
        "vardecl": "_build_var_decl",
        "let_stmt": "_build_var_decl",
        "assign": "_build_var_decl",
        "func_decl": "_build_func_decl",
        "fn_decl": "_build_func_decl",
        "param": "_build_param",

        # Types
        "type_ref": "_build_type_ref",
        "type": "_build_type_ref",
        "type/plain": "_build_type_ref",
        "generic_type": "_build_generic_inst",
        "type/generic": "_build_generic_inst",

        # Top-level
        "program": "_build_program",
        "module": "_build_program",
    }

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def build(self, root_tok):
        return self._build_node(root_tok)

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_node(self, tok):
        """Convert a single parsekit Tok (or list of Tokes) to a BoundaryNode."""
        if tok is None:
            return None

        # Unwrap single-element lists (parsekit sometimes wraps captures in lists)
        if isinstance(tok, (list, tuple)):
            if len(tok) == 0:
                return None
            if len(tok) == 1:
                return self._build_node(tok[0])
            # Multiple items: build each and return as a list
            return [self._build_node(t) for t in tok]

        handler_name = (
            self.HANDLERS.get(getattr(tok, "label", None))
            or self.HANDLERS.get(getattr(tok, "type", None))
        )
        if handler_name:
            return getattr(self, handler_name)(tok)

        # FALLBACK: Wrap unknown/language-specific labels in a BCustomNode.
        return nodes.BCustomNode(
            tok=tok,
            tag=getattr(tok, "label", None) or getattr(tok, "type", ""),
            payload=tok,
        )

    @staticmethod
    def _single(val):
        """Extract a single value from a capture.

        parsekit captures for optional/alternative/repeated patterns may
        return a list even when only one item matched. Unwrap it.
        """
        if isinstance(val, (list, tuple)):
            return val[0] if val else None
        return val

    @staticmethod
    def _list(val):
        """Extract a list of values from a capture."""
        if val is None:
            return []
        return val if isinstance(val, list) else [val]

    # ──────────────────────────────────────────────────────────────────────
    # Top-level
    # ──────────────────────────────────────────────────────────────────────

    def _build_program(self, tok):
        """Build a BBlock containing all top-level items."""
        fields = getattr(tok, "fields", {})
        items = self._list(fields.get("items") or fields.get("body"))
        return nodes.BBlock(
            tok=tok,
            stmts=[self._build_node(item) for item in items],
        )

    # ──────────────────────────────────────────────────────────────────────
    # Leaf expressions
    # ──────────────────────────────────────────────────────────────────────

    def _build_ident(self, tok):
        return nodes.BIdentifier(tok=tok, name=tok.text)

    def _build_literal(self, tok):
        return nodes.BLiteral(
            tok=tok,
            value=tok.text,
            literal_type=getattr(tok, "type", "").lower(),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Compound expressions
    # ──────────────────────────────────────────────────────────────────────

    def _build_binary(self, tok):
        fields = getattr(tok, "fields", {})
        op_tok = self._single(fields.get("op"))
        return nodes.BBinaryOp(
            tok=tok,
            left=self._build_node(self._single(fields.get("lhs"))),
            op=op_tok.text if op_tok else "+",
            right=self._build_node(self._single(fields.get("rhs"))),
        )

    def _build_chain(self, tok):
        """a < b < c -> nested binary ops (or a dedicated chain node if needed)."""
        fields = getattr(tok, "fields", {})
        operands = fields.get("operands", [])
        ops = fields.get("ops", [])
        if not operands:
            return nodes.BCustomNode(tok=tok, tag="chain", payload=tok)
        # Fold left-to-right: ((a < b) < c)
        result = self._build_node(operands[0])
        for i, op_tok in enumerate(ops):
            right = self._build_node(operands[i + 1]) if i + 1 < len(operands) else None
            result = nodes.BBinaryOp(
                tok=tok,
                left=result,
                op=op_tok.text if hasattr(op_tok, "text") else str(op_tok),
                right=right,
            )
        return result

    def _build_unary(self, tok):
        fields = getattr(tok, "fields", {})
        op_tok = self._single(fields.get("op"))
        return nodes.BUnaryOp(
            tok=tok,
            op=op_tok.text if op_tok else "",
            operand=self._build_node(self._single(fields.get("arg"))),
            is_prefix=(getattr(tok, "label", "") == "prefix"),
        )

    def _build_call(self, tok):
        fields = getattr(tok, "fields", {})
        callee = self._build_node(self._single(fields.get("callee") or fields.get("fn")))
        args = self._list(fields.get("args") or fields.get("arg"))
        return nodes.BCallExpr(
            tok=tok,
            callee=callee,
            args=[self._build_node(a) for a in args],
        )

    def _build_member(self, tok):
        fields = getattr(tok, "fields", {})
        member_tok = self._single(fields.get("member"))
        return nodes.BMemberExpr(
            tok=tok,
            object=self._build_node(self._single(fields.get("object") or fields.get("obj"))),
            member=member_tok.text if member_tok else "",
        )

    def _build_index(self, tok):
        fields = getattr(tok, "fields", {})
        return nodes.BIndexExpr(
            tok=tok,
            object=self._build_node(self._single(fields.get("object") or fields.get("obj"))),
            index=self._build_node(self._single(fields.get("index"))),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Statements
    # ──────────────────────────────────────────────────────────────────────

    def _build_block(self, tok):
        """Build a BBlock from the 'body' capture, NOT from raw kids.

        Using tok.kids would include delimiter tokens ({, }) which are not
        statements. The grammar's body= capture gives us only the actual stmts.
        """
        fields = getattr(tok, "fields", {})
        body = self._list(fields.get("body") or fields.get("stmts"))
        return nodes.BBlock(
            tok=tok,
            stmts=[self._build_node(stmt) for stmt in body],
        )

    def _build_expr_stmt(self, tok):
        fields = getattr(tok, "fields", {})
        return nodes.BExprStmt(
            tok=tok,
            expr=self._build_node(self._single(fields.get("e") or fields.get("expr"))),
        )

    def _build_return(self, tok):
        fields = getattr(tok, "fields", {})
        value = self._single(fields.get("value"))
        return nodes.BReturnStmt(
            tok=tok,
            value=self._build_node(value) if value is not None else None,
        )

    def _build_if(self, tok):
        fields = getattr(tok, "fields", {})
        return nodes.BIfStmt(
            tok=tok,
            cond=self._build_node(self._single(fields.get("cond"))),
            then_body=self._build_node(self._single(fields.get("body") or fields.get("then"))),
            else_body=self._build_node(self._single(fields.get("else_body") or fields.get("else"))),
        )

    def _build_while(self, tok):
        fields = getattr(tok, "fields", {})
        return nodes.BWhileStmt(
            tok=tok,
            cond=self._build_node(self._single(fields.get("cond"))),
            body=self._build_node(self._single(fields.get("body"))),
        )

    def _build_for(self, tok):
        fields = getattr(tok, "fields", {})
        return nodes.BForStmt(
            tok=tok,
            init=self._build_node(self._single(fields.get("init"))),
            cond=self._build_node(self._single(fields.get("cond"))),
            step=self._build_node(self._single(fields.get("step"))),
            body=self._build_node(self._single(fields.get("body"))),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Declarations
    # ──────────────────────────────────────────────────────────────────────

    def _build_var_decl(self, tok):
        fields = getattr(tok, "fields", {})
        name_tok = self._single(fields.get("name"))
        type_hint = self._single(fields.get("type") or fields.get("ty"))
        value = self._single(fields.get("value") or fields.get("val"))
        return nodes.BVarDecl(
            tok=tok,
            name=name_tok.text if name_tok else "",
            type_hint=self._build_node(type_hint) if type_hint is not None else None,
            value=self._build_node(value) if value is not None else None,
        )

    def _build_func_decl(self, tok):
        fields = getattr(tok, "fields", {})
        name_tok = self._single(fields.get("name"))
        params = self._list(fields.get("params") or fields.get("param"))
        ret = self._single(fields.get("ret") or fields.get("return_type"))
        body = self._single(fields.get("body"))
        return nodes.BFuncDecl(
            tok=tok,
            name=name_tok.text if name_tok else "",
            params=[self._build_node(p) for p in params],
            return_type=self._build_node(ret) if ret is not None else None,
            body=self._build_node(body) if body is not None else None,
        )

    def _build_param(self, tok):
        fields = getattr(tok, "fields", {})
        name_tok = self._single(fields.get("name"))
        type_hint = self._single(fields.get("ty") or fields.get("type"))
        return nodes.BParamDecl(
            tok=tok,
            name=name_tok.text if name_tok else "",
            type_hint=self._build_node(type_hint) if type_hint is not None else None,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Types
    # ──────────────────────────────────────────────────────────────────────

    def _build_type_ref(self, tok):
        return nodes.BTypeRef(tok=tok, name=getattr(tok, "text", ""))

    def _build_generic_inst(self, tok):
        fields = getattr(tok, "fields", {})
        base = self._build_node(self._single(fields.get("base")))
        args = self._list(fields.get("args") or fields.get("arg"))
        return nodes.BGenericInst(
            tok=tok,
            base=base,
            args=[self._build_node(a) for a in args],
        )