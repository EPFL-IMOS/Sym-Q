from abc import ABC

import torch

# Module-level constants
PLACEHOLDER = "PH"
OPT_SEQ = {
    "x_1": "x_1",
    "x_2": "x_2",
    "x_3": "x_3",
    "c": "c",
    "abs": "abs",
    "+": "add",
    "*": "mul",
    "/": "div",
    "sqrt": "sqrt",
    "exp": "exp",
    "ln": "ln",
    "**": "pow",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "coth": "coth",
    "-3": "-3",
    "-2": "-2",
    "-1": "-1",
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
}
OPT_MAP = {v: k for k, v in OPT_SEQ.items()}


class Node:
    UNARY_OPERATORS = {
        "sin",
        "cos",
        "tan",
        "exp",
        "sqrt",
        "log",
        "ln",
        "abs",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "coth",
    }
    BINARY_OPERATORS = {"+", "*", "/", "**"}

    def __init__(self, parent=None):
        self.val = PLACEHOLDER
        self.parent = parent

    def __str__(self):
        return self.val

    @property
    def binary(self) -> bool:
        """Check if the node is a binary operation."""
        if self.val in Node.UNARY_OPERATORS:
            return False
        elif self.val in Node.BINARY_OPERATORS:
            return True
        return None

    def set_value(self, val: str) -> list:
        """
        Set the value of the node and expand the placeholders accordingly.

        Returns:
        - A list of the newly created placeholder nodes.
        """
        assert isinstance(val, str), "Node value must be a string."

        self.val = OPT_MAP[val]

        if self.binary is False:
            self.child = Node(self)
            return [self.child]
        elif self.binary is True:
            self.left = Node(self)
            self.right = Node(self)
            return [self.right, self.left]  # LIFO: right first

        return []


class Expression(ABC):
    pass


class TargetExpression(Expression):
    def __init__(
        self,
        point_set: torch.FloatTensor,
        expr: str = None,
        skeleton: str = None,
        opt_sequence: list = None,
    ):
        """
        Initialize the Expression with a default value or an existing expression.
        """
        self._point_set = point_set
        self._expr = expr
        self._skeleton = skeleton
        self._opt_sequence = opt_sequence

    @property
    def point_set(self) -> torch.FloatTensor:
        return self._point_set

    @property
    def expr(self) -> str:
        return self._expr

    @property
    def skeleton(self) -> str:
        return self._skeleton

    @property
    def opt_sequence(self) -> list:
        return self._opt_sequence


class AgentExpression(Expression):
    def __init__(self):
        """
        Initialize the Expression with a default value or an existing expression.
        """
        self.tree = Node()
        self._opt_sequence = []
        self._to_be_expanded = [self.tree]

    @property
    def skeleton(self) -> str:
        return self._update_skeleton(self.tree)

    @property
    def opt_sequence(self) -> list:
        return self._opt_sequence

    def add_opt(self, opt: str) -> int:
        """
        Add operation to the expression.

        Returns:
        - A flag indicating the status of the expression:
            - 0: operation successfully added.
            - 1: no more placeholder.
        """

        if not self._to_be_expanded:
            raise ValueError("No placeholder found in the expression.")

        node = self._to_be_expanded.pop()

        # Set the value of the node and add the newly created placeholders
        h_list = node.set_value(opt)
        self._to_be_expanded.extend(h_list)

        # Add the operation to the history
        self._opt_sequence.append(opt)

        return 0 if self._to_be_expanded else 1

    def _update_skeleton(self, node: Node):
        """Update the string representation of the expression."""
        if node.binary is False:
            return f"{node.val}({self._update_skeleton(node.child)})"
        if node.binary is True:
            return f"({self._update_skeleton(node.left)} {node.val} {self._update_skeleton(node.right)})"
        return node.val


if __name__ == "__main__":
    expr = AgentExpression()
    print(expr.skeleton)
    print(expr.add_opt("sin"))
    print(expr.skeleton)
    print(expr.add_opt("sin"))
    print(expr.skeleton)
    print(expr.add_opt("add"))
    print(expr.skeleton)
    print(expr.add_opt("x_1"))
    print(expr.skeleton)
    print(expr.add_opt("x_2"))
    print(expr.skeleton)
