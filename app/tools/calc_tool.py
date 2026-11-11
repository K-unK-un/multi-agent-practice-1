"""calculator 工具：受限表达式求值（AST 白名单），供 eligibility 专家做确定性计算。

安全纪律：绝不使用 eval/exec。只允许数字、四则与取模、括号、一元正负号，
以及 round(x[, n]) / min(...) / max(...) / abs(x) 四个内置函数。
非法语法或非白名单节点统一返回中文错误，不抛异常。
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Any

from langchain_core.tools import tool

from app.core.blackboard import emit

_MAX_EXPR_LEN = 200
_MAX_NODES = 100

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_ALLOWED_FUNCS = {"round", "min", "max", "abs"}


class _UnsafeExpression(Exception):
    """表达式包含白名单之外的语法节点。"""


def _check_number(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _UnsafeExpression("非数值")
    if isinstance(value, float) and not math.isfinite(value):
        raise _UnsafeExpression("非有限数值")
    return value


def _eval_node(node: ast.AST, counter: list[int]) -> int | float:
    counter[0] += 1
    if counter[0] > _MAX_NODES:
        raise _UnsafeExpression("表达式过于复杂")

    if isinstance(node, ast.Expression):
        return _eval_node(node.body, counter)
    if isinstance(node, ast.Constant):
        return _check_number(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left, counter)
        right = _eval_node(node.right, counter)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("除数不能为零")
        if isinstance(node.op, ast.Mod) and right == 0:
            raise ZeroDivisionError("取模的除数不能为零")
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand, counter))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise _UnsafeExpression("不允许的函数调用")
        if node.keywords:
            raise _UnsafeExpression("不支持关键字参数")
        args = [_eval_node(arg, counter) for arg in node.args]
        name = node.func.id
        if name == "round":
            if len(args) not in (1, 2):
                raise _UnsafeExpression("round 参数数量不合法")
            if len(args) == 2:
                digits = args[1]
                if isinstance(digits, float) and not digits.is_integer():
                    raise _UnsafeExpression("round 的位数必须为整数")
                return round(args[0], int(digits))
            return round(args[0])
        if name == "abs":
            if len(args) != 1:
                raise _UnsafeExpression("abs 参数数量不合法")
            return abs(args[0])
        # min / max
        if not args:
            raise _UnsafeExpression("min/max 至少需要一个参数")
        return min(args) if name == "min" else max(args)

    raise _UnsafeExpression(f"不允许的节点：{type(node).__name__}")


def _format_number(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.10g}"


@tool
def calculator_tool(expression: str) -> str:
    """计算受限数学表达式，返回确定性结果，供资格/金额/天数推导使用。

    支持的语法：
    - 数字：整数与小数（如 10、3.5；科学计数法如 1e3 按数字解析）；
    - 运算：+  -  *  /  %（取模）与括号，支持一元正负号；
    - 函数：round(x[, n])、min(...)、max(...)、abs(x)；
    - 不支持：幂运算、比较、逻辑、变量、字符串、下标、属性访问等。

    Args:
        expression: 待计算表达式，例如 "round(5000 * 0.8, 2)" 或 "(10 + 2) / 3"。
    """
    expression = (expression or "").strip()
    if not expression:
        return "计算失败：表达式为空"
    if len(expression) > _MAX_EXPR_LEN:
        return f"计算失败：表达式过长（最多 {_MAX_EXPR_LEN} 个字符）"

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        emit("tool", tool="calculator", ok=False)
        return f"表达式格式错误：{exc.msg}"

    try:
        result = _eval_node(tree, [0])
        if isinstance(result, float) and not math.isfinite(result):
            raise _UnsafeExpression("非有限数值")
        formatted = _format_number(result)
    except _UnsafeExpression:
        emit("tool", tool="calculator", ok=False)
        return "表达式包含不允许的运算"
    except ZeroDivisionError:
        emit("tool", tool="calculator", ok=False)
        return "计算失败：除数不能为零"
    except (OverflowError, ValueError, TypeError):
        emit("tool", tool="calculator", ok=False)
        return "计算失败：数值超出可计算范围"

    emit("tool", tool="calculator", ok=True)
    return f"计算结果：{formatted}（请结合规则表确认单位）"
