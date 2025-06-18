import math
from statistics import mean, median, stdev

def calculate_expression(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Supports basic arithmetic, math functions, and statistics.
    """
    try:
        # Define a safe evaluation environment
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed_names.update({
            "abs": abs,
            "round": round,
            "mean": mean,
            "median": median,
            "stdev": stdev,
            "min": min,
            "max": max,
            "sum": sum,
        })

        # Evaluate the expression
        result = eval(expression, {"__builtins__": None}, allowed_names)
        return f"The result of the calculation is: {result}"
    except Exception as e:
        return f"[Error] Calculation failed: {str(e)}"
