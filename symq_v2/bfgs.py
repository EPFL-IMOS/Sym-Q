"""
This module provides an implementation of the BFGS optimization algorithm, specifically
tailored for our use case. It is a modified version of the implementation found in 
Joint_Supervised_Learning_for_SR/src/architectures/bfgs.py
"""

import re
import math
import warnings
from joblib import Parallel, delayed

import numpy as np
import sympy as sp
from scipy.optimize import minimize

import traceback
from util import MODULES

c = sp.Symbol("c")
subs_dict = {
    sp.Add(c, c): c,
    c - c: 0,
    c / c: 1,
    sp.Pow(c, c): c,
    sp.Pow(c, -3): c,
    sp.Pow(c, -2): c,
    sp.Pow(c, -1): c,
    sp.Pow(c, 0): 1,
    sp.Pow(c, 1): c,
    sp.Pow(c, 2): c,
    sp.Pow(c, 3): c,
    sp.Pow(c, 4): c,
    sp.Pow(c, 5): c,
    sp.Pow(-3, c): c,
    sp.Pow(-2, c): c,
    sp.Pow(-1, c): c,
    sp.Pow(0, c): 0,
    sp.Pow(1, c): 1,
    sp.Pow(2, c): c,
    sp.Pow(3, c): c,
    sp.Pow(4, c): c,
    sp.Pow(5, c): c,
    sp.sin(c): c,
    sp.cos(c): c,
    sp.tan(c): c,
    sp.asin(c): c,
    sp.acos(c): c,
    sp.atan(c): c,
    sp.sinh(c): c,
    sp.cosh(c): c,
    sp.tanh(c): c,
    sp.coth(c): c,
    sp.sqrt(c): c,
    sp.log(c): c,
    sp.exp(c): c,
    sp.Abs(c): c,
}


def simplify_constant(expr):
    parse_expr = sp.sympify(expr)
    parse_expr = parse_expr.xreplace(subs_dict)
    return str(parse_expr)


def bfgs(expr_candidate, traversal, _X, y, const_candidate=[], n_try=1, n_jobs=1):
    # X: [n_points, n_vars]
    # y: [n_points]

    # 1. Preprocess X by removing all-zero columns
    X = _X[:, ~np.all(_X == 0, axis=0)].astype(np.float64)
    n_vars = X.shape[1]
    total_variables = [f"x_{i}" for i in range(1, n_vars + 1)]

    # 2. Replace 'c' with 'constant' and count constants
    expr = expr_candidate
    for  _ in range(20):
        parsed_expr = simplify_constant(expr)
        if parsed_expr == expr:
            break
        else:
            expr = parsed_expr
    expr_candidate = parsed_expr
    candidate = re.sub(r"\bc\b", "constant", expr_candidate)
    n_const = candidate.count("constant")

    # 3. Create SymPy symbols for variables and constants
    sym_vars = sp.symbols(total_variables)
    const_symbols = sp.symbols(f"c0:{n_const}")

    # 4. Replace 'constant' with specific constant symbols (c0, c1, ...)
    for i in range(n_const):
        candidate = candidate.replace("constant", f"c{i}", 1)

    # 5. Parse the expression with constants
    expression = sp.sympify(candidate)
    try:
        expression = sp.simplify(expression)
    except Exception as e:
        pass

    # 6. Create a lambdified function for the expression
    if n_const == 0:
        # No constants to optimize
        try:
            if "I" in str(expression):
                return None
            numeric_func = sp.lambdify(sym_vars, expression, modules=MODULES)
            y_pred = numeric_func(*X.T)
        except Exception as e:
            print(f"Encountered in no const: {e} | {expr_candidate}")
            traceback.print_exc()
            return None

        mse = np.mean((y_pred - y) ** 2)
        rmse = np.sqrt(mse)
        nrmse = rmse / np.std(y)
        r2 = 1 - np.sum((y_pred - y) ** 2) / np.sum((y - np.mean(y)) ** 2)

        if r2 > 1:
            return None

        return {
            "expression": str(expression),
            "skeleton": expr_candidate,
            "traversal": traversal,
            "mse": mse,
            "r2": r2,
            "reward": r2 * (1 - nrmse),
        }

    # Create a vectorized lambdified function including constants
    all_symbols = list(sym_vars) + list(const_symbols)
    numeric_func = sp.lambdify(all_symbols, expression, modules=MODULES)

    # 7. Define the loss function
    def loss_function(consts):
        # consts is a list or array of constants [c0, c1, ...]
        (*var_values,) = X.T  # Unpack variable columns
        c_values = consts
        preds = numeric_func(*var_values, *c_values)
        return np.mean((preds - y) ** 2)

    # 8. Optimization function
    def optimize_expression(seed):
        np.random.seed(seed)
        # Initialize constants
        if const_candidate:
            # Choose randomly from const_candidate with 20% probability
            x0 = np.array(
                [
                    (
                        np.random.choice(const_candidate)
                        if np.random.rand() < 0.2
                        else np.random.randn()
                    )
                    for _ in range(n_const)
                ],
                dtype=np.float64,
            )
        else:
            x0 = np.array(
                [
                    (
                        np.random.randint(-10, 10)
                        if np.random.rand() < 0.5
                        else np.random.randn()
                    )
                    for _ in range(n_const)
                ],
                dtype=np.float64,
            )

        try:
            # Perform BFGS optimization
            res = minimize(loss_function, x0, method="BFGS")
            optimized_consts = res.x
        except Exception as e:
            print(f"Encountered in minimize: {e} | {expr_candidate}")
            # traceback.print_exc()
            return None

        # Create a numeric function with constants substituted
        try:
            # Substitute optimized constants into the expression
            substituted_expr = expression.subs(
                {const_symbols[i]: optimized_consts[i] for i in range(n_const)}
            )
            if "I" in str(expression):
                return None
            final_func = sp.lambdify(sym_vars, substituted_expr, modules=MODULES)
            y_pred = final_func(*X.T)
        except Exception as e:
            print(f"Encountered in lambdify: {e} | {expr_candidate}")
            traceback.print_exc()
            return None

        if not np.all(np.isfinite(y_pred)):
            return None

        mse = np.mean((y_pred - y) ** 2)
        rmse = np.sqrt(mse)
        nrmse = rmse / np.std(y)
        r2 = 1 - np.sum((y_pred - y) ** 2) / np.sum((y - np.mean(y)) ** 2)

        if r2 > 1:
            return None

        return {
            "expression": str(substituted_expr),
            "skeleton": expr_candidate,
            "traversal": traversal,
            "mse": mse,
            "r2": r2,
            "reward": r2 * (1 - nrmse),
        }

    # 9. Parallel optimization attempts
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(optimize_expression)(seed) for seed in range(n_try)
    )

    # 10. Filter out failed optimizations
    valid_results = [
        {
            "expression": res["expression"],
            "skeleton": expr_candidate,
            "traversal": traversal,
            "mse": float(res["mse"]),
            "r2": float(res["r2"]),
            "reward": float(res["reward"]),
        }
        for res in results
        if res is not None
        and np.isscalar(res["r2"])
        and np.isscalar(res["mse"])
        and not np.iscomplex(res["r2"])
        and not np.iscomplex(res["mse"])
        and not math.isnan(res["r2"])
        and not math.isnan(res["mse"])
    ]

    if not valid_results:
        # If all optimizations failed, return None
        return None

    # 11. Select the best result based on reward
    best_result = max(valid_results, key=lambda x: (x["r2"], -x["mse"]))

    return best_result


if __name__ == "__main__":
    import warnings
    from scipy.optimize.linesearch import LineSearchWarning

    # Test the BFGS optimizer
    expr_candidate = "x_1 * x_3 * c + x_2 ** 2"

    n_points = 300  # Number of data points
    x1 = np.linspace(-10, 10, n_points)
    x2 = np.linspace(-10, 10, n_points)
    x3 = np.linspace(-10, 10, n_points)
    X = np.stack([x1, x2, x3], axis=1)
    y = x1 * 0.7 * x3 + x2**2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=LineSearchWarning)
        print(bfgs(expr_candidate,[], X, y, const_candidate=[2], n_try=100, n_jobs=-1))
