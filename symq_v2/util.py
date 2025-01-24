import numpy as np
import torch
import sympy
from sympy.parsing.sympy_parser import parse_expr
from sympy import Float, preorder_traversal, Integer, simplify

import yaml
from box import Box
import multiprocessing
import warnings


def load_cfg(path: str) -> Box:
    with open(path, "r") as file:
        # Load the YAML content
        cfg = Box(yaml.safe_load(file))
    return cfg


# seq_to_action = {
#     "x_1": 0,
#     "x_2": 1,
#     "c": 2,
#     "abs": 3,
#     "add": 4,
#     "mul": 5,
#     "div": 6,
#     "sqrt": 7,
#     "exp": 8,
#     "log": 9,
#     "pow": 10,
#     "sin": 11,
#     "cos": 12,
#     "tan": 13,
#     "asin": 14,
#     "acos": 15,
#     "atan": 16,
#     "sinh": 17,
#     "cosh": 18,
#     "tanh": 19,
#     "coth": 20,
#     "-3": 21,
#     "-2": 22,
#     "-1": 23,
#     "0": 24,
#     "1": 25,
#     "2": 26,
#     "3": 27,
#     "4": 28,
#     "5": 29,
# }


BENCHMARK = {
    "Nguyen_1": "x_1**3 + x_1**2 + x_1",
    "Nguyen_2": "x_1**4 + x_1**3 + x_1**2 + x_1",
    "Nguyen_3": "x_1**5 + x_1**4 + x_1**3 + x_1**2 + x_1",
    "Nguyen_4": "x_1**6 + x_1**5 + x_1**4 + x_1**3 + x_1**2 + x_1",
    "Nguyen_5": "sin(x_1**2)*cos(x_1) - 1",
    "Nguyen_6": "sin(x_1) + sin(x_1 + x_1**2)",
    "Nguyen_7": "log(x_1+1)+log(x_1**2+1)",
    "Nguyen_8": "x_1**0.5",
    "Nguyen_9": "sin(x_1) + sin(x_2**2)",
    "Nguyen_10": "2*sin(x_1)*cos(x_2)",
    "Nguyen_11": "x_1**x_2",
    "Nguyen_12": "x_1**4-x_1**3+0.5*x_2**2-x_2",
    # "Nguyen_2_": "4 * x_1**4 + x_1**3 + x_1**2 + x_1",
    # "Nguyen_5_": "sin(x_1**2)*cos(x_1) - 2",
    # "Nguyen_8_": "x_1**(1/3)",
    # "Nguyen_8__": "x_1**(2/3)",
    # "Nguyen_1c": "3.39*x_1**3 + 2.12*x_1**2 + 1.78*x_1",
    # "Nguyen_5c": "sin(x_1**2)*cos(x_1) - 0.75",
    # "Nguyen_7c": "log(x_1+1.4)+log(x_1**2+1.3)",
    # "Nguyen_8c": "(1.23 * x_1)**0.5",
    # "Nguyen_10c": "sin(1.5*x_1)*cos(0.5*x_2)",
    "Keijzer_3": "0.3 * x_1 * sin(2*pi*x_1)",
    "Keijzer_4": "x_1**3*exp(-x_1)*cos(x_1)*sin(x_1)*(sin(x_1)**2 *cos(x_1)-1)",
    "Keijzer_6": "x_1 * (x_1+1) * 0.5",
    "Keijzer_7": "log(x_1)",
    "Keijzer_8": "sqrt(x_1)",
    "Keijzer_9": "log(x_1+sqrt(x_1**2+1))",
    "Keijzer_10": "x_1**x_2",
    "Keijzer_11": "x_1*x_2+sin((x_1-1)*(x_2-1))",
    "Keijzer_12": "x_1**4 - x_1**3 + 0.5*x_2**2 - x_2",
    "Keijzer_13": "6*sin(x_1)*cos(x_2)",
    "Keijzer_14": "8/(2+x_1**2+x_2**2)",
    "Keijzer_15": "0.2*x_1**3+0.5*x_2**3-x_2-x_1",
    "Constant_1": "3.39*x_1**3+2.12*x_1**2+1.78*x_1",
    "Constant_2": "sin(x_1**2)*cos(x_1)-0.75",
    "Constant_3": "sin(1.5*x_1)*cos(0.5*x_2)",
    "Constant_4": "2.7*x_1**x_2",
    "Constant_5": "sqrt(1.23*x_1)",
    "Constant_6": "x_1**0.426",
    "Constant_7": "2*sin(1.3*x_1)*cos(x_2)",
    "Constant_8": "log(x_1+1.4)+log(x_1**2+1.3)",
    "R_1": "(x_1+1)**3/(x_1**2-x_1+1)",
    "R_2": "(x_1**5-3*x_1**3+1)/(x_1**2+1)",
    "R_3": "(x_1**6+x_1**5)/(x_1**4+x_1**3+x_1**2+x_1+1)",
    "Feynman_1": "exp(-x_1**2/2)/sqrt(2*pi)",
    "Feynman_2": "exp(-(x_1/x_2)**2/2)/(sqrt(2*pi)*x_2)",
    "Feynman_3": "x_1*x_2",
    "Feynman_4": "x_1*x_2",
    "Feynman_5": "0.5*x_1*x_2**2",
    "Feynman_6": "x_1/x_2",
    "Feynman_7": "1.5*x_1*x_2",
    "Feynman_8": "x_1/(4*3.14*x_2**2)",
    "Feynman_9": "(x_1*x_2**2)/2",
    "Feynman_10": "x_1*x_2**2",
    "Feynman_11": "x_1/(2*(1+x_2))",
    "Feynman_12": "x_1*x_2/(2*3.14)",
}


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seq_to_tree(sequence, cfg):
    max_step = cfg.max_step
    seq2action = get_seq2action(cfg)
    tree = torch.zeros((max_step, len(seq2action)), dtype=torch.float32)
    for i, op_seq in enumerate(sequence):
        tree[i, op_seq] = 1

    return tree


def handle_timeout(signum, frame):
    raise TimeoutError("Time out")


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def comp_wn(model):
    K = 0
    N = 0
    for p in model.parameters():
        N += torch.sum(p**2)
        K += p.numel()
    return torch.sqrt(N / K)


def get_seq2action(cfg):
    num_vars = cfg.num_vars
    if num_vars < 1 or num_vars > 3:
        raise ValueError("Invalid number of variables")

    seq2action = {
        "c": 0,
        "abs": 1,
        "add": 2,
        "mul": 3,
        "div": 4,
        "sqrt": 5,
        "exp": 6,
        "ln": 7,
        "pow": 8,
        "sin": 9,
        "cos": 10,
        "tan": 11,
        "-3": 12,
        "-2": 13,
        "-1": 14,
        "0": 15,
        "1": 16,
        "2": 17,
        "3": 18,
        "4": 19,
        "5": 20,
        "asin": 21,
        "acos": 22,
        "atan": 23,
        "sinh": 24,
        "cosh": 25,
        "tanh": 26,
        "coth": 27,
        "x_1": 28,
    }

    if num_vars == 2:
        seq2action["x_2"] = 29
    elif num_vars == 3:
        seq2action["x_2"] = 29
        seq2action["x_3"] = 30

    return seq2action


def run_with_timeout(func, args=(), kwargs={}, timeout=20):
    pool = multiprocessing.Pool(processes=1)
    result = pool.apply_async(func, args=args, kwds=kwargs)
    try:
        return result.get(timeout=timeout)
    except multiprocessing.TimeoutError:
        pool.terminate()
        pool.join()
        return None


def filter_valid_points(X, Y, value_bound=5e4):
    valid_indices = np.isfinite(Y) & np.isreal(Y) & (np.abs(Y) < value_bound)
    X_valid = X[valid_indices, :]
    Y_valid = Y[valid_indices]
    return X_valid, Y_valid


def generateDataFast(eq, n_points, n_vars, decimals, min_x, max_x):
    total_variables = [f"x_{i}" for i in range(1, n_vars + 1)]
    symbols = sympy.symbols(total_variables)
    expr = sympy.sympify(eq)
    lambdified_expr = sympy.lambdify(symbols, expr, "numpy")

    initial_n_points = n_points * 30  # Generate 30 times more points initially
    X = np.round(
        np.random.uniform(min_x, max_x, (initial_n_points, n_vars)),
        decimals=decimals,
    )

    # Get the set of variables actually used in the expression
    used_variables = {str(s) for s in expr.free_symbols}
    unused_indices = [
        i for i, var in enumerate(total_variables) if var not in used_variables
    ]
    if unused_indices:
        X[:, unused_indices] = 0.0

    # Evaluate the expression for all generated values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        Y = lambdified_expr(*X.T)

    if np.isscalar(Y):
        return None, None

    # Filter out invalid values (NaN, Inf, or complex numbers)
    X_valid, Y_valid = filter_valid_points(X, Y)

    # Check if the number of valid points is less than the required number
    num_valid_points = Y_valid.size
    if num_valid_points < n_points:
        return None, None
    else:
        # Select the required subset of valid points
        return X_valid[:n_points, :], Y_valid[:n_points]


# Custom module dictionary
MODULES = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    # Define 'coth' using np.cosh and np.sinh as NumPy doesn't have a direct coth function
    "coth": lambda x: np.cosh(x) / np.sinh(x),
    "sqrt": np.sqrt,
    "log": np.log,
    "ln": np.log,
    "exp": np.exp,
    "abs": np.abs,
    "numpy": np,  # Include numpy for other functions and operations
    "re": np.real,
    "im": np.imag,
    "Max": np.maximum,
    "Min": np.minimum,
    "floor": np.floor,
    "ceiling": np.ceil,
    "sign": np.sign,
}


def round_floats(ex1):
    ex2 = ex1

    for a in preorder_traversal(ex1):
        if isinstance(a, Float):
            if abs(a) < 0.0001:
                ex2 = ex2.subs(a, Integer(0))
            else:
                ex2 = ex2.subs(a, Float(round(a, 3), 3))
    return ex2


def parse_equation(expr, n_vars):
    def sub(x, y):
        return sympy.Add(x, -y)

    def div(x, y):
        return sympy.Mul(x, 1 / y)

    def square(x):
        return sympy.Pow(x, 2)

    def cube(x):
        return sympy.Pow(x, 3)

    def quart(x):
        return sympy.Pow(x, 4)

    def PLOG(x, base=None):
        if isinstance(x, sympy.Float):
            if x < 0:
                x = sympy.Abs(x)
        if base:
            return sympy.log(x, base)
        else:
            return sympy.log(x)

    def PSQRT(x):
        if isinstance(x, sympy.Float):
            if x < 0:
                return sympy.sqrt(sympy.Abs(x))
        return sympy.sqrt(x)

    local_dict = {f"x_{i}": sympy.Symbol(f"x_{i}") for i in range(1, n_vars + 1)}

    local_dict.update(
        {
            "add": sympy.Add,
            "mul": sympy.Mul,
            "max": sympy.Max,
            "min": sympy.Min,
            "sub": sub,
            "div": div,
            "square": square,
            "cube": cube,
            "quart": quart,
            #    'PLOG':PLOG,
            #    'PLOG10':PLOG,
            "PSQRT": PSQRT,
        }
    )

    model_sym = parse_expr(expr, local_dict=local_dict)
    model_sym = round_floats(model_sym)
    model_sym = simplify(model_sym, ratio=1)
    return model_sym

def post_process_symexpr(sym_diff, sym_frac):
    try:
        if not sym_diff.is_constant():
            sym_diff = round_floats(simplify(sym_diff, ratio=1))
        sym_diff_is_const = sym_diff.is_constant()
    except:
        sym_diff_is_const = False

    try:
        symbolic_frac_is_const = sym_frac.is_constant()
        if symbolic_frac_is_const is None:
            symbolic_frac_is_const = False
    except:
        symbolic_frac_is_const = False

    return (sym_diff, sym_frac, sym_diff_is_const, symbolic_frac_is_const)

def evaluate_equality(true_model, cleaned_model):
    sym_diff = round_floats(true_model - cleaned_model)
    sym_frac = round_floats(cleaned_model / true_model)

    results = run_with_timeout(post_process_symexpr, args=(sym_diff, sym_frac), timeout=120)

    if results is None:
        sym_diff, sym_frac, sym_diff_is_const, symbolic_frac_is_const = sym_diff, sym_frac, False, False
    else:
        sym_diff, sym_frac, sym_diff_is_const, symbolic_frac_is_const = results

    return {
        "true_model": str(true_model),
        "cleaned_model": str(cleaned_model),
        "symbolic_error": str(sym_diff),
        "symbolic_fraction": str(sym_frac),
        "symbolic_error_is_zero": str(sym_diff) == "0",
        "symbolic_error_is_constant": sym_diff_is_const,
        "symbolic_fraction_is_constant": symbolic_frac_is_const,
    }
