"""
Improved BFGS optimization algorithm implementation.
This module provides a robust implementation of the BFGS optimization algorithm,
specifically tailored for symbolic regression use cases.

Key improvements over the original:
1. Unified NumPy usage - no more Torch/NumPy/SymPy type mixing
2. Consistent modules usage for optimization and evaluation
3. Numerical domain loss computation instead of symbolic
4. Proper gradient computation for faster convergence
5. Domain protection for problematic functions
6. Better error handling and NaN protection
7. True timeout control with callback
8. Fun/grad caching to avoid duplicate computations
9. Only include constants that actually appear in expressions
10. Improved no_domain semantics with proper flagging
11. Safe coth implementation and optional bounds
12. Optional data preprocessing with better defaults
"""

import time
import re
import warnings
from typing import Tuple, Union, Optional, List, Dict, Any

import numpy as np
import sympy as sp
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Safe division helper
def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Safe division that handles near-zero denominators."""
    return a / np.where(np.abs(b) < eps, np.sign(b) * eps, b)

# Enhanced module dictionary with domain protection and safe operations
MODULES = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": lambda x: np.arcsin(np.clip(x, -1+1e-12, 1-1e-12)),
    "acos": lambda x: np.arccos(np.clip(x, -1+1e-12, 1-1e-12)),
    "atan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "coth": lambda x: _safe_div(np.cosh(x), np.sinh(x)),
    "sqrt": lambda x: np.sqrt(np.clip(x, 0, None)),
    "log": lambda x: np.log(np.clip(x, 1e-12, None)),
    "exp": np.exp,
    "Abs": np.abs,
    "numpy": np,
}


def make_timeout_cb(limit_sec: float):
    """Create a timeout callback for scipy.optimize.minimize."""
    t0 = time.perf_counter()
    def _cb(xk):
        if time.perf_counter() - t0 > limit_sec:
            raise RuntimeError("timeout")
    return _cb


class FunGradCache:
    """Cache for function and gradient to avoid duplicate computations."""
    def __init__(self, fg):
        self.fg = fg
        self.x = None
        self.cache = None
    
    def fun(self, c):
        if self.x is None or not np.array_equal(c, self.x):
            self.cache = self.fg(c)
            self.x = c.copy()
        return self.cache[0]
    
    def jac(self, c):
        if self.x is None or not np.array_equal(c, self.x):
            self.cache = self.fg(c)
            self.x = c.copy()
        return self.cache[1]


class TimedFun:
    """Time-limited function wrapper for optimization (legacy compatibility)."""
    def __init__(self, fun, stop_after=10):
        self.fun_in = fun
        self.started = False
        self.stop_after = stop_after

    def fun(self, x, *args):
        if self.started is False:
            self.started = time.time()
        elif abs(time.time() - self.started) >= self.stop_after:
            raise ValueError("Time is over.")
        self.fun_value = self.fun_in(*x, *args)
        self.x = x
        return self.fun_value


def _to_numpy(array: Union[np.ndarray, Any]) -> np.ndarray:
    """Convert various array types to numpy array."""
    # Handle torch tensors if torch is available
    try:
        import torch
        if isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy()
    except ImportError:
        pass
    
    return np.asarray(array, dtype=np.float64)


def _prepare_data(X: Union[np.ndarray, Any], y: Union[np.ndarray, Any], 
                  drop_large: bool = False, zero_dim_to_one: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare and validate input data."""
    # Convert to numpy with consistent dtype
    X = _to_numpy(X)
    y = _to_numpy(y).squeeze()
    
    # Handle 3D tensor case (batch, samples, features)
    if X.ndim == 3:
        X = X.reshape(-1, X.shape[-1])
        y = y.reshape(-1)
    
    # Ensure 2D input with 2 features
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"Expected X to be 2D with 2 features, got shape {X.shape}")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")
    
    # Optional data preprocessing (disabled by default for better stability)
    if drop_large:
        # Remove samples with high values (original logic)
        mask = (X < 200).all(axis=1)
        X, y = X[mask], y[mask]
        if len(X) == 0:
            raise ValueError("All samples removed by drop_large filter")
    
    if zero_dim_to_one:
        # Handle unused dimensions (original logic)
        zero_dim = (X == 0).all(axis=0)
        X[:, zero_dim] = 1
    
    return X, y


def _prepare_expression(pred_str: str) -> Tuple[sp.Expr, List[sp.Symbol], str]:
    """Prepare symbolic expression and extract only constants that actually appear."""
    # First, replace standalone 'c' with 'constant' placeholder
    candidate = re.sub(r"\bc\b", "constant", pred_str)
    
    # Replace 'constant' with c0, c1, etc.
    for i in range(candidate.count("constant")):
        candidate = candidate.replace("constant", f"c{i}", 1)
    
    # Create symbols and parse expression with locals for safety
    x1, x2 = sp.symbols("x_1 x_2", real=True)
    expr = sp.sympify(candidate, locals={"x_1": x1, "x_2": x2})
    
    # Only extract constants that actually appear in the expression
    present = sorted({
        s for s in expr.free_symbols
        if s.name.startswith("c") and s.name[1:].isdigit()
    }, key=lambda s: int(s.name[1:]))
    
    c_syms = list(present)
    
    return expr, c_syms, candidate


def _create_loss_function(expr: sp.Expr, c_syms: List[sp.Symbol], X: np.ndarray, y: np.ndarray) -> Tuple[callable, callable, Dict[str, bool]]:
    """Create loss function and gradient function for optimization."""
    x1, x2 = sp.symbols("x_1 x_2", real=True)
    
    # Create prediction function - handle both cases properly
    if c_syms:
        pred_fn = sp.lambdify((x1, x2, *c_syms), expr, modules=[MODULES, "numpy"])
    else:
        pred_fn = sp.lambdify((x1, x2), expr, modules=[MODULES, "numpy"])
    
    # Create gradient functions
    if c_syms:
        d_expr = [sp.diff(expr, ci) for ci in c_syms]
        d_pred_fn = sp.lambdify((x1, x2, *c_syms), d_expr, modules=[MODULES, "numpy"])
    else:
        d_pred_fn = None
    
    X1, X2 = X[:, 0], X[:, 1]
    N = X.shape[0]
    
    # Domain error flag
    domain_err_flag = {"hit": False}
    
    def loss_and_grad(c: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute loss and gradient."""
        try:
            # Compute predictions
            if c_syms and len(c) > 0:
                yhat = pred_fn(X1, X2, *c)
            else:
                yhat = pred_fn(X1, X2)
            
            # Ensure yhat is 1D numpy array
            yhat = np.asarray(yhat, dtype=float).reshape(-1)
            
            # Check for invalid values
            if not np.all(np.isfinite(yhat)) or yhat.shape != y.shape:
                domain_err_flag["hit"] = True
                return 1e12, np.zeros(len(c_syms), dtype=float)
            
            # Compute loss
            resid = yhat - y
            loss = float(np.mean(resid * resid))
            
            # Compute gradient
            if c_syms and d_pred_fn is not None and len(c) > 0:
                g_preds = d_pred_fn(X1, X2, *c)
                if len(c_syms) == 1:
                    g_preds = [np.asarray(g_preds, dtype=float).reshape(-1)]
                else:
                    g_preds = [np.asarray(g, dtype=float).reshape(-1) for g in g_preds]
                
                grad = np.array([
                    (2.0 / N) * np.sum(resid * g_preds[i]) 
                    for i in range(len(c_syms))
                ], dtype=float)
                
                # Handle invalid gradients
                if not np.all(np.isfinite(grad)):
                    domain_err_flag["hit"] = True
                    grad = np.zeros(len(c_syms), dtype=float)
            else:
                grad = np.array([], dtype=float)
            
            return loss, grad
            
        except Exception as e:
            # Return large penalty for any error
            domain_err_flag["hit"] = True
            return 1e12, np.zeros(len(c_syms), dtype=float)
    
    return loss_and_grad, pred_fn, domain_err_flag


def bfgs(pred_str: str, X: Union[np.ndarray, Any], y: Union[np.ndarray, Any], 
         drop_large: bool = True, zero_dim_to_one: bool = True, 
         n_restarts: int = 20, max_tries: int = 50, time_limit: float = 10.0,
         use_bounds: bool = True, bounds_range: float = 10.0) -> Tuple[sp.Expr, np.ndarray, float, sp.Expr, bool]:
    """
    BFGS optimization for symbolic regression.
    
    Args:
        pred_str: String representation of the symbolic expression
        X: Input features (N, 2) or (B, N, 2) tensor/array
        y: Target values (N,) or (B, N) tensor/array
        drop_large: Whether to remove samples with high values (default: True for compatibility)
        zero_dim_to_one: Whether to set zero dimensions to 1 (default: True for compatibility)
        n_restarts: Number of valid restarts to achieve
        max_tries: Maximum number of attempts
        time_limit: Time limit in seconds
        use_bounds: Whether to use bounds for constants
        bounds_range: Range for constant bounds (-bounds_range, bounds_range)
        
    Returns:
        final_expr: Optimized symbolic expression
        consts: Optimized constants
        final_loss: Final loss value
        expr: Original expression
        no_domain: Whether domain errors occurred
    """
    # Prepare data with optional preprocessing
    X, y = _prepare_data(X, y, drop_large=drop_large, zero_dim_to_one=zero_dim_to_one)
    
    # Prepare expression
    expr, c_syms, candidate = _prepare_expression(pred_str)
    
    # Create loss function with domain error tracking
    loss_and_grad, pred_fn, domain_err_flag = _create_loss_function(expr, c_syms, X, y)
    
    # Create fun/grad cache for efficiency
    fg_cache = FunGradCache(loss_and_grad)
    
    # Set up bounds if requested
    bounds = None
    if use_bounds and c_syms:
        bounds = [(-bounds_range, bounds_range)] * len(c_syms)
    
    # Store results
    F_loss = []
    consts_ = []
    funcs = []
    
    # Multiple restarts with improved random seed handling
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    num_valid = 0
    tryout = 0
    start_time = time.perf_counter()
    
    while num_valid < n_restarts and tryout < max_tries:
        if (time.perf_counter() - start_time) > time_limit:
            break
            
        # Random initialization
        if c_syms:
            x0 = rng.normal(0.0, 1.0, size=len(c_syms))
        else:
            x0 = np.array([])
        
        try:
            # Create timeout callback
            cb = make_timeout_cb(time_limit - (time.perf_counter() - start_time))
            
            # Optimize using L-BFGS-B with gradient and caching
            if c_syms:
                res = minimize(
                    fun=fg_cache.fun,
                    x0=x0,
                    method="L-BFGS-B",
                    jac=fg_cache.jac,
                    bounds=bounds,
                    options={"maxiter": 1000, "ftol": 1e-12},
                    callback=cb
                )
                consts_.append(res.x)
            else:
                consts_.append(np.array([]))
            
            # Create final expression
            final_expr = expr
            if c_syms and len(consts_[-1]) > 0:
                for ci, val in zip(c_syms, consts_[-1]):
                    final_expr = final_expr.subs(ci, float(val))
            
            funcs.append(final_expr)
            
            # Compute final loss
            X1, X2 = X[:, 0], X[:, 1]
            if c_syms and len(consts_[-1]) > 0:
                y_found = pred_fn(X1, X2, *consts_[-1])
            else:
                y_found = pred_fn(X1, X2)
            
            y_found = np.asarray(y_found, dtype=float).reshape(-1)
            final_loss = float(np.mean(np.square(y_found - y)))
            F_loss.append(final_loss)
            
            # Check if valid
            if not np.isnan(final_loss) and np.isfinite(final_loss):
                num_valid += 1
                
        except Exception as e:
            print(f"Encountered in bfgs: {e}")
            # Add dummy results for failed attempts
            consts_.append(np.zeros(len(c_syms), dtype=float))
            funcs.append(expr)
            F_loss.append(1e12)
        
        tryout += 1
    
    # Select best result
    try:
        k_best = np.nanargmin(F_loss)
    except ValueError:
        k_best = 0
    
    # Use domain error flag for more accurate no_domain detection
    no_domain = domain_err_flag["hit"]
    
    return funcs[k_best], consts_[k_best], F_loss[k_best], expr, no_domain
