import pathlib
import warnings
from joblib import Parallel, delayed

import torch
import numpy as np
import sympy as sp
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize.linesearch import LineSearchWarning

from SymQ import SymQ
from bfgs import bfgs
from beam_search import beam_search
from util import load_cfg, get_seq2action, run_with_timeout, MODULES

num_vars = 2

class SymQRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        # Load the configuration file
        file_path = pathlib.Path(__file__).parent.absolute()
        self.cfg = load_cfg(f"{file_path}/cfg_{num_vars}var.yaml")
        self.seq2action = get_seq2action(self.cfg)
        weights_path = f"{file_path}/{self.cfg.Inference.weights_path}"

        # Define the device
        if self.cfg.Inference.use_gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        # Load the model
        self.model = SymQ(self.cfg, device).to(device)
        try:
            model_state_dict = torch.load(weights_path, map_location="cpu")[
                "model_state_dict"
            ]
            state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
        except KeyError:
            model_state_dict = torch.load(weights_path, map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.sympy_expr = None
        self.traversal = []
        self.traversals = []
        self.lambdified_func = None
        self.variables_in_expr = None
        self.candidates = []
        self.normalized = self.cfg.Inference.normalized
        self.normalization_val = []

    def fit(self, X, y, **kwargs):
        self.sympy_expr = None
        self.traversal = []
        self.traversals = []
        self.lambdified_func = None
        self.variables_in_expr = None
        self.candidates = []
        self.normalization_val = []

        if X.shape[1] > self.cfg.num_vars:
            raise ValueError(
                f"The number of variables in X is greater than the number of variables in the model, in this case {self.cfg.num_vars}."
            )

        # Mask for the variables
        mask = torch.ones(len(self.seq2action)).to(self.model.device)

        for idx in range(X.shape[1]):
            if (X[:, idx] == 0).all():
                var_name = f"x_{idx+1}"
                if var_name in self.seq2action:
                    mask[self.seq2action[var_name]] = 0

        # Normalize the data
        if self.normalized:
            x_mean = np.mean(X, axis=0)
            x_std = np.std(X, axis=0)
            self.normalization_val = [x_mean, x_std]
            # Normalize the data to -10 to 10
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                X = (X - x_mean) / x_std * 10
            X[np.isnan(X)] = 0

        # Subset the data
        n_points = self.cfg.Inference.n_points
        # Generate a random permutation of indices
        indices = np.random.permutation(X.shape[0])

        # Shuffle X and y using the permutation indices
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        X = X_shuffled[:n_points, :]
        y = y_shuffled[:n_points]

        # Run the beam search
        done_candidates = beam_search(
            self.model,
            self.cfg,
            X,
            y,
            mask,
            self.cfg.Inference.beam_size,
            self.cfg.Inference.n_jobs,
            self.cfg.Inference.penalize_length,
        )

        self.traversals = [can["traversal"] for can in done_candidates]

        # Run the BFGS optimization
        if "const_candidate" in kwargs:
            const_candidate = kwargs["const_candidate"]
        else:
            const_candidate = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=LineSearchWarning)
            cans = Parallel(n_jobs=self.cfg.Inference.n_jobs, backend="threading")(
                delayed(run_with_timeout)(
                    bfgs,
                    args=(
                        candidate["skeleton"],
                        candidate["traversal"],
                        X,
                        y,
                        const_candidate,
                        self.cfg.Inference.n_try,
                        self.cfg.Inference.n_jobs,
                    ),
                    timeout=self.cfg.Inference.timeout,
                )
                for candidate in done_candidates
            )

            # Select the best candidate
            valid_candidates = [can for can in cans if can is not None]
            if not valid_candidates:
                raise ValueError("No valid candidates found.")
            self.candidates = valid_candidates
            best_candidate = max(valid_candidates, key=lambda x: (x["r2"], -x["mse"]))
            expr = best_candidate["expression"]
            if self.normalized:
                x_mean, x_std = self.normalization_val
                for idx in range(X.shape[1]):
                    expr = expr.replace(f"x_{idx+1}", f"(x_{idx+1} - {x_mean[idx]}) / {x_std[idx]} * 10")
            self.sympy_expr = sp.sympify(expr)
            self.traversal = best_candidate["traversal"]

            # Precompile the lambdified function
            self.variables_in_expr = sorted(
                [str(s) for s in self.sympy_expr.free_symbols],
                key=lambda s: int(s.split("_")[1]),
            )
            self.lambdified_func = sp.lambdify(
                self.variables_in_expr, self.sympy_expr, modules=MODULES
            )

        return self

    def predict(self, X):
        # Map variable names to their corresponding columns in X
        var_indices = [int(s.split("_")[1]) - 1 for s in self.variables_in_expr]
        inputs = [X[:, idx] for idx in var_indices]
        return self.lambdified_func(*inputs)


if __name__ == "__main__":
    reg = SymQRegressor()
