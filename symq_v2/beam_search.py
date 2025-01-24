import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
import copy
from typing import List, Dict, Any
from env import SymbolicWorldEnv


def beam_search(model, cfg, X, y, mask, beam_size, n_jobs=1, penalize_length=True):
    """
    Performs beam search using deep copies of the environment to avoid replaying action sequences.

    Args:
        model: The model used to predict actions.
        cfg: Configuration object containing necessary parameters.
        X: Input features.
        y: Target values.
        mask: Mask tensor for the model.
        beam_size: The beam size for beam search.
        n_jobs: Number of parallel jobs.

    Returns:
        A list of completed candidates sorted by their cumulative scores.
    """
    max_step = cfg["max_step"]
    device = model.device

    # Initialize tensors directly on the target device
    if cfg.num_vars ==2 and cfg.SymQ.use_pretrain:
        n_vars = 3
    else:
        n_vars = cfg.num_vars
    X_tensor = torch.zeros(
        (X.shape[0], n_vars), device=device, dtype=torch.float32
    )
    X_tensor[:, : X.shape[1]] = torch.as_tensor(X, device=device, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, device=device, dtype=torch.float32)
    raw_point_set = torch.cat((X_tensor, y_tensor.unsqueeze(1)), dim=1).unsqueeze(0)
    raw_point_set = raw_point_set.permute(0, 2, 1)

    # Calculate the number of non-zero variables
    n_x = (X_tensor != 0).any(dim=0).sum().item()
    required_vars = {f"x_{i}" for i in range(1, n_x + 1)}

    # Initialize the initial environment
    initial_env = SymbolicWorldEnv(cfg)
    n_actions = initial_env.action_space.n
    initial_env.reset()
    tree = torch.as_tensor(initial_env.tree, device=device, dtype=torch.float32)

    # Get initial action scores and select top-k actions
    _, q_values = model.act(raw_point_set, tree, mask)
    log_probs = F.log_softmax(q_values, dim=-1)
    topk_vals, topk_indices = torch.topk(
        log_probs, max_step, dim=-1
    )  # Shape: [batch, max_step]

    # Containers for candidates and completed sequences
    candidates: List[Dict[str, Any]] = []
    done_candidates: List[Dict[str, Any]] = []

    # Helper function to step each beam candidate
    def step_candidate(action, value, env_instance):
        """
        Initializes a candidate by applying an action to the environment.

        Args:
            action: The action to apply.
            value: The log probability of the action.
            env_instance: An instance of SymbolicWorldEnv.

        Returns:
            A dictionary representing the candidate or None if invalid.
        """
        local_env = copy.deepcopy(env_instance)
        _, _, done, _, info = local_env.step(action.item())

        if not done:
            return {
                "cumulative_score": value.item(),
                "env": local_env,
            }
        else:
            if info["terminated"] == "time_out":
                return None

            skeleton = info["agent_expr"].skeleton

            # Validate skeleton
            if not all([var in skeleton for var in required_vars]):
                return None

            return {
                "cumulative_score": (
                    value.item() if penalize_length else value.item() / local_env.n_step
                ),
                "skeleton": skeleton,
                "traversal": info["agent_expr"].opt_sequence,
            }

    # Initialize candidates in parallel
    env_init_results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(step_candidate)(topk_indices[0, idx], topk_vals[0, idx], initial_env)
        for idx in range(topk_indices.size(1))
    )

    for res in env_init_results:
        if res is None:
            continue
        if "skeleton" in res:
            done_candidates.append(res)
        else:
            candidates.append(res)

    # Beam search iterations
    for _ in range(1, max_step):
        if not candidates:
            break  # No more candidates to expand

        # Prepare batched inputs for the model
        # Assuming each environment has 'point_set' and 'tree' attributes
        batched_point_set = torch.cat([raw_point_set for _ in candidates], dim=0)
        batched_tree = torch.cat(
            [
                torch.as_tensor(
                    candidate["env"].tree, device=device, dtype=torch.float32
                )
                for candidate in candidates
            ],
            dim=0,
        )

        # Get model predictions
        _, q_values = model.act(batched_point_set, batched_tree, mask)
        log_probs = F.log_softmax(q_values, dim=-1)

        # Get cumulative scores
        pre_scores = (
            torch.tensor(
                [candidate["cumulative_score"] for candidate in candidates],
                device=device,
            )
            .unsqueeze(1)
            .expand(-1, n_actions)
        )

        # Combine scores
        combined_scores = log_probs + pre_scores  # Shape: [batch_size, n_actions]

        # Flatten and get top-k
        flattened_scores = combined_scores.view(-1)
        num_topk = min(beam_size * 2, flattened_scores.size(0))
        top_values, top_indices = torch.topk(flattened_scores, num_topk)

        # Decode indices to candidate and action indices
        candidate_indices = top_indices // n_actions
        action_indices = top_indices % n_actions

        # Prepare tasks for parallel execution
        expand_tasks = [
            (
                action_indices[idx],
                top_values[idx],
                candidates[candidate_indices[idx].item()]["env"],
            )
            for idx in range(top_indices.numel())
        ]

        # Expand candidates in parallel
        expand_results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(step_candidate)(action_idx, value, env_instance)
            for action_idx, value, env_instance in expand_tasks
        )

        new_candidates = []
        for res in expand_results:
            if res is None:
                continue
            if "skeleton" in res:
                done_candidates.append(res)
            else:
                new_candidates.append(res)

        # Trim the done_candidates to maintain beam size
        if len(done_candidates) > beam_size:
            done_candidates.sort(key=lambda x: x["cumulative_score"], reverse=True)
            done_candidates = done_candidates[:beam_size]

        # Update candidates for the next step
        candidates = new_candidates

    # Final sorting of completed candidates
    done_candidates.sort(key=lambda x: x["cumulative_score"], reverse=True)

    return done_candidates
