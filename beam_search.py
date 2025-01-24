import torch
import torch.nn.functional as F


def beam_search(model, env, point_set, expr, beam_size=128):
    raw_point_set = point_set
    max_step = model.cfg["num_actions"]
    obs, info = env.reset(raw_point_set, expr)

    point_set = obs["point_set"]
    tree = obs["tree"]

    _, q_values = model.act(point_set, tree)

    topk_vals, topk_indices = torch.topk(
        F.log_softmax(q_values, dim=-1), max_step, dim=-1
    )

    # Beam container
    candidates = []
    done_candidates = []

    # Fill containers with all actions on STEP 1
    for idx, action in enumerate(topk_indices[0]):
        env.reset(raw_point_set, expr)

        action_sequence = []
        obs, _, done, _, info = env.step(action.item())
        action_sequence.append(action.item())

        if not done:
            candidates.append(
                {
                    "sequence": action_sequence,
                    "cumulative_score": topk_vals[0][idx],
                    "score": topk_vals[0][idx],
                    "tree": obs["tree"],
                }
            )
        else:
            done_candidates.append(
                {
                    "sequence": action_sequence,
                    "cumulative_score": topk_vals[0][idx],
                    "score": topk_vals[0][idx],
                    "skeleton": info["agent_expr"].skeleton,
                    "opt_seq": info["agent_expr"].opt_sequence,
                    "R2": info["R2"],
                }
            )

    # Beam search after first action
    for _ in range(max_step - 1):
        # Exit if there are no more candidates
        if len(candidates) == 0:
            break

        trees = [candidate["tree"].squeeze(0) for candidate in candidates]
        trees = torch.stack(trees, dim=0)

        pre_scores = [candidate["cumulative_score"] for candidate in candidates]
        pre_scores = torch.stack(pre_scores, dim=0).unsqueeze(1)
        pre_scores = pre_scores.repeat(1, max_step)

        point_sets = point_set.repeat(len(candidates), 1, 1)

        _, q_values = model.act(point_sets, trees)
        probs = F.log_softmax(q_values, dim=-1) + pre_scores

        flattened_probs = probs.view(-1)
        top_values, top_indices = torch.topk(
            flattened_probs, min(beam_size * 2, len(candidates) * max_step)
        )

        # New candidates
        new_candidates = []

        for idx in range(top_indices.shape[0]):
            # Determine the original beam index and word index from flattened index
            beam_idx = top_indices[idx] // max_step
            opt_idx = top_indices[idx] % max_step

            # Copy old candidate
            old_candidate = candidates[beam_idx]

            # Check if the action leads to termination
            env.reset(raw_point_set, expr)

            for action in old_candidate["sequence"]:
                obs, _, done, _, info = env.step(action)

            # Apply operation and go to next step
            obs, _, done, _, info = env.step(opt_idx.item())

            action_sequence = old_candidate["sequence"].copy()
            action_sequence.append(opt_idx.item())

            if not done:
                new_candidates.append(
                    {
                        "sequence": action_sequence,
                        "cumulative_score": top_values[idx],
                        "score": top_values[idx],
                        "tree": obs["tree"],
                    }
                )
            else:
                score = top_values[idx] / len(action_sequence)
                if not np.isinf(score.cpu().item()):
                    done_candidates.append(
                        {
                            "sequence": action_sequence,
                            "cumulative_score": top_values[idx] / len(action_sequence),
                            "score": top_values[idx],
                            "skeleton": info["agent_expr"].skeleton,
                            "opt_seq": info["agent_expr"].opt_sequence,
                            "R2": info["R2"],
                        }
                    )

                if len(done_candidates) > beam_size:
                    # sort according to cumulative_score
                    done_candidates.sort(
                        key=lambda x: x["cumulative_score"], reverse=True
                    )
                    done_candidates = done_candidates[:beam_size]

        candidates = new_candidates

    done_candidates.sort(key=lambda x: x["cumulative_score"], reverse=True)

    return done_candidates


if __name__ == "__main__":
    import os
    import gc
    import ast
    import yaml
    import json
    import signal
    import argparse

    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    from tqdm import tqdm
    import numpy as np
    import sympy as sp

    from bfgs import bfgs
    from SymQ import SymQ
    from utils import BENCHMARK, fix_seed, handle_timeout
    from wrapper import load_dataset
    from symbolic_world import SymbolicWorldEnv

    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    sys.path.append(
        str(Path(__file__).parent.parent) + "/Joint_Supervised_Learning_for_SR/"
    )
    from Joint_Supervised_Learning_for_SR.src.utils import generateDataFast

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, default="")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu id")
    parser.add_argument("--target", type=str, default="", help="dataset")
    args = parser.parse_args()

    folder_path = os.path.dirname(args.weights_path)

    # Load config file
    cfg = yaml.load(open("cfg.yaml", "r"), Loader=yaml.FullLoader)

    # Initialize model and optimizer
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = SymQ(cfg, device).to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.eval()
    print(f"SymQ running on {device}")

    # SSDNC recovery
    if args.target == "ssdnc_seq_recovery":
        SSDNC_dataset = load_dataset("SSDNC", cfg)
        num_tests = len(SSDNC_dataset)
        env = SymbolicWorldEnv(cfg, cal_r2=False)
        record = {}

        for eq_num in tqdm(range(num_tests)):
            gc.collect()
            point_set, _, _, _, _, skeleton, seq, expr = SSDNC_dataset[eq_num]
            point_set = point_set.to(device).unsqueeze(0).transpose(1, 2)

            seq = ast.literal_eval(seq)
            done_candidates = beam_search(model, env, point_set, expr)

            rank = -1
            recovered = False

            for r, candidate in enumerate(done_candidates):
                if seq == candidate["opt_seq"] and rank == -1:
                    rank = r
                    recovered = True
                    break

            record[eq_num] = {
                "rank": rank,
                "recovered": recovered,
            }

            with open(f"{folder_path}/beam_search_SSDNC_recovery.json", "w") as f:
                json.dump(record, f, indent=5)

    # Benchmark
    if args.target == "benchmark":
        signal.signal(signal.SIGALRM, handle_timeout)
        env = SymbolicWorldEnv(cfg, cal_r2=False)

        for benchmark in ["Nguyen", "Keijzer", "Constant", "R", "Feynman"]:
            print(f"Evaluating {benchmark}...")
            record = {}

            for eq_name, expr in tqdm(BENCHMARK.items()):
                signal.alarm(0)
                if benchmark not in eq_name:
                    continue
                gc.collect()
                fix_seed(0)
                x, y = generateDataFast(
                    expr, 100, 2, 8, -10, 10, total_variabels=["x_1", "x_2"]
                )
                x = torch.tensor(x, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                point_set = (
                    torch.concat((x, y.unsqueeze(1)), dim=1).to(device).unsqueeze(0)
                )

                done_candidates = beam_search(model, env, point_set, expr)
                valid_candidates = []

                for candidate in done_candidates:

                    try:
                        # Set the signal handler for the SIGALRM signal
                        signal.alarm(20)

                        candidate_expr, _, mse, _, _ = bfgs(
                            candidate["skeleton"], x.unsqueeze(0), y
                        )

                        if np.isnan(mse) or np.isinf(mse):
                            continue

                        candidate["mse"] = mse
                        candidate["expression"] = candidate_expr

                        agent_expr = str(candidate_expr)

                        total_variables = ["x_1", "x_2"]
                        X_dict = {
                            x_: x[:, idx].cpu()
                            for idx, x_ in enumerate(total_variables)
                        }
                        y_pred = sp.lambdify(
                            ",".join(total_variables), sp.sympify(agent_expr)
                        )(**X_dict)

                        r2 = (
                            1
                            - torch.sum(torch.square(y - y_pred))
                            / torch.sum(torch.square(y - torch.mean(y)))
                        ).item()

                        if isinstance(r2, float) or isinstance(r2, int):
                            candidate["R2"] = r2
                        else:
                            continue

                        valid_candidates.append(candidate)

                        signal.alarm(0)

                    except Exception as e:
                        signal.alarm(0)
                        print(f"Exception encountered: {e}")

                best_candidate = max(valid_candidates, key=lambda x: x["R2"])

                log_cans = [
                    {
                        "mse": float(can["mse"]),
                        "expression": str(can["expression"]),
                        "r2": can["R2"],
                        "skeleton": can["skeleton"],
                    }
                    for can in valid_candidates
                ]

                record[eq_name] = {
                    "R2": best_candidate["R2"],
                    "equation": expr,
                    "agent_skeleton": best_candidate["skeleton"],
                    "agent_expression": str(best_candidate["expression"]),
                    "candidates": log_cans,
                }

                with open(
                    f"{folder_path}/beam_search_benchmark_{benchmark}.json", "w"
                ) as f:
                    json.dump(record, f, indent=5)

    if args.target == "ssdnc_r2":
        signal.signal(signal.SIGALRM, handle_timeout)

        SSDNC_dataset = load_dataset("SSDNC", cfg)
        num_tests = len(SSDNC_dataset)
        env = SymbolicWorldEnv(cfg, cal_r2=False)
        record = {}

        for eq_num in tqdm(range(num_tests)):
            signal.alarm(0)
            gc.collect()
            fix_seed(0)

            point_set, _, _, _, _, skeleton, seq, expr = SSDNC_dataset[eq_num]
            point_set = point_set.to(device).unsqueeze(0).transpose(1, 2)
            x = point_set[0, :, :2].cpu()
            y = point_set[0, :, -1].cpu()

            try:
                done_candidates = beam_search(model, env, point_set, expr)
            except Exception as e:
                record[eq_num] = {
                    "error": str(e),
                }
                with open(f"{folder_path}/beam_search_SSDNC_R2.json", "w") as f:
                    json.dump(record, f, indent=4)
                continue

            valid_candidates = []

            for candidate in done_candidates:
                if "PH" in candidate["skeleton"]:
                    continue
                try:
                    signal.alarm(120)

                    candidate_expr, _, mse, _, _ = bfgs(
                        candidate["skeleton"], x.unsqueeze(0), y
                    )

                    if np.isnan(mse) or np.isinf(mse):
                        continue

                    candidate["mse"] = mse
                    candidate["expression"] = candidate_expr

                    agent_expr = str(candidate_expr)

                    total_variables = ["x_1", "x_2"]
                    X_dict = {
                        x_: x[:, idx].cpu() for idx, x_ in enumerate(total_variables)
                    }
                    y_pred = sp.lambdify(
                        ",".join(total_variables), sp.sympify(agent_expr)
                    )(**X_dict)

                    r2 = (
                        1
                        - torch.sum(torch.square(y - y_pred))
                        / torch.sum(torch.square(y - torch.mean(y)))
                    ).item()

                    if isinstance(r2, float) or isinstance(r2, int):
                        candidate["R2"] = r2
                    else:
                        continue

                    valid_candidates.append(candidate)
                    signal.alarm(0)

                except Exception as e:
                    signal.alarm(0)
                    print(f"Exception encountered: {e}")
                    continue

            best_candidate = max(valid_candidates, key=lambda x: x["R2"])

            log_cans = [
                {
                    "mse": float(can["mse"]),
                    "expression": str(can["expression"]),
                    "r2": can["R2"],
                    "skeleton": can["skeleton"],
                }
                for can in valid_candidates
            ]

            record[eq_num] = {
                "R2": best_candidate["R2"],
                "equation": expr,
                "skeleton": best_candidate["skeleton"],
                "expression": str(best_candidate["expression"]),
                "candidates": log_cans,
            }

            with open(f"{folder_path}/beam_search_SSDNC_R2.json", "w") as f:
                json.dump(record, f, indent=4)
