import os
import time
import shutil
import pickle
from datetime import datetime


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from tqdm import tqdm

from SymQ import SymQ
from loss import SupConLoss
from dataset import HDF5Dataset
from util import fix_seed, load_cfg, comp_wn


def setup(rank, world_size):
    # IP address of the machine that will host the process with rank 0.
    os.environ["MASTER_ADDR"] = "localhost"
    # A free port on the machine that will host the process with rank 0
    os.environ["MASTER_PORT"] = "62777"

    # initialize the process group
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    except RuntimeError:
        print(
            "[!] Distributed package doesn't have NCCL built in ... initializing with GLOO (sub-optimal)"
        )
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run(
    cfg,
    model,
    dataloader,
    optimizer,
    epoch,
    rank,
    best_loss,
    perf_track=None,
    mode="train",
    scaler=None,
    use_cuda=True,
):
    total_correct = 0
    total_samples = 0

    total_q_loss = 0
    total_cl_loss = 0
    total_loss = 0

    if use_cuda:
        gpu_id = cfg.training.visible_devices[rank]
    else:
        gpu_id = "cpu"
    device = torch.device(gpu_id)

    log_perf = rank == 0 and perf_track is not None and mode == "train"

    if cfg.training.use_mse:
        QL = nn.MSELoss()
    else:
        QL = nn.CrossEntropyLoss()

    CL = SupConLoss(device=gpu_id)

    if mode == "train":
        model.train()
    else:
        model.eval()

    if log_perf:
        opt_time = time.time()

    if rank == 0:
        yielder = tqdm(dataloader)
    else:
        yielder = dataloader

    for batch in yielder:
        # Move data to device
        batch_on_device = []
        for data in batch:
            batch_on_device.append(data.to(gpu_id, non_blocking=True))

        (
            points,
            prefix,
            eq_id,
            next_token,
            q_values,
        ) = batch_on_device

        if log_perf:
            data_time = time.time()
            perf_track["data_gen"].append(data_time - opt_time)

        # Forward pass
        with torch.set_grad_enabled(mode == "train"):
            with autocast():
                logit, point_embedding = model(points, prefix)

                normalized_point_embedding = F.normalize(point_embedding, p=2, dim=-1)

                CL_embedding = torch.stack(
                    (
                        normalized_point_embedding,
                        normalized_point_embedding,
                    ),
                    dim=1,
                )

                if log_perf:
                    fw_time = time.time()
                    perf_track["fw_time"].append(fw_time - data_time)

                act = torch.argmax(logit, dim=1)

                correct = (act == next_token).sum().item()

                total_correct += correct
                total_samples += next_token.size(0)

                # Compute loss
                if cfg.training.use_mse:
                    q_loss = QL(F.sigmoid(logit), q_values)
                else:
                    q_loss = QL(logit.view(-1, logit.shape[-1]), next_token.view(-1))

                cl_loss = CL(CL_embedding, eq_id)

                loss = cfg.loss.ql * q_loss + cfg.loss.cl * cl_loss

                if mode == "train":

                    if log_perf:
                        loss_time = time.time()
                        perf_track["loss_time"].append(loss_time - fw_time)

                    # Backward pass and optimization
                    optimizer.zero_grad()

                    if scaler:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                    torch.cuda.empty_cache()

                    if log_perf:
                        opt_time = time.time()
                        perf_track["opt_time"].append(opt_time - loss_time)
                        perf_track["update_time"].append(opt_time - data_time)
                        perf_track["R0_memory"].append(
                            torch.cuda.memory_allocated(device)
                        )

                # Documentation
                total_q_loss += q_loss.item()
                total_cl_loss += cl_loss.item()
                total_loss += loss.item()

    loss_dict = {}

    # Reduce loss across all processes
    total_q_loss_tensor = torch.tensor(total_q_loss).to(gpu_id) / len(dataloader)
    total_cl_loss_tensor = torch.tensor(total_cl_loss).to(gpu_id) / len(dataloader)
    total_loss_tensor = torch.tensor(total_loss).to(gpu_id) / len(dataloader)

    dist.all_reduce(total_q_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_cl_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)

    # Reduce accuracy across all processes
    correct_predictions_tensor = torch.tensor(total_correct).to(gpu_id)
    total_predictions_tensor = torch.tensor(total_samples).to(gpu_id)

    dist.all_reduce(correct_predictions_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_predictions_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        accuracy = correct_predictions_tensor.item() / total_predictions_tensor.item()
        # Statistics
        world_size = dist.get_world_size()
        total_q_loss = total_q_loss_tensor.item() / world_size
        total_cl_loss = total_cl_loss_tensor.item() / world_size
        total_loss = total_loss_tensor.item() / world_size

        if total_loss < best_loss and mode == "eval":
            model_save_name = f"{cfg.logging.log_dir}/{mode}_best.pth"
            torch.save(model.state_dict(), model_save_name)
            best_loss = total_loss

        if mode == "train":
            loss_dict = {
                "train_loss": total_loss,
                "train_q_loss": total_q_loss,
                "train_cl_loss": total_cl_loss,
                "train_step_acc": accuracy,
                "train_weights_norm": comp_wn(model),
            }
        if mode == "eval":
            loss_dict = {
                "val_loss": total_loss,
                "val_q_loss": total_q_loss,
                "val_cl_loss": total_cl_loss,
                "val_step_acc": accuracy,
            }

    if (
        rank == 0
        and cfg.logging.log_dir is not None
        and cfg.logging.save_model
        and mode == "train"
    ):
        if (epoch + 1) % 10 == 0:
            model_save_name = f"{cfg.logging.log_dir}/epoch_{epoch+1}.pth"
            # Checkpoint
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            }
            torch.save(state, model_save_name)

    if log_perf:
        with open(cfg.logging.log_dir + "/perf_track.pkl", "wb") as f:
            pickle.dump(perf_track, f)

    dist.barrier()

    return loss_dict, best_loss


def launch(rank, world_size, cfg, use_cuda):

    # --------------- Setup -------------#
    print(f"[ Rank {rank}]: Setting up ...")
    setup(rank, world_size)
    # -----------------------------------#

    fix_seed(cfg.training.seed)

    # ---------------- Data ------------#

    # Training data
    train_dataset = HDF5Dataset(
        f"{cfg.Dataset.dataset_folder}/{cfg.num_vars}_var/train", cfg
    )
    if rank == 0:
        print(f"[ > ] Training samples: {len(train_dataset)}")

    val_dataset = HDF5Dataset(
        f"{cfg.Dataset.dataset_folder}/{cfg.num_vars}_var/val", cfg
    )
    if rank == 0:
        print(f"[ > ] Validation samples: {len(val_dataset)}")

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,  # Warning: on some machines this can lead to bugs
        sampler=train_sampler,
        # https://github.com/pytorch/pytorch/issues/82077
        persistent_workers=cfg.training.num_workers > 0,
    )

    # Validation data
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,  # Warning: on some machines this can lead to bugs
        sampler=val_sampler,
        # https://github.com/pytorch/pytorch/issues/82077
        persistent_workers=cfg.training.num_workers > 0,
    )

    # ------------Model & Opt-------------#

    # Initialize model and optimizer
    if use_cuda:
        gpu_id = cfg.training.visible_devices[rank]
    else:
        gpu_id = "cpu"
    model = SymQ(cfg, gpu_id).to(gpu_id)
    print(f"[ Rank {rank} ]: Model running on device {gpu_id}")

    if cfg["SymQ"]["use_pretrain"]:
        device = torch.device(gpu_id)
        pretrain_weights = torch.load(cfg["SymQ"]["pretrain_path"], map_location=device)["state_dict"]
        encoder_weights = {
            k[4:]: v for k, v in pretrain_weights.items() if k.startswith("enc.")
        }
        
        model.set_encoder.load_state_dict(encoder_weights, strict=True)

    if cfg["SymQ"]["freeze_encoder"]:
        for param in model.set_encoder.parameters():
            param.requires_grad = False

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    # # Load resume path
    if cfg.training.resume_path:
        checkpoint = torch.load(
            cfg.training.resume_path, map_location=torch.device(gpu_id)
        )
        new_state_dict = {
            k.replace("module.", ""): v
            for k, v in checkpoint["model_state_dict"].items()
        }
        model.load_state_dict(
            new_state_dict,
            strict=True,
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[ Rank {rank} ]: Model loaded from {cfg.training.resume_path}")

    if use_cuda:
        model = DDP(model, device_ids=[gpu_id], find_unused_parameters=False)

    # --------------Logging--------------#

    perf_track = None
    if rank == 0:
        perf_track = {
            "cfg": {
                "device": gpu_id,
                "updates": len(train_loader),
                "num_weights": sum(p.numel() for p in model.parameters()),
            },
            "fw_time": [],
            "loss_time": [],
            "opt_time": [],
            "eval_time": [],
            "update_time": [],
            "data_gen": [],
            "R0_memory": [],
        }

        with open(cfg.logging.log_dir + "/perf_track.pkl", "wb") as f:
            pickle.dump(perf_track, f)

        # Initialize wandb
        if cfg.logging.wandb:
            wandb.init(project="SymQ", config=cfg)

    # ---------Training Loop-------------#
    best_train_loss = np.inf
    best_val_loss = np.inf

    if cfg.training.use_scaler:
        scaler = GradScaler()
    else:
        scaler = None

    for epoch in range(cfg.training.epochs):

        if rank == 0:
            start_time = time.time()

        train_sampler.set_epoch(epoch)
        train_loss_dict, best_train_loss = run(
            cfg,
            model,
            train_loader,
            optimizer,
            epoch,
            rank,
            best_train_loss,
            perf_track=perf_track,
            scaler=scaler,
            use_cuda=use_cuda,
        )

        val_sampler.set_epoch(epoch)
        val_loss_dict, best_val_loss = run(
            cfg,
            model,
            val_loader,
            optimizer,
            epoch,
            rank,
            best_val_loss,
            perf_track=None,
            mode="eval",
            use_cuda=use_cuda,
        )
        loss_log = {**train_loss_dict, **val_loss_dict}

        if rank == 0:
            print(
                f"\n[ > ] Epoch {epoch + 1}, train: {loss_log['train_loss']}, val: {loss_log['val_loss']}, time:{time.time() - start_time:.4f} s\n"
            )
            if cfg.logging.wandb:
                wandb.log(loss_log)

    if rank == 0:
        print("\n\n[!] Cleaning Threads ...")

    cleanup()

    if rank == 0 and cfg.logging.wandb:
        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_var", type=int, default=3)
    args = parser.parse_args()

    if args.n_var not in [2, 3]:
        print(f"Number of variables must be 2 or 3, got {args.n_var}")
        exit()

    # Create a folder to save the weights
    start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = f"model/{start_time}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"This training will be saved in {log_dir}")

    # Copy related files to the folder
    for file in [
        f"cfg_{args.n_var}var.yaml",
        "SymQ.py",
        "train.py",
        "encoder.py",
    ]:
        shutil.copyfile(file, f"{log_dir}/{file}")

    # Load config file
    cfg = load_cfg(f"cfg_{args.n_var}var.yaml")
    cfg.logging.log_dir = log_dir

    # Make the training reproducible
    fix_seed(cfg.training.seed)

    # Check if CUDA is available
    use_cuda = not cfg.training.no_cuda
    world_size = cfg.training.gpus

    if world_size == "all":
        world_size = torch.cuda.device_count()

    if use_cuda and torch.cuda.device_count() > 1:
        print(
            f"[!] Running on {world_size} GPUs [locally avilable GPUs: {torch.cuda.device_count()}]"
        )
    else:
        print(
            f"[!] Running in CPU mode [locally avilable GPUs: {torch.cuda.device_count()}]"
        )
    print("\n" + "-" * 51 + "\n\n")

    # Spawn training threads
    try:
        mp.spawn(launch, args=(world_size, cfg, use_cuda), nprocs=world_size, join=True)
    except Exception as e:
        print(f"[!] Error: {e}")
        cleanup()
        if cfg.logging.wandb:
            wandb.finish()
    except KeyboardInterrupt:
        cleanup()
        if cfg.logging.wandb:
            wandb.finish()
