import os
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from util import seq_to_tree, get_seq2action


def get_h5_files_in_folder(folder_path):
    h5_files = []
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".h5"):
                h5_files.append(os.path.join(root, file))
    return sorted(h5_files)


class HDF5Dataset(Dataset):
    def __init__(self, folder_path, cfg):
        self.cfg = cfg
        self.seq2action = get_seq2action(cfg)

        if "train" in folder_path:
            self.hdf5_files = get_h5_files_in_folder(folder_path)
        elif "val" in folder_path:
            self.hdf5_files = get_h5_files_in_folder(folder_path)

        self.datasets = []
        self.indices = []
        self.cumulative_sizes = []

        for hdf5_file in tqdm(self.hdf5_files):
            with h5py.File(hdf5_file, "r") as hf:
                length = len(hf["action"])
                self.indices.append((hdf5_file, length))
                self.cumulative_sizes.append(
                    self.cumulative_sizes[-1] + length
                    if self.cumulative_sizes
                    else length
                )

    def __len__(self):
        """Return the total number of prefix samples"""
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        file_idx = next(i for i, v in enumerate(self.cumulative_sizes) if v > idx)
        if file_idx > 0:
            idx -= self.cumulative_sizes[file_idx - 1]

        hdf5_file = self.hdf5_files[file_idx]

        # Open the HDF5 file lazily (each time a sample is requested)
        with h5py.File(hdf5_file, "r") as hf:
            points_idx = hf["map"][idx]
            points = torch.tensor(hf["points"][points_idx], dtype=torch.float32)

            # Padding for x_3 if 2 var version using pretrain
            if self.cfg["num_vars"] == 2 and self.cfg["SymQ"]["use_pretrain"]:
                zeros = torch.zeros(1, points.shape[-1], dtype=torch.float32)
                points = torch.cat((points[:-1, :], zeros, points[-1:, :]), dim=0)

            prefix = hf["prefix"][idx]
            prefix = [int(p) for p in prefix if not np.isnan(p)]
            prefix = seq_to_tree(prefix, self.cfg)

            eq_id = torch.tensor(hf["eq_id"][idx], dtype=torch.long)
            action = torch.tensor(hf["action"][idx], dtype=torch.long)
            q_values = torch.ones(len(self.seq2action), dtype=torch.float32) * 0.1
            q_values[action] = 0.9

        return points, prefix, eq_id, action, q_values


if __name__ == "__main__":
    from util import load_cfg
    from torch.utils.data import DataLoader

    cfg = load_cfg("cfg_2var.yaml")

    folder_path = f"{cfg.Dataset.dataset_folder}/{cfg.num_vars}_var/val"
    dataset = HDF5Dataset(folder_path, cfg)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    batch = next(iter(dataloader))
    print(f"Points: {batch[0].shape}")
    print(f"Prefix: {batch[1].shape}")
    print(f"Eq ID: {batch[2].shape}")
    print(f"Next Action: {batch[3].shape}")
    print(f"Q Values: {batch[4].shape}")
