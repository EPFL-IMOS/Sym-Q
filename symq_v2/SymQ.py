import torch
import torch.nn as nn

from util import get_seq2action
from encoder import SetEncoder, TreeEncoder, RNNEncoder


class SymQ(nn.Module):

    def __init__(self, cfg, device):
        super(SymQ, self).__init__()
        self.device = device

        if cfg["num_vars"] == 2 and cfg["SymQ"]["use_pretrain"]:
            cfg["SetEncoder"]["dim_input"] = 4
        else:
            cfg["SetEncoder"]["dim_input"] = cfg["num_vars"] + 1
        self.set_encoder = SetEncoder(cfg["SetEncoder"]).to(device)

        seq2action = get_seq2action(cfg)
        cfg["TreeEncoder"]["max_step"] = cfg["max_step"]
        cfg["TreeEncoder"]["num_actions"] = len(seq2action)
        if cfg["SymQ"]["use_transformer"]:
            self.tree_encoder = TreeEncoder(cfg["TreeEncoder"]).to(device)
        else:
            self.tree_encoder = RNNEncoder(cfg["TreeEncoder"]).to(device)

        self.cfg = cfg["SymQ"]
        self.cfg["num_actions"] = len(seq2action)

        if self.cfg["batch_norm"]:
            self.batch_norm1 = nn.BatchNorm1d(self.cfg["dim_hidden"])
            self.batch_norm2 = nn.BatchNorm1d(self.cfg["dim_hidden"])

        if self.cfg["embedding_fusion"] == "concat":
            dim_fusion = (
                cfg["SetEncoder"]["dim_output"] + cfg["TreeEncoder"]["dim_output"]
            )
        elif self.cfg["embedding_fusion"] == "add":
            assert cfg["SetEncoder"]["dim_output"] == cfg["TreeEncoder"]["dim_output"]
            dim_fusion = cfg["SetEncoder"]["dim_output"]
        else:
            raise KeyError(
                f"Unknown embedding fusion method: {self.cfg['embedding_fusion']}"
            )

        self.downsample = nn.Linear(
            int(cfg["SetEncoder"]["dim_hidden"] * cfg["SetEncoder"]["num_features"]),
            cfg["SetEncoder"]["dim_output"],
        )
        self.BN_downsample = nn.BatchNorm1d(cfg["SetEncoder"]["dim_output"])
        self.act_downsample = nn.GELU()
        self.linear1 = nn.Linear(dim_fusion, self.cfg["dim_hidden"])
        self.relu1 = nn.GELU()

        self.linear2 = nn.Linear(self.cfg["dim_hidden"], self.cfg["dim_hidden"])
        self.relu2 = nn.GELU()

        self.linear3 = nn.Linear(self.cfg["dim_hidden"], self.cfg["dim_hidden"])
        self.relu3 = nn.GELU()

        self.linear4 = nn.Linear(self.cfg["dim_hidden"], self.cfg["num_actions"])

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the weights of the module.
        Uses Xavier Uniform initialization for linear layers and sets bias to zero.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def _encode_set(self, x):
        x = x.permute(0, 2, 1)
        x = self.set_encoder(x)

        return x.view(x.size(0), -1)

    def _encode_tree(self, x):
        return self.tree_encoder(x)

    def forward(self, point_set, tree):
        # point_set: [batch_size, num_var, num_point]
        # tree: [batch_size, seq_len, num_action]
        set_embedding = self._encode_set(point_set)
        set_embedding = self.act_downsample(
            self.BN_downsample(self.downsample(set_embedding))
        )

        tree_embedding = self._encode_tree(tree)

        if self.cfg["embedding_fusion"] == "concat":
            embeddings = torch.cat((set_embedding, tree_embedding), 1)
        elif self.cfg["embedding_fusion"] == "add":
            embeddings = set_embedding + tree_embedding

        x = self.linear1(embeddings)
        if self.cfg["batch_norm"]:
            x = self.batch_norm1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        if self.cfg["batch_norm"]:
            x = self.batch_norm2(x)
        x = self.relu2(x)

        x = self.relu3(self.linear3(x))

        if self.cfg["set_skip_connection"]:
            assert x.shape == set_embedding.shape
            x = x + set_embedding

        x = self.linear4(x)

        return x, set_embedding

    def act(self, point_set, tree, mask=None):
        with torch.no_grad():
            q_values, _ = self.forward(point_set, tree)

        if mask is not None:
            q_values = torch.where(mask == 1, q_values, -1e9)

        return q_values.argmax(dim=-1), q_values


if __name__ == "__main__":
    from util import load_cfg
    from dataset import HDF5Dataset
    from torch.utils.data import DataLoader

    cfg = load_cfg("cfg.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SymQ(cfg, device).to(device)
    print(model)

    folder_path = f"{cfg.Dataset.dataset_folder}/{cfg.num_vars}_var/train"
    dataset = HDF5Dataset(folder_path, cfg)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    batch = next(iter(dataloader))

    model_output = model(batch[0].to(device), batch[1].to(device))
    print(f"Logits: {model_output[0].shape}")
    print(f"Set Embedding: {model_output[1].shape}")
