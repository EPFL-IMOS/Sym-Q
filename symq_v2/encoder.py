import math
import torch
import torch.nn as nn
import pytorch_lightning as pl


from attention_block import ISAB, PMA


class SetEncoder(pl.LightningModule):
    def __init__(self, cfg):
        super(SetEncoder, self).__init__()
        self.linear = cfg.linear
        self.bit16 = cfg.bit16
        self.norm = cfg.norm
        assert (
            cfg.linear != cfg.bit16
        ), "one and only one between linear and bit16 must be true at the same time"
        if cfg.norm:
            self.register_buffer("mean", torch.tensor(cfg.mean))
            self.register_buffer("std", torch.tensor(cfg.std))

        self.activation = cfg.activation
        self.input_normalization = cfg.input_normalization
        if cfg.linear:
            self.linearl = nn.Linear(cfg.dim_input, 16 * cfg.dim_input)
        self.selfatt = nn.ModuleList()
        # dim_input = 16*dim_input
        self.selfatt1 = ISAB(
            16 * cfg.dim_input, cfg.dim_hidden, cfg.num_heads, cfg.num_inds, ln=cfg.ln
        )
        for i in range(cfg.n_l_enc):
            self.selfatt.append(
                ISAB(
                    cfg.dim_hidden,
                    cfg.dim_hidden,
                    cfg.num_heads,
                    cfg.num_inds,
                    ln=cfg.ln,
                )
            )
        self.outatt = PMA(cfg.dim_hidden, cfg.num_heads, cfg.num_features, ln=cfg.ln)

    def float2bit(
        self, f, num_e_bits=5, num_m_bits=10, bias=127.0, dtype=torch.float32
    ):
        ## SIGN BIT
        s = (
            torch.sign(f + 0.001) * -1 + 1
        ) * 0.5  # Swap plus and minus => 0 is plus and 1 is minus
        s = s.unsqueeze(-1)
        f1 = torch.abs(f)
        ## EXPONENT BIT
        e_scientific = torch.floor(torch.log2(f1))
        e_scientific[e_scientific == float("-inf")] = -(2 ** (num_e_bits - 1) - 1)
        e_decimal = e_scientific + (2 ** (num_e_bits - 1) - 1)
        e = self.integer2bit(e_decimal, num_bits=num_e_bits)
        ## MANTISSA
        f2 = f1 / 2 ** (e_scientific)
        m2 = self.remainder2bit(f2 % 1, num_bits=bias)
        fin_m = m2[:, :, :, :num_m_bits]  # [:,:,:,8:num_m_bits+8]
        return torch.cat([s, e, fin_m], dim=-1).type(dtype)

    def remainder2bit(self, remainder, num_bits=127):
        dtype = remainder.type()
        exponent_bits = torch.arange(num_bits, device=self.device).type(dtype)
        exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
        out = (remainder.unsqueeze(-1) * 2**exponent_bits) % 1
        return torch.floor(2 * out)

    def integer2bit(self, integer, num_bits=8):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1, device=self.device).type(
            dtype
        )
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) / 2**exponent_bits
        return (out - (out % 1)) % 2

    def forward(self, x):
        if self.bit16:
            x = self.float2bit(x)
            x = x.view(x.shape[0], x.shape[1], -1)
            if self.norm:
                x = (x - 0.5) * 2
        if self.input_normalization:
            means = x[:, :, -1].mean(axis=1).reshape(-1, 1)
            std = x[:, :, -1].std(axis=1).reshape(-1, 1)
            std[std == 0] = 1
            x[:, :, -1] = (x[:, :, -1] - means) / std

        if self.linear:
            if self.activation == "relu":
                x = torch.relu(self.linearl(x))
            elif self.activation == "sine":
                x = torch.sin(self.linearl(x))
            else:
                x = self.linearl(x)
        x = self.selfatt1(x)
        for layer in self.selfatt:
            x = layer(x)
        x = self.outatt(x)
        return x


class TreeEncoder(nn.Module):

    def __init__(self, cfg):
        super(TreeEncoder, self).__init__()

        self.max_step = cfg.max_step
        self.num_actions = cfg.num_actions
        self.embed_dim = cfg["dim_hidden"]
        self.embedding_layer = nn.Linear(self.num_actions, cfg["dim_hidden"])

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg["dim_hidden"], nhead=cfg["num_heads"], batch_first=True
            ),
            num_layers=cfg["num_layers"],
        )

        pe = self.create_positional_encodings(self.max_step, self.embed_dim)
        self.register_buffer("pe", pe)

        self.output_layer = nn.Linear(cfg["dim_hidden"], cfg["dim_output"])

    def create_positional_encodings(self, max_length, embed_dim):
        """
        Create positional encodings.

        Args:
            max_length (int): Maximum length of the sequential representation.
            embed_dim (int): Embedding dimension.

        Returns:
            torch.Tensor: Tensor containing positional encodings of shape (max_length, embed_dim).
        """
        pe = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        """
        Forward pass for TreeStructureEncoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, max_step, input_dim).

        Returns:
            torch.Tensor: Encoded output tensor after processing through the TransformerEncoder and
                          a final linear layer.
        """
        embeddings = self.embedding_layer(x)

        embeddings = self.pe + embeddings

        transformer_output = self.transformer_encoder(embeddings)

        # Average over all sequence positions to get a fixed-sized representation
        sequence_representation = transformer_output.mean(dim=1)

        output = self.output_layer(sequence_representation)

        return output


class RNNEncoder(nn.Module):

    def __init__(self, cfg):
        super(RNNEncoder, self).__init__()
        self.num_actions = cfg.num_actions
        self.embed_dim = cfg["dim_hidden"]

        # Embedding layer to map input to the hidden dimension
        self.embedding_layer = nn.Linear(self.num_actions, cfg["dim_hidden"])

        # RNN layer (can be LSTM, GRU, or vanilla RNN, here we use GRU)
        self.rnn = nn.GRU(
            input_size=cfg["dim_hidden"],  # Embedding dimension
            hidden_size=cfg["dim_hidden"],  # Hidden state dimension
            num_layers=cfg["num_layers"],  # Number of RNN layers
            batch_first=True,  # (batch_size, seq_len, input_size)
            bidirectional=False,
        )

        # Output layer to map the final hidden state to the output
        self.output_layer = nn.Linear(cfg["dim_hidden"], cfg["dim_output"])

    def forward(self, x):
        """
        Forward pass for RNNEncoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, max_step, input_dim).

        Returns:
            torch.Tensor: Encoded output tensor after processing through the RNN and
                          a final linear layer.
        """
        # Apply embedding layer
        embeddings = self.embedding_layer(x)  # (batch_size, seq_len, embed_dim)

        # Pass through the RNN
        rnn_output, _ = self.rnn(
            embeddings
        )  # rnn_output: (batch_size, seq_len, hidden_size)

        # Take the final hidden state (you can also average across time steps if needed)
        final_hidden_state = rnn_output[:, -1, :]  # (batch_size, hidden_size)

        # Pass the final hidden state through the output layer
        output = self.output_layer(final_hidden_state)

        return output
