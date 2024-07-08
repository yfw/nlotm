import math

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from utils import LearnedPositionalEmbedding1D


def get_prior_model(args):
    return {
        "dvae_tf": DvaeTransformerPrior,
        "discrete_block_tf": DiscreteBlockTransformerPrior,
    }[args.prior_type](args)

class DvaeTransformerPrior(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.prior_d_model
        self.num_heads = args.prior_num_heads
        self.dropout = args.prior_dropout
        self.num_layers = args.prior_num_decoder_layers
        self.vocab_size = args.vocab_size
        self.norm_first = args.prior_norm_first
        if hasattr(args, "sample_length"):
            self.sample_length = args.sample_length
        else:
            self.sample_length = 1
        self.z_length = self.sample_length * ((args.image_size // 4)**2)

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos = LearnedPositionalEmbedding1D(self.z_length, self.d_model)
        encoder_layers = TransformerEncoderLayer(
            self.d_model,
            self.num_heads,
            4 * self.d_model,
            self.dropout,
            batch_first=True,
            norm_first=self.norm_first
        )
        self.tf = TransformerEncoder(encoder_layers, self.num_layers)
        self.bos = nn.Parameter(torch.Tensor(1, 1, self.d_model))
        nn.init.xavier_uniform_(self.bos)
        self.head = nn.Linear(self.d_model, self.vocab_size, bias=False)

    def _prep_z(self, z):
        z = z * math.sqrt(self.d_model)
        z = self.pos(z)
        return z

    def forward(self, z):
        B, T = z.shape
        # B, T, D
        z = self.embedding(z)
        z = torch.cat([self.bos.expand(B, -1, -1), z], dim=1)[:, :-1]
        z = self._prep_z(z)
        mask = torch.triu(torch.ones(T, T, device=z.device) * float("-inf"), diagonal=1)
        z_pred_logits = self.head(self.tf(z, mask))
        return z_pred_logits

    def get_z_for_recon(self, z):
        z_pred = self(z)
        z_pred = torch.argmax(z_pred, dim=-1)
        return z_pred

    def loss(self, z):
        B, T = z.shape
        z_pred_logits = self(z)
        cross_entropy = nn.CrossEntropyLoss(reduction="none")(
            z_pred_logits.view(B * T, -1), z.view(B * T)
        )
        cross_entropy = cross_entropy.view(B, T).sum(dim=1).mean()
        return cross_entropy

    @torch.no_grad()
    def sample(self, B, temperature=1.0, top_k=None):
        z = self.bos.expand(B, 1, -1)
        z_gen = z.new_zeros(0)
        for _ in range(self.z_length):
            z_inp = self._prep_z(z)
            T = z_inp.shape[1]
            mask = torch.triu(torch.ones(T, T, device=z.device) * float("-inf"), diagonal=1)
            z_pred_logits = self.head(self.tf(z_inp, mask))
            z_next_logits = z_pred_logits[:, -1]
            z_next_logits = z_next_logits / temperature
            # from nanogpt
            if top_k is not None:
                v, _ = torch.topk(z_next_logits, min(top_k, z_next_logits.size(-1)))
                z_next_logits[z_next_logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(z_next_logits, dim=-1)
            z_next = torch.multinomial(probs, num_samples=1)
            z_gen = torch.cat([z_gen, z_next], dim=1)
            z = torch.cat([z, self.embedding(z_next)], dim=1)
        return z_gen


class DiscreteBlockTransformerPrior(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_slots = args.num_slots
        self.num_blocks = args.num_blocks
        self.num_prototypes = args.num_prototypes
        self.d_model = args.prior_d_model
        self.num_heads = args.prior_num_heads
        self.dropout = args.prior_dropout
        self.num_layers = args.prior_num_decoder_layers
        self.vocab_size = self.num_prototypes * self.num_blocks
        if hasattr(args, "sample_length"):
            self.sample_length = args.sample_length
        else:
            self.sample_length = 1
        self.z_length = self.num_slots * self.num_blocks * self.sample_length

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos = LearnedPositionalEmbedding1D(self.z_length, self.d_model, dropout=self.dropout)
        encoder_layers = TransformerEncoderLayer(
            self.d_model,
            self.num_heads,
            4 * self.d_model,
            self.dropout,
            batch_first=True,
        )
        self.tf = TransformerEncoder(encoder_layers, self.num_layers)
        self.bos = nn.Parameter(torch.Tensor(1, 1, self.d_model))
        nn.init.xavier_uniform_(self.bos)
        self.head = nn.Linear(self.d_model, self.vocab_size, bias=False)

    def _prep_z(self, z):
        z = z * math.sqrt(self.d_model)
        z = self.pos(z)
        return z

    def forward(self, z):
        z_orig = z
        B, T = z.shape
        # B, T, D
        z = self.embedding(z)
        z = torch.cat([self.bos.expand(B, -1, -1), z], dim=1)[:, :-1]
        z = self._prep_z(z)
        mask = torch.triu(torch.ones(T, T, device=z.device) * float("-inf"), diagonal=1)
        z_pred_logits = self.head(self.tf(z, mask))
        return z_pred_logits

    def get_z_for_recon(self, z):
        z_pred = self(z)
        z_pred = torch.argmax(z_pred, dim=-1)
        return z_pred

    def loss(self, z):
        B, T = z.shape
        z_pred_logits = self(z)
        cross_entropy = nn.CrossEntropyLoss(reduction="none")(
            z_pred_logits.view(B * T, -1), z.view(B * T)
        )
        cross_entropy = cross_entropy.view(B, T).sum(dim=1).mean()
        return cross_entropy

    @torch.no_grad()
    def sample(self, B=1, gen_timesteps=None, cond=None, temperature=1.0, top_k=None):
        if gen_timesteps is None:
            gen_steps = self.z_length
        else:
            gen_steps = gen_timesteps * self.num_slots * self.num_blocks

        if cond is not None:
            B = cond.shape[0]

        z = self.bos.expand(B, 1, -1)
        if cond is not None:
            cond = self.embedding(cond)
            cond = rearrange(cond, 'b t m d -> b (t m) d')
            z = torch.cat([z, cond], dim=1)
        z_gen = z.new_zeros(0)
        for idx in range(gen_steps):
            z_inp = self._prep_z(z)
            T = z_inp.shape[1]
            mask = torch.triu(torch.ones(T, T, device=z.device) * float("-inf"), diagonal=1)
            z_pred_logits = self.head(self.tf(z_inp, mask))
            z_next_logits = z_pred_logits[:, -1]
            z_next_logits = z_next_logits / temperature
            # from nanogpt
            if top_k is not None:
                v, _ = torch.topk(z_next_logits, min(top_k, z_next_logits.size(-1)))
                z_next_logits[z_next_logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(z_next_logits, dim=-1)
            z_next = torch.multinomial(probs, num_samples=1)
            z_gen = torch.cat([z_gen, z_next], dim=1)
            z = torch.cat([z, self.embedding(z_next)], dim=1)
        return z_gen
        
