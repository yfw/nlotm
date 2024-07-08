from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dvae import dVAE
from transformer import TransformerEncoder, TransformerDecoder
from utils import (
    linear,
    BlockLayerNorm,
    BlockAttention,
    BlockGRU,
    BlockLinear,
    CartesianPositionalEmbedding,
    Conv2dBlock,
    conv2d,
    OneHotDictionary,
    LearnedPositionalEmbedding1D,
    gumbel_softmax,
)

class VqEmaDcrBlockPrototypeMemory(nn.Module):
    # with dead code revival
    def __init__(self, num_prototypes, num_blocks, d_model):
        super().__init__()

        self.num_prototypes = num_prototypes
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.d_block = self.d_model // self.num_blocks
        self.dead_code_threshold = 1.0

        # block prototype memory
        # M*P, D
        self.mem = nn.Embedding(self.num_blocks * self.num_prototypes, self.d_block)

        # ema
        self.register_buffer(
            "_ema_cluster_size", torch.zeros(self.num_blocks * self.num_prototypes)
        )
        self._ema_w = nn.Parameter(
            torch.Tensor(self.num_blocks * self.num_prototypes, self.d_block)
        )
        self._ema_w.data.normal_()
        self._decay = 0.99
        self._epsilon = 1e-5

    def get_mem(self):
        return self.mem.weight

    def forward(self, queries):
        """
        queries: B, N, d_model
        return: B, N, d_model
        """

        B, N, _ = queries.shape

        # B*N*M, D
        flat_queries = queries.view(B * N * self.num_blocks, -1)

        # B*N*M, M*P
        distances = (flat_queries[:, None] - self.mem.weight[None]).pow(2).sum(dim=-1)

        # M, B*N, M, P
        distances = rearrange(
            distances,
            "(b n m1) (m2 p) -> m1 (b n) m2 p",
            b=B,
            n=N,
            p=self.num_prototypes,
        )

        indices = []
        for i in range(self.num_blocks):
            # B*N
            max_indices = (
                # B*N, P
                -distances[i, :, i]
            ).max(dim=1)[1]

            # shift indices since they are stored in one big nn.Embedding
            max_indices = max_indices + (i * self.num_prototypes)
            indices.append(max_indices)

        # M, B*N
        indices = torch.stack(indices)
        # B, N, M
        indices = rearrange(indices, "m (b n) -> b n m", b=B, n=N)

        # B, N, M*D
        embeddings = rearrange(self.mem.weight[indices], "b n m d -> b n (m d)")

        if self.training:
            encodings = torch.zeros(
                B * N * self.num_blocks,
                self.num_blocks * self.num_prototypes,
                device=queries.device,
            )
            encodings.scatter_(1, indices.reshape(-1, 1), 1)
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self.num_blocks * self.num_prototypes * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_queries)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            # dead code revivial if _ema_cluster_size less than self.dead_code_threshold
            # simmilar to Jukebox from OpenAI, except we need to do it at the block level

            # Create candidates for new codes from the queries
            candidates = rearrange(queries, 'b n (m d) -> (b n) m d', m=self.num_blocks)
            # self.mem.weight -> (self.num_blocks * self.num_prototypes), D
            # repeat so we have more candidates than the codebook size (self.num_prototypes)
            BN, _, D = candidates.shape
            n_repeats = (BN + self.num_prototypes - 1) // BN
            std = 0.01 / np.sqrt(D)
            candidates = candidates.repeat(n_repeats, 1, 1)
            candidates = candidates + torch.randn_like(candidates) * std
            candidates = candidates[torch.randperm(candidates.shape[0])][:self.num_prototypes]
            candidates = rearrange(candidates, 'p m d -> (m p) d')
            alive = (self._ema_cluster_size[..., None] >= self.dead_code_threshold).float()
            new_weight = (alive * (self._ema_w / self._ema_cluster_size.unsqueeze(1))) + ((1 - alive) * candidates)

            self.mem.weight = nn.Parameter(
                new_weight
            )

        # straight through estimator
        embeddings = (embeddings - queries).detach() + queries

        vq_loss = torch.tensor(0.0, requires_grad=True, device=queries.device)
        commitment_loss = (embeddings.detach() - queries).pow(2).mean()

        return embeddings, indices, vq_loss, commitment_loss

class BlockPrototypeMemory(nn.Module):
    def __init__(self, num_prototypes, num_blocks, d_model):
        super().__init__()

        self.num_prototypes = num_prototypes
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.d_block = self.d_model // self.num_blocks

        # block prototype memory
        self.mem_params = nn.Parameter(
            torch.zeros(1, num_prototypes, num_blocks, self.d_block), requires_grad=True
        )
        nn.init.trunc_normal_(self.mem_params)
        self.mem_proj = nn.Sequential(
            linear(self.d_block, 4 * self.d_block),
            nn.ReLU(),
            linear(4 * self.d_block, 4 * self.d_block),
            nn.ReLU(),
            linear(4 * self.d_block, 4 * self.d_block),
            nn.ReLU(),
            linear(4 * self.d_block, self.d_block),
        )

        # norms
        self.norm_mem = BlockLayerNorm(d_model, num_blocks)
        self.norm_query = BlockLayerNorm(d_model, num_blocks)

        # block attention
        self.attn = BlockAttention(d_model, num_blocks)

    def get_mem(self):
        # get memories
        mem = self.mem_proj(self.mem_params)  # 1, num_prototypes, num_blocks, d_block
        mem = mem.reshape(1, self.num_prototypes, -1)  # 1, num_prototypes, d_model

        # norms
        mem = self.norm_mem(mem)  # 1, num_prototypes, d_model

        mem = rearrange(mem, "1 p (m d) -> (m p) d", m=self.num_blocks)
        return mem

    def forward(self, queries):
        """
        queries: B, N, d_model
        return: B, N, d_model
        """

        B, N, _ = queries.shape

        # get memories
        mem = self.mem_proj(self.mem_params)  # 1, num_prototypes, num_blocks, d_block
        mem = mem.reshape(1, self.num_prototypes, -1)  # 1, num_prototypes, d_model

        # norms
        mem = self.norm_mem(mem)  # 1, num_prototypes, d_model
        queries = self.norm_query(queries)  # B, N, d_model

        # broadcast
        mem = mem.expand(B, -1, -1)  # B, num_prototypes, d_model


        output = self.attn(queries, mem, mem)  # B, N, d_model
        indices, vq_loss, commitment_loss = (
            None,
            torch.tensor(0.0, requires_grad=True, device=queries.device),
            torch.tensor(0.0, requires_grad=True, device=queries.device),
        )
        return output, indices, vq_loss, commitment_loss


class Nlotm(nn.Module):
    def __init__(
        self,
        num_iterations,
        num_slots,
        input_size,
        slot_size,
        mlp_hidden_size,
        num_prototypes,
        num_blocks,
        vq_type,
        slot_init_type,
        epsilon=1e-8,
    ):
        super().__init__()

        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.num_blocks = num_blocks
        self.vq_type = vq_type
        self.slot_init_type = slot_init_type
        self.epsilon = epsilon

        # parameters for Gaussian initialization (shared by all slots).
        self.slot_mu = nn.Parameter(torch.Tensor(1, 1, slot_size))
        self.slot_log_sigma = nn.Parameter(torch.Tensor(1, 1, slot_size))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

        if self.slot_init_type == 'learned':
            self.learned_init_slots = nn.Embedding(self.num_slots, self.slot_size)
            self.learned_init_sigma = 1.0


        # norms
        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = BlockLayerNorm(slot_size, num_blocks)

        # linear maps for the attention module.
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)

        # slot update functions.
        self.gru = BlockGRU(slot_size, slot_size, num_blocks)
        self.mlp = nn.Sequential(
            BlockLinear(slot_size, mlp_hidden_size, num_blocks),
            nn.ReLU(),
            BlockLinear(mlp_hidden_size, slot_size, num_blocks),
        )
        
        self.prototype_memory = {
            "none": BlockPrototypeMemory,
            "vq_ema_dcr": VqEmaDcrBlockPrototypeMemory,
        }[self.vq_type](num_prototypes, num_blocks, slot_size)

    def _step(self, slots, k, v):
        slots_prev = slots
        slots = self.norm_slots(slots)

        # Attention.
        q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
        attn_logits = torch.bmm(k, q.transpose(-1, -2))
        attn_vis = F.softmax(attn_logits, dim=-1)
        # `attn_vis` has shape: [batch_size, num_inputs, num_slots].

        # Weighted mean.
        attn = attn_vis + self.epsilon
        attn = attn / torch.sum(attn, dim=-2, keepdim=True)
        updates = torch.bmm(attn.transpose(-1, -2), v)
        # `updates` has shape: [batch_size, num_slots, slot_size].

        # Slot update.
        slots = self.gru(
            updates.view(-1, self.slot_size), slots_prev.view(-1, self.slot_size)
        )
        slots = slots.view(-1, self.num_slots, self.slot_size)
        slots = slots + self.mlp(self.norm_mlp(slots))
        slots, indices, vq_loss, commitment_loss = self.prototype_memory(
            slots
        )
        return slots, attn_vis, indices, vq_loss, commitment_loss

    def forward(self, inputs):
        B, num_inputs, input_size = inputs.size()

        # initialize slots
        if self.slot_init_type == 'learned':
            slots = self.learned_init_slots.weight.unsqueeze(0).repeat(B, 1, 1)
            slots = slots + (torch.randn_like(slots) * self.learned_init_sigma)
        elif self.slot_init_type == 'random':
            slots = inputs.new_empty(B, self.num_slots, self.slot_size).normal_()
            slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots
        else:
            raise NotImplementedError()

        # setup key and value
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        k = (self.slot_size ** (-0.5)) * k

        for _ in range(self.num_iterations):
            slots, attn_vis, indices, vq_loss, commitment_loss = self._step(slots, k, v)

        return slots, attn_vis, indices, vq_loss, commitment_loss

    
class ImageEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.cnn = nn.Sequential(
            Conv2dBlock(
                args.image_channels,
                args.cnn_hidden_size,
                5,
                1 if args.image_size == 64 else 2,
                2,
            ),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1 if args.image_size in (64, 128) else 2, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
            conv2d(args.cnn_hidden_size, args.d_model, 5, 1, 2),
        )

        image_size = 64

        self.pos = CartesianPositionalEmbedding(
            args.d_model,
            image_size,
        )

        self.layer_norm = nn.LayerNorm(args.d_model)

        self.mlp = nn.Sequential(
            linear(args.d_model, args.d_model, weight_init="kaiming"),
            nn.ReLU(),
            linear(args.d_model, args.d_model),
        )

        self.nlotm = Nlotm(
            args.num_iterations,
            args.num_slots,
            args.d_model,
            args.slot_size,
            args.mlp_hidden_size,
            args.num_prototypes,
            args.num_blocks,
            args.vq_type,
            args.slot_init_type,
        )


class ImageDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.slot_proj = BlockLinear(
            args.slot_size, args.d_model * args.num_blocks, args.num_blocks
        )

        self.block_pos = nn.Parameter(
            torch.zeros(1, 1, args.d_model * args.num_blocks), requires_grad=True
        )
        self.block_pos_proj = nn.Sequential(
            BlockLinear(
                args.d_model * args.num_blocks,
                args.d_model * args.num_blocks,
                args.num_blocks,
            ),
            nn.ReLU(),
            BlockLinear(
                args.d_model * args.num_blocks,
                args.d_model * args.num_blocks,
                args.num_blocks,
            ),
        )

        self.block_coupler = TransformerEncoder(num_blocks=1, d_model=args.d_model, num_heads=4)

        self.dict = OneHotDictionary(args.vocab_size, args.d_model)

        self.bos = nn.Parameter(torch.Tensor(1, 1, args.d_model))
        nn.init.xavier_uniform_(self.bos)

        self.decoder_pos = LearnedPositionalEmbedding1D(
            1 + (args.image_size // 4) ** 2, args.d_model
        )

        self.tf = TransformerDecoder(
            args.num_decoder_layers,
            (args.image_size // 4) ** 2,
            args.d_model,
            args.num_decoder_heads,
            args.dropout,
        )

        self.head = linear(args.d_model, args.vocab_size, bias=False)


class NlotmImageAutoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_iterations = args.num_iterations
        self.num_slots = args.num_slots
        self.cnn_hidden_size = args.cnn_hidden_size
        self.slot_size = args.slot_size
        self.mlp_hidden_size = args.mlp_hidden_size
        self.num_prototypes = args.num_prototypes
        self.image_channels = args.image_channels
        self.image_size = args.image_size
        self.vocab_size = args.vocab_size
        self.d_model = args.d_model
        self.num_blocks = args.num_blocks
        self.block_size = self.slot_size // self.num_blocks
        self.vq_type = args.vq_type
        if hasattr(args, 'downstream_data_type'):
            self.downstream_data_type = args.downstream_data_type
        if hasattr(args, 'downstream_type'):
            self.downstream_type = args.downstream_type
        else:
            self.downstream_type = None

        # dvae
        self.dvae = dVAE(args.vocab_size, args.image_channels)

        # encoder networks
        self.image_encoder = ImageEncoder(args)

        # decoder networks
        self.image_decoder = ImageDecoder(args)

    def forward(self, image, tau):
        """
        image: B, C, H, W
        tau: float
        """
        B, C, H, W = image.size()

        # dvae encode
        z_logits = F.log_softmax(
            self.dvae.encoder(image), dim=1
        )  # B, vocab_size, H_enc, W_enc
        z_soft = gumbel_softmax(
            z_logits, tau, False, dim=1
        )  # B, vocab_size, H_enc, W_enc
        z_hard = gumbel_softmax(
            z_logits, tau, True, dim=1
        ).detach()  # B, vocab_size, H_enc, W_enc
        z_hard = z_hard.permute(0, 2, 3, 1).flatten(
            start_dim=1, end_dim=2
        )  # B, H_enc * W_enc, vocab_size
        z_emb = self.image_decoder.dict(z_hard)  # B, H_enc * W_enc, d_model
        z_emb = torch.cat(
            [self.image_decoder.bos.expand(B, -1, -1), z_emb], dim=1
        )  # B, 1 + H_enc * W_enc, d_model
        z_emb = self.image_decoder.decoder_pos(z_emb)  # B, 1 + H_enc * W_enc, d_model

        # dvae recon
        dvae_recon = self.dvae.decoder(z_soft).reshape(B, C, H, W)  # B, C, H, W
        dvae_mse = ((image - dvae_recon) ** 2).sum() / B  # 1

        emb = self.image_encoder.cnn(image)  # B, cnn_hidden_size, H, W
        emb = self.image_encoder.pos(emb)  # B, cnn_hidden_size, H, W
        H_enc, W_enc = emb.shape[-2:]

        emb_set = emb.permute(0, 2, 3, 1).flatten(
            start_dim=1, end_dim=2
        )  # B, H * W, cnn_hidden_size
        emb_set = self.image_encoder.mlp(
            self.image_encoder.layer_norm(emb_set)
        )  # B, H * W, cnn_hidden_size
        emb_set = emb_set.reshape(
            B, H_enc * W_enc, self.d_model
        )  # B, H * W, cnn_hidden_size

        (
            slots,
            attns,
            indices,
            vq_loss,
            commitment_loss,
        ) = self.image_encoder.nlotm(
            emb_set
        )  # slots: B, num_slots, slot_size
        # attns: B, num_slots, num_inputs

        attns = (
            attns.transpose(-1, -2)
            .reshape(B, self.num_slots, 1, H_enc, W_enc)
            .repeat_interleave(H // H_enc, dim=-2)
            .repeat_interleave(W // W_enc, dim=-1)
        )  # B, num_slots, 1, H, W
        attns = image.unsqueeze(1) * attns + (1.0 - attns)  # B, num_slots, C, H, W

        slots = self.image_decoder.slot_proj(
            slots
        )  # B, num_slots, num_blocks * d_model
        # block coupling
        slots = slots + self.image_decoder.block_pos_proj(
            self.image_decoder.block_pos
        )  # B, num_slots, num_blocks * d_model
        slots = slots.reshape(
            B, self.num_slots, self.num_blocks, -1
        )  # B, num_slots, num_blocks, d_model
        slots = self.image_decoder.block_coupler(
            slots.flatten(end_dim=1)
        )  # B * num_slots, num_blocks, d_model
        slots = slots.reshape(
            B, self.num_slots * self.num_blocks, -1
        )  # B, num_slots * num_blocks, d_model

        # decode
        pred = self.image_decoder.tf(z_emb[:, :-1], slots)   # B, H_enc * W_enc, d_model
        pred = self.image_decoder.head(pred)  # B, H_enc * W_enc, vocab_size
        cross_entropy = -(z_hard * torch.log_softmax(pred, dim=-1)).sum() / B  # 1

        return (
            dvae_recon.clamp(0.0, 1.0),
            cross_entropy,
            dvae_mse,
            attns,
            vq_loss,
            commitment_loss,
        )

    def encode(self, image):
        """
        image: B, C, H, W
        """
        B, C, H, W = image.size()

        emb = self.image_encoder.cnn(image)  # B, cnn_hidden_size, H, W
        emb = self.image_encoder.pos(emb)  # B, cnn_hidden_size, H, W
        H_enc, W_enc = emb.shape[-2:]

        emb_set = emb.permute(0, 2, 3, 1).flatten(
            start_dim=1, end_dim=2
        )  # B, H * W, cnn_hidden_size
        emb_set = self.image_encoder.mlp(
            self.image_encoder.layer_norm(emb_set)
        )  # B, H * W, cnn_hidden_size
        emb_set = emb_set.reshape(
            B, H_enc * W_enc, self.d_model
        )  # B, H * W, cnn_hidden_size

        (
            slots,
            attns,
            indices,
            vq_loss,
            commitment_loss,
        ) = self.image_encoder.nlotm(
            emb_set
        )  # slots: B, num_slots, slot_size
        # attns: B, num_slots, num_inputs

        attns = (
            attns.transpose(-1, -2)
            .reshape(B, self.num_slots, 1, H_enc, W_enc)
            .repeat_interleave(H // H_enc, dim=-2)
            .repeat_interleave(W // W_enc, dim=-1)
        )  # B, num_slots, 1, H, W
        attns_vis = image.unsqueeze(1) * attns + (1.0 - attns)  # B, num_slots, C, H, W

        return slots, attns_vis, attns, indices

    def replace_block(self, slots, slot_idx, block_idx, new_block_value):
        assert slots.shape[0] == 1, "only slots from one image"

        new_slots = slots.clone()

        d_block = self.slot_size // self.num_blocks

        mem = self.image_encoder.nlotm.prototype_memory.get_mem()
        new_slots[0][slot_idx][block_idx * d_block : (block_idx + 1) * d_block] = mem[
            new_block_value
        ]

        return new_slots

    def decode(self, slots):
        """
        slots: B, N, slot_size
        """
        B, num_slots, slot_size = slots.size()
        H_enc, W_enc = (self.image_size // 4), (self.image_size // 4)
        gen_len = H_enc * W_enc

        slots = self.image_decoder.slot_proj(
            slots
        )  # B, num_slots, num_blocks * d_model
        # block coupling
        slots = slots + self.image_decoder.block_pos_proj(
            self.image_decoder.block_pos
        )  # B, num_slots, num_blocks * d_model
        slots = slots.reshape(
            B, self.num_slots, self.num_blocks, -1
        )  # B, num_slots, num_blocks, d_model
        slots = self.image_decoder.block_coupler(
            slots.flatten(end_dim=1)
        )  # B * num_slots, num_blocks, d_model
        slots = slots.reshape(
            B, self.num_slots * self.num_blocks, -1
        )  # B, num_slots * num_blocks, d_model

        # generate image tokens auto-regressively
        z_gen = slots.new_zeros(0)
        input = self.image_decoder.bos.expand(B, 1, -1)
        for t in range(gen_len):
            decoder_output = self.image_decoder.tf(
                self.image_decoder.decoder_pos(input),
                slots
            )
            z_next = F.one_hot(
                self.image_decoder.head(decoder_output)[:, -1:].argmax(dim=-1),
                self.vocab_size,
            )
            z_gen = torch.cat((z_gen, z_next), dim=1)
            input = torch.cat((input, self.image_decoder.dict(z_next)), dim=1)

        z_gen = z_gen.transpose(1, 2).float().reshape(B, -1, H_enc, W_enc)
        gen_transformer = self.dvae.decoder(z_gen)

        return gen_transformer.clamp(0.0, 1.0)

    def reconstruct_autoregressive(self, image):
        """
        image: batch_size x image_channels x H x W
        """
        B, C, H, W = image.size()
        slots, attns, _, _ = self.encode(image)
        recon_transformer = self.decode(slots)
        recon_transformer = recon_transformer.reshape(B, C, H, W)

        return recon_transformer

    def get_z(self, image):
        if self.downstream_data_type == 'dvae':
            # dvae encode
            z_logits = F.log_softmax(
                self.dvae.encoder(image), dim=1
            )  # B, vocab_size, H_enc, W_enc
            z_hard = gumbel_softmax(
                z_logits, 0.1, True, dim=1
            ).detach()  # B, vocab_size, H_enc, W_enc
            z_hard = z_hard.permute(0, 2, 3, 1).flatten(
                start_dim=1, end_dim=2
            )  # B, H_enc * W_enc, vocab_size
            z_hard = torch.argmax(z_hard, dim=-1)
            if self.downstream_type == 'dvae_tf':
                return z_hard
            elif self.downstream_type == 'dvae_cont_tf':
                z_one_hot = F.one_hot(
                    z_hard,
                    self.vocab_size,
                )
                z_cont = self.image_decoder.dict(z_one_hot)
                return z_cont
            else:
                raise ValueError()
        else:
            slots, _, _, indices = self.encode(image)
            if indices is None or self.downstream_type == 'cont_block_tf':
                z = slots
            else:
                z = indices
            B = z.shape[0]
            z = z.reshape(B, -1)
            return z

    def indices2slots(self, indices):
        mem = self.image_encoder.nlotm.prototype_memory.get_mem()
        slots = mem[indices.long()]
        slots = rearrange(slots, 'b (n m) d -> b n (m d)', n=self.num_slots)
        return slots

    def recon_z(self, z):
        if self.downstream_data_type == 'dvae':
            H = int(z.shape[1]**0.5)
            z = F.one_hot(z.long(), num_classes=self.vocab_size)
            z = rearrange(z, 'b (h w) d -> b d h w', h=H)
            recon = self.dvae.decoder(z.float())
            return recon
        else:
            if self.vq_type == 'none':
                slots = z
                recon = self.decode(slots)
                return recon
            else:
                # z are indices -> get slots
                slots = self.indices2slots(z)
                recon = self.decode(slots)
                return recon
