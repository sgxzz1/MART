import random
import urllib
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import hashlib
import os
import math
import logging
import numpy as np
from generate_trajectory import generate_trajectory_true

logger = logging.getLogger(__name__)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        if config.bias:
            # key, query, value projections for all heads
            self.key = nn.Linear(config.n_embd, config.n_embd)
            self.query = nn.Linear(config.n_embd, config.n_embd)
            self.value = nn.Linear(config.n_embd, config.n_embd)
            # output projection
            self.proj = nn.Linear(config.n_embd, config.n_embd)
        else:
            # key, query, value projections for all heads
            self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
            # output projection
            self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.max_seqlen, config.max_seqlen))
                             .view(1, 1, config.max_seqlen, config.max_seqlen))
        self.register_buffer("mask_bias",
                             torch.triu(torch.ones(config.max_seqlen, config.max_seqlen) * float('-inf'),
                                        diagonal=1).view(1, 1, config.max_seqlen, config.max_seqlen))
        self.n_head = config.n_head
        self.attn_weights = 0

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.mul(att, self.mask[:, :, :T, :T])
        att = att + self.mask_bias[:, :, :T, :T]
        att = F.softmax(att, dim=-1)
        self.attn_weights = att
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Cosais(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.visual = ModifiedResNet(vision_layers, embed_dim, transformer_heads, image_resolution, vision_width)
        self.cf = config
        self.DisLoss = config.DisLoss
        self.precise = config.precise
        self.device = config.device
        self.bs = config.batch_size
        self.AssLoss = config.AssLoss
        self.lat_size = config.lat_size
        self.lon_size = config.lon_size
        self.sog_size = config.sog_size
        self.cog_size = config.cog_size
        self.full_size = config.full_size
        self.n_lat_embd = config.n_lat_embd
        self.n_lon_embd = config.n_lon_embd
        self.n_sog_embd = config.n_sog_embd
        self.n_cog_embd = config.n_cog_embd
        self.target_points = config.target_points
        self.register_buffer(
            "att_sizes",
            torch.tensor([config.lat_size, config.lon_size, config.sog_size, config.cog_size]))

        if hasattr(config, "lat_min"):  # the ROI is provided.
            self.lat_min = config.lat_min
            self.lat_max = config.lat_max
            self.lon_min = config.lon_min
            self.lon_max = config.lon_max
            self.lat_range = config.lat_max - config.lat_min
            self.lon_range = config.lon_max - config.lon_min
            self.sog_range = 30.

        if hasattr(config, "mode"):  # mode: "pos" or "velo".
            # "pos": predict directly the next positions.
            # "velo": predict the velocities, use them to
            # calculate the next positions.
            self.mode = config.mode
        else:
            self.mode = "pos"
        self.lat_emb = nn.Embedding(self.lat_size, config.n_lat_embd)
        self.lon_emb = nn.Embedding(self.lon_size, config.n_lon_embd)
        self.sog_emb = nn.Embedding(self.sog_size, config.n_sog_embd)
        self.cog_emb = nn.Embedding(self.cog_size, config.n_cog_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # self.attn_weights = torch.tensor(0., requires_grad=True)
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.head = nn.Linear(config.n_embd, self.full_size, bias=False)
        self.max_seqlen = config.max_seqlen
        # self.apply(functools.partial(self._init_weights, config.finetune))
        self.apply(self._init_weights)
        # if config.finetune:
        #     self.lat_emb.weight.requires_grad = False
        #     self.lon_emb.weight.requires_grad = False
        #     self.sog_emb.weight.requires_grad = False
        #     self.cog_emb.weight.requires_grad = False
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_max_seqlen(self):
        return self.max_seqlen

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():

            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # elif pn == 'visual.attnpool.positional_embedding':
                #     decay.add(mn)
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('logit_scale')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def to_indexes(self, x):
        """Convert tokens to indexes.

        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated
                to [0,1).
            model: currently only supports "uniform".

        Returns:
            idxs: a Tensor (dtype: Long) of indexes.
        """
        idxs = (x * self.att_sizes).long()

        return idxs, idxs

    def cos_loss(self, similarity, targets):
        cos_loss = (F.cross_entropy(similarity, targets, reduction='none') +
                    F.cross_entropy(similarity.t(), targets, reduction='none')) / 2.0
        return cos_loss

    def cos_cal(self, image: torch.tensor, traj: torch.tensor, masks: torch.tensor):
        device = traj.device
        image = image.to(device)
        masks = masks.to(device)
        masks_cp = masks.clone()
        bs = masks_cp.shape[0]
        length = len(masks_cp[0])

        image_norm = image.norm(dim=-1, keepdim=True)
        traj_norm = traj.norm(dim=-1, keepdim=True)
        image_ = image / image_norm  # shape:t, c
        traj_ = traj / traj_norm
        targets = torch.arange(length).to(device)
        cos_loss = torch.tensor(0)
        for i in range(bs):
            similarity = self.logit_scale.exp() * image_[i] @ traj_[i].t()
            traj_len = masks_cp[i].sum()
            mask_bias = torch.ones((length, length)) * (-10000.)
            mask_bias = mask_bias.to(device)
            mask_bias[:traj_len.long(), :traj_len.long()] = 0
            tmp = self.cos_loss(similarity + mask_bias, targets)
            tmp = (tmp * masks_cp[i]).sum() / masks_cp[i].sum()
            cos_loss = cos_loss + tmp
        cos_loss = cos_loss / bs
        return cos_loss

    def generate_trajectories(self, batch_size, device):
        list = []
        for i in range(batch_size):
            tmp = generate_trajectory_true(i, self.max_seqlen + 1, self.cf)
            list.append(tmp)
        numpy_list = np.array(list)
        sim_traj = torch.tensor(numpy_list).to(device)
        sim_masks = torch.ones(batch_size, self.max_seqlen).to(device)
        return sim_traj, sim_masks

    def forward(self, seqs, masks=None, grid_indexes=None, with_targets=False):
        """
                Args:
                    seqs: a Tensor of size (batchsize, seqlen, 2). x has been truncated
                        to [0,1).
                    grid_indexes: a Tensor of size (batchsize, seqlen, points, 2)
                    masks: a Tensor of the same size of x. masks[idx] = 0. if
                        x[idx] is a padding.
                    with_targets: if True, inputs = x[:,:-1,:], targets = x[:,1:,:],
                        otherwise inputs = x.
                Returns:
                    logits, loss, cos_loss
                """
        # Convert to indexes
        device = seqs.device
        batch_size = seqs.size(0)
        idxs, idxs_uniform = self.to_indexes(seqs)
        if with_targets:
            inputs = idxs[:, :-1, :].contiguous()
            targets = idxs[:, 1:, :].contiguous()

        else:
            inputs = idxs
            targets = None
        _, seqlen, _ = inputs.size()
        assert seqlen <= self.max_seqlen + 1, "Cannot forward, model block size is exhausted."
        # forward the GPT model
        lat_embeddings = self.lat_emb(inputs[:, :, 0])  # (bs, seqlen, lat_size)
        lon_embeddings = self.lon_emb(inputs[:, :, 1])
        sog_embeddings = self.sog_emb(inputs[:, :, 2])
        cog_embeddings = self.cog_emb(inputs[:, :, 3])
        token_emb = torch.cat((lat_embeddings, lon_embeddings), dim=-1)
        # each position maps to a (learnable) vector (1, seqlen, n_embd)
        position_emb = self.pos_emb[:, :seqlen, :]
        fea = token_emb


        batchsize, seqlen, _ = fea.shape
        fea = torch.cat((fea, torch.cat((sog_embeddings, cog_embeddings), dim=-1)), dim=-1)
        if self.AssLoss and with_targets:
            sim_traj, sim_masks = self.generate_trajectories(batch_size, device)
            sim_idxs, _ = self.to_indexes(sim_traj[:, :, :4])
            sim_inputs = sim_idxs[:, :-1]
            sim_targets = sim_idxs[:, 1:]
            sim_lat = self.lat_emb(sim_inputs[:, :, 0])
            sim_lon = self.lon_emb(sim_inputs[:, :, 1])
            sim_sog = self.sog_emb(sim_inputs[:, :, 2])
            sim_cog = self.cog_emb(sim_inputs[:, :, 3])
            fea = torch.cat((fea, torch.cat((sim_lat, sim_lon, sim_sog, sim_cog), dim=-1)), dim=0)
        fea = fea + position_emb

        fea = self.drop(fea)
        fea = self.blocks(fea)
        fea = self.ln_f(fea)  # (bs, seqlen, n_embd)
        logits = self.head(fea)  # (bs, seqlen, full_size)
        lat_logits, lon_logits, sog_logits, cog_logits = \
            torch.split(logits, (self.lat_size, self.lon_size, self.sog_size, self.cog_size), dim=-1)
        # Calculate the loss
        loss = None
        sim_loss = None
        if targets is not None:
            if self.AssLoss and with_targets:
                #     lat_loss = F.cross_entropy(lat_logits.view(-1, self.lat_size),
                #                                targets[:, :, 0].view(-1),
                #                                reduction="none").view(batchsize, seqlen)
                #     lon_loss = F.cross_entropy(lon_logits.view(-1, self.lon_size),
                #                                targets[:, :, 1].view(-1),
                #                                reduction="none").view(batchsize, seqlen)
                #     loss_tuple = (lat_loss, lon_loss)

                # else:
                lat_logits, sim_lat_logits = torch.split(lat_logits, (batchsize, batchsize), dim=0)
                lon_logits, sim_lon_logits = torch.split(lon_logits, (batchsize, batchsize), dim=0)
                sog_logits, sim_sog_logits = torch.split(sog_logits, (batchsize, batchsize), dim=0)
                cog_logits, sim_cog_logits = torch.split(cog_logits, (batchsize, batchsize), dim=0)
                sim_lat_loss = F.cross_entropy(sim_lat_logits.view(-1, self.lat_size),
                                               sim_targets[:, :, 0].reshape(-1),
                                               reduction="none").view(lat_logits.size(0), seqlen)
                sim_lon_loss = F.cross_entropy(sim_lon_logits.view(-1, self.lon_size),
                                               sim_targets[:, :, 1].reshape(-1),
                                               reduction="none").view(lon_logits.size(0), seqlen)
                sim_loss_tuple = (sim_lat_loss, sim_lon_loss)
                sim_loss = sum(sim_loss_tuple)
                sim_loss = (sim_loss * sim_masks).sum(dim=1) / sim_masks.sum(dim=1)
                sim_loss = sim_loss.mean()
            if self.DisLoss:
                lat_probs = F.softmax(lat_logits, dim=-1)
                lon_probs = F.softmax(lon_logits, dim=-1)
                sog_probs = F.softmax(sog_logits, dim=-1)
                cog_probs = F.softmax(cog_logits, dim=-1)

                # 这里是由于使用预训练之后开始训练的时候由于某些值的logit值太大，导致softmax转为概率分布的时候有的值太小就变成0，在对数化操作会出现问题
                if torch.min(lat_probs) == 0 or torch.min(lon_probs) == 0:
                    lat_probs = lat_probs + 1e-10
                    lon_probs = lon_probs + 1e-10

                lat_probs_log = torch.log(lat_probs)
                lon_probs_log = torch.log(lon_probs)
                sog_probs_log = torch.log(sog_probs)
                cog_probs_log = torch.log(cog_probs)

                std = 0.5
                if self.precise:
                    std_ = 0.8
                else:
                    std_ = 0.5
                x_lat = torch.arange(self.lat_size).to(device)
                x_lon = torch.arange(self.lon_size).to(device)
                x_sog = torch.arange(self.sog_size).to(device)
                x_cog = torch.arange(self.cog_size).to(device)

                mean_lat = targets[:, :, 0].reshape(-1, 1)
                mean_lon = targets[:, :, 1].reshape(-1, 1)
                mean_sog = targets[:, :, 2].reshape(-1, 1)
                mean_cog = targets[:, :, 3].reshape(-1, 1)

                gaussian_lat = torch.exp(-0.5 * ((x_lat - mean_lat) / std_) ** 2).view(batchsize, seqlen, -1)
                gaussian_lon = torch.exp(-0.5 * ((x_lon - mean_lon) / std_) ** 2).view(batchsize, seqlen, -1)
                gaussian_sog = torch.exp(-0.5 * ((x_sog - mean_sog) / std) ** 2).view(batchsize, seqlen, -1)
                gaussian_cog = torch.exp(-0.5 * ((x_cog - mean_cog) / std) ** 2).view(batchsize, seqlen, -1)

                gaussian_lat = gaussian_lat / gaussian_lat.sum(dim=-1)[:, :, None]
                gaussian_lon = gaussian_lon / gaussian_lon.sum(dim=-1)[:, :, None]
                gaussian_sog = gaussian_sog / gaussian_sog.sum(dim=-1)[:, :, None]
                gaussian_cog = gaussian_cog / gaussian_cog.sum(dim=-1)[:, :, None]

                gaussian_lat = -gaussian_lat
                gaussian_lon = -gaussian_lon
                gaussian_sog = -gaussian_sog
                gaussian_cog = -gaussian_cog

                lat_loss = (lat_probs_log * gaussian_lat).sum(dim=-1)
                lon_loss = (lon_probs_log * gaussian_lon).sum(dim=-1)
                sog_loss = (sog_probs_log * gaussian_sog).sum(dim=-1)
                cog_loss = (cog_probs_log * gaussian_cog).sum(dim=-1)
            else:
                lat_loss = F.cross_entropy(lat_logits.view(-1, self.lat_size),
                                           targets[:, :, 0].view(-1),
                                           reduction="none").view(batchsize, seqlen)
                lon_loss = F.cross_entropy(lon_logits.view(-1, self.lon_size),
                                           targets[:, :, 1].view(-1),
                                           reduction="none").view(batchsize, seqlen)
                sog_loss = F.cross_entropy(sog_logits.view(-1, self.sog_size),
                                           targets[:, :, 2].view(-1),
                                           reduction="none").view(batchsize, seqlen)
                cog_loss = F.cross_entropy(cog_logits.view(-1, self.cog_size),
                                           targets[:, :, 3].view(-1),
                                           reduction="none").view(batchsize, seqlen)

            loss_tuple = (lat_loss, lon_loss, sog_loss, cog_loss)
            # loss_tuple = (lat_loss, lon_loss)
            loss = sum(loss_tuple)
            if masks is not None:
                loss = (loss * masks).sum(dim=1) / masks.sum(dim=1)

            loss = loss.mean()
        if sim_loss is not None:
            loss += sim_loss
        return logits, loss, sim_loss
