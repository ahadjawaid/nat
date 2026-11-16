# Copyright (c) 2026 Ahad Jawaid
"""The NAT codec's inference network: a causal convolutional encoder, a multi-scale finite scalar quantizer whose codes
become time-ordered tokens, and a causal decoder that turns the tokens back into an action chunk."""
import math

import torch
from torch import nn
from torch.nn import functional


class NATCodec(nn.Module):
    """Packed chunks of time_horizon steps x group_width columns, in [-1, 1], to tokens and back."""

    def __init__(self, config):
        super().__init__()
        encoder, quantizer, decoder = config["encoder"], config["quantizer"], config["decoder"]
        self.time_horizon, self.group_width = config["time_horizon"], config["group_width"]
        values_per_latent = math.prod(encoder["downsampling_ratios"])
        latents = self.time_horizon * self.group_width // values_per_latent
        self.encoder = ActionEncoder(**encoder)
        self.quantizer = MultiScaleFSQ(encoder["latent_dim"], latents=latents, **quantizer)
        self.decoder = ActionDecoder(encoder["latent_dim"], values_per_latent, latents=latents, **decoder)

    def encode(self, packed):
        """(batch, time_horizon, group_width) actions to (batch, tokens) token ids."""
        latents = self.encoder(packed.reshape(len(packed), 1, -1))  # time-major: each latent covers whole steps
        return self.quantizer.tokens(latents)

    def decode(self, tokens):
        """(batch, tokens) token ids to (batch, time_horizon, group_width) actions."""
        actions = self.decoder(self.quantizer.features(tokens))
        return actions.reshape(len(tokens), self.time_horizon, self.group_width)


class ActionEncoder(nn.Module):
    """A causal SEANet encoder: a convolution, then per downsampling ratio a residual unit and a strided convolution,
    an LSTM with a skip connection, and a convolution to the latent width."""

    def __init__(self, channels, filters, downsampling_ratios, kernel_size, residual_kernel_size, last_kernel_size,
                 compress, lstm_layers, latent_dim):
        super().__init__()
        self.input_conv = CausalConv1d(channels, filters, kernel_size)
        self.stages = nn.ModuleList()
        for ratio in downsampling_ratios:
            self.stages.append(EncoderStage(filters, filters // compress, residual_kernel_size, ratio))
            filters *= 2
        self.lstm = nn.LSTM(filters, filters, lstm_layers)
        self.output_conv = CausalConv1d(filters, latent_dim, last_kernel_size)

    def forward(self, actions):
        hidden = self.input_conv(actions)
        for stage in self.stages:
            hidden = stage(hidden)
        sequence = hidden.permute(2, 0, 1)
        hidden = (self.lstm(sequence)[0] + sequence).permute(1, 2, 0)
        return self.output_conv(functional.elu(hidden))


class EncoderStage(nn.Module):
    def __init__(self, channels, hidden_channels, residual_kernel_size, ratio):
        super().__init__()
        self.residual = ResidualUnit(channels, hidden_channels, residual_kernel_size)
        self.downsample = CausalConv1d(channels, 2 * channels, 2 * ratio, stride=ratio)

    def forward(self, hidden):
        return self.downsample(functional.elu(self.residual(hidden)))


class ResidualUnit(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size):
        super().__init__()
        self.conv = CausalConv1d(channels, hidden_channels, kernel_size)
        self.projection = CausalConv1d(hidden_channels, channels, 1)
        self.shortcut = CausalConv1d(channels, channels, 1)

    def forward(self, hidden):
        return self.shortcut(hidden) + self.projection(functional.elu(self.conv(functional.elu(hidden))))


class CausalConv1d(nn.Conv1d):
    """A convolution that sees only the current and earlier steps, its input zero-padded on the left (and on the right
    only to complete the last stride)."""

    def forward(self, hidden):
        kernel, stride = (self.kernel_size[0] - 1) * self.dilation[0] + 1, self.stride[0]
        left = kernel - stride
        frames = math.ceil((hidden.shape[-1] - kernel + left) / stride + 1)
        right = (frames - 1) * stride + kernel - left - hidden.shape[-1]
        return super().forward(functional.pad(hidden, (left, right)))


class MultiScaleFSQ(nn.Module):
    """Residual finite scalar quantization at several temporal scales: scale s pools scales[s] latents into one code
    of prod(levels) values. A chunk's tokens are ordered by the first latent each code covers, coarse scales first,
    and scale s's codes are offset by s x codebook_size, so a prefix of whole latents decodes on its own."""

    def __init__(self, latent_dim, levels, scales, latents):
        super().__init__()
        self.scales, self.latents, self.codebook_size = scales, latents, math.prod(levels)
        self.project_in = nn.ModuleList([nn.Linear(latent_dim, len(levels)) for _ in scales])
        self.project_out = nn.ModuleList([nn.Linear(len(levels), latent_dim) for _ in scales])
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.int32), persistent=False)
        self.register_buffer("basis", torch.cumprod(torch.tensor([1] + levels[:-1]), 0, dtype=torch.int32),
                             persistent=False)
        self.token_order = sorted((position * span, scale, position) for scale, span in enumerate(scales)
                                  for position in range(-(-latents // span)))
        for scale in range(len(scales)):
            self.register_buffer(f"scale_{scale}_columns", torch.tensor(
                [index for index, (_, owner, _) in enumerate(self.token_order) if owner == scale]), persistent=False)

    def tokens(self, latents):
        residual, codes = latents, []
        for scale, span in enumerate(self.scales):
            pooled = functional.avg_pool1d(residual, span) if span > 1 else residual
            indices, quantized = self.quantize(self.project_in[scale](pooled.transpose(1, 2)).float())
            residual = residual - self.upsampled(self.project_out[scale](quantized).transpose(1, 2), span)
            codes.append(indices)
        return torch.stack([codes[scale][:, position] + scale * self.codebook_size
                            for _, scale, position in self.token_order], dim=1)

    def features(self, tokens):
        features = 0
        for scale, span in enumerate(self.scales):
            indices = tokens.index_select(1, getattr(self, f"scale_{scale}_columns")) - scale * self.codebook_size
            codes = self.codes(indices.clamp(0, self.codebook_size - 1))
            features = features + self.upsampled(self.project_out[scale](codes).transpose(1, 2), span)
        return features

    def quantize(self, projected):
        """Bounded, rounded values (codes in [-1, 1]) and their codebook indices."""
        half_level = (self.levels - 1) * (1 + 1e-3) / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        bounded = (projected + (offset / half_level).atanh()).tanh() * half_level - offset
        codes = (bounded + (bounded.round() - bounded)) / (self.levels // 2)
        indices = ((codes * (self.levels // 2) + self.levels // 2) * self.basis).sum(-1).to(torch.int32)
        return indices.long(), codes

    def codes(self, indices):
        digits = (indices.unsqueeze(-1) // self.basis) % self.levels
        return (digits - self.levels // 2) / (self.levels // 2)

    def upsampled(self, quantized, span):
        return torch.repeat_interleave(quantized, span, dim=-1)[..., :self.latents]


class ActionDecoder(nn.Module):
    """A causal Vocos-style decoder: a convolution, residual and causal attention blocks, a conditioned layer norm,
    ConvNeXt blocks and a linear head to each latent's values_per_latent time-major action values."""

    def __init__(self, latent_dim, values_per_latent, latents, dim, intermediate_dim, num_blocks, condition_embeddings,
                 condition_id, attention_context, group_norm_groups):
        super().__init__()
        self.embed = CausalConv1d(latent_dim, dim, 7)
        self.temporal_blocks = nn.Sequential(
            ResnetBlock(dim, group_norm_groups), ResnetBlock(dim, group_norm_groups),
            CausalSelfAttention(dim, group_norm_groups, attention_context, latents),
            ResnetBlock(dim, group_norm_groups), ResnetBlock(dim, group_norm_groups))
        self.temporal_norm = PerTimeGroupNorm(group_norm_groups, dim, eps=1e-6)
        self.input_norm = ConditionedLayerNorm(condition_embeddings, dim)
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim, intermediate_dim, condition_embeddings)
                                     for _ in range(num_blocks)])
        self.output_norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, values_per_latent)
        self.register_buffer("condition_id", torch.tensor([condition_id]), persistent=False)

    def forward(self, features):
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled, benchmark=torch.backends.cudnn.benchmark,
                                        deterministic=torch.backends.cudnn.deterministic, allow_tf32=False):
            hidden = self.temporal_norm(self.temporal_blocks(self.embed(features)))
            hidden = self.input_norm(hidden.transpose(1, 2), self.condition_id).transpose(1, 2)
            for block in self.blocks:
                hidden = block(hidden, self.condition_id)
            return self.head(self.output_norm(hidden.transpose(1, 2))).reshape(len(features), -1)


class ResnetBlock(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.norm1, self.conv1 = PerTimeGroupNorm(groups, channels, eps=1e-6), CausalConv1d(channels, channels, 3)
        self.norm2, self.conv2 = PerTimeGroupNorm(groups, channels, eps=1e-6), CausalConv1d(channels, channels, 3)

    def forward(self, hidden):
        return hidden + self.conv2(swish(self.norm2(self.conv1(swish(self.norm1(hidden))))))


class CausalSelfAttention(nn.Module):
    """Attention over the current and up to context - 1 earlier latents."""

    def __init__(self, channels, groups, context, latents):
        super().__init__()
        self.norm = PerTimeGroupNorm(groups, channels, eps=1e-6)
        self.query, self.key, self.value, self.output = (nn.Conv1d(channels, channels, 1) for _ in range(4))
        distances = torch.arange(latents)[:, None] - torch.arange(latents)[None, :]
        self.register_buffer("allowed", (distances >= 0) & (distances < context), persistent=False)

    def forward(self, hidden):
        normalized = self.norm(hidden)
        queries, keys, values = self.query(normalized), self.key(normalized), self.value(normalized)
        weights = torch.bmm(queries.transpose(1, 2), keys) * queries.shape[1] ** -.5
        allowed = self.allowed[:queries.shape[-1], :keys.shape[-1]]
        weights = weights.masked_fill(~allowed, -torch.inf).softmax(-1)
        return hidden + self.output(torch.bmm(values, weights.transpose(1, 2)))


class PerTimeGroupNorm(nn.GroupNorm):
    """Group normalization of each latent on its own, so no latent sees a later one."""

    def forward(self, hidden):
        batch, channels, length = hidden.shape
        normalized = super().forward(hidden.transpose(1, 2).reshape(batch * length, channels, 1))
        return normalized.reshape(batch, length, channels).transpose(1, 2)


class ConditionedLayerNorm(nn.Module):
    """Layer normalization with a learned scale and shift per condition id."""

    def __init__(self, conditions, channels):
        super().__init__()
        self.scale, self.shift = nn.Embedding(conditions, channels), nn.Embedding(conditions, channels)

    def forward(self, hidden, condition_id):
        normalized = functional.layer_norm(hidden, (hidden.shape[-1],), eps=1e-6)
        return normalized * self.scale(condition_id) + self.shift(condition_id)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, intermediate_channels, conditions):
        super().__init__()
        self.depthwise_conv = CausalConv1d(channels, channels, 7, groups=channels)
        self.norm = ConditionedLayerNorm(conditions, channels)
        self.expand, self.contract = nn.Linear(channels, intermediate_channels), nn.Linear(intermediate_channels, channels)
        self.layer_scale = nn.Parameter(torch.ones(channels))

    def forward(self, hidden, condition_id):
        update = self.norm(self.depthwise_conv(hidden).transpose(1, 2), condition_id)
        update = self.layer_scale * self.contract(functional.gelu(self.expand(update)))
        return hidden + update.transpose(1, 2)


def swish(hidden):
    return hidden * torch.sigmoid(hidden)
