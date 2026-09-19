import math

import torch
import torch.nn as nn
import torch.nn.functional as functional
from mamba_ssm import Mamba


class MedianTopKPool1d(nn.Module):
    def __init__(self, frequency, retention_ratio=0.25):
        super().__init__()
        self.frequency = frequency
        self.k = max(1, math.ceil(retention_ratio * frequency))

    def forward(self, x):
        batch_size, channels, length = x.shape
        if length % self.frequency:
            raise ValueError("Input length must contain complete one-second groups.")
        grouped = x.reshape(batch_size, channels, length // self.frequency, self.frequency)
        ordered = grouped.sort(dim=-1).values
        median = (ordered[..., (self.frequency - 1) // 2] + ordered[..., self.frequency // 2]) * 0.5
        topk = grouped.topk(self.k, dim=-1).values.mean(dim=-1)
        return torch.cat([median, topk], dim=1)


class DownsampleBlock(nn.Module):
    def __init__(self, frequency, embedding_dim, retention_ratio=0.25):
        super().__init__()
        self.frequency = frequency
        self.conv = nn.Conv1d(1, embedding_dim, kernel_size=frequency)
        self.pool = MedianTopKPool1d(frequency, retention_ratio)
        self.projection = nn.Linear(2 * embedding_dim, embedding_dim)

    def forward(self, x):
        batch_size, seconds, frequency = x.shape
        if frequency != self.frequency:
            raise ValueError("Input frequency does not match encoder frequency.")
        x = x.reshape(batch_size, 1, seconds * frequency)
        left_padding = (frequency - 1) // 2
        x = functional.pad(x, (left_padding, frequency - 1 - left_padding))
        x = functional.gelu(self.conv(x))
        return self.projection(self.pool(x).transpose(1, 2))


class MultiLagVariableAttention(nn.Module):
    def __init__(self, embedding_dim, lags, head_dim):
        super().__init__()
        if not lags or 0 not in lags or any(lag < 0 for lag in lags):
            raise ValueError("Lags must be non-negative and include zero.")
        self.lags = tuple(lags)
        self.num_heads = len(lags)
        self.head_dim = head_dim
        self.query = nn.Linear(embedding_dim, self.num_heads * head_dim, bias=False)
        self.key = nn.Linear(embedding_dim, self.num_heads * head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, self.num_heads * head_dim, bias=False)
        self.output = nn.Linear(self.num_heads * head_dim, embedding_dim, bias=False)
        self.router = nn.Linear(3 * embedding_dim, self.num_heads)

    def forward(self, x, return_attention=False):
        batch_size, steps, variables, _ = x.shape
        previous = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        router_input = torch.cat([x, previous, x - previous], dim=-1)
        logits = self.router(router_input)
        lags = x.new_tensor(self.lags)
        valid = torch.arange(steps, device=x.device)[:, None] >= lags[None, :]
        routing = torch.softmax(logits.masked_fill(~valid[None, :, None], float("-inf")), dim=-1)
        query, key, value = [
            layer(x).reshape(batch_size, steps, variables, self.num_heads, self.head_dim)
            for layer in (self.query, self.key, self.value)
        ]
        outputs, attention_maps = [], []
        for head, lag in enumerate(self.lags):
            if lag >= steps:
                output = x.new_zeros(batch_size, steps, variables, self.head_dim)
                weights = x.new_zeros(batch_size, steps, variables, variables)
            else:
                q = query[:, lag:, :, head]
                k = key[:, :steps - lag, :, head]
                v = value[:, :steps - lag, :, head]
                active_weights = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim), dim=-1)
                active_output = active_weights @ v
                output = torch.cat([x.new_zeros(batch_size, lag, variables, self.head_dim), active_output], dim=1)
                weights = torch.cat([x.new_zeros(batch_size, lag, variables, variables), active_weights], dim=1)
            outputs.append(output * routing[..., head, None])
            attention_maps.append(weights)
        output = self.output(torch.cat(outputs, dim=-1))
        if return_attention:
            return output, routing, torch.stack(attention_maps, dim=3)
        return output


class InteractionBlock(nn.Module):
    def __init__(self, embedding_dim, lags, head_dim, dropout):
        super().__init__()
        self.attention = MultiLagVariableAttention(embedding_dim, lags, head_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )

    def forward(self, x, return_attention=False):
        if return_attention:
            attended, routing, attention = self.attention(x, return_attention=True)
        else:
            attended = self.attention(x)
        x = x + attended
        x = x + self.ffn(self.norm(x))
        if return_attention:
            return x, routing, attention
        return x


class MambaBlock(nn.Module):
    def __init__(self, hidden_dim, state_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mamba = Mamba(d_model=hidden_dim, d_state=state_dim, d_conv=4, expand=2)

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class LAVAN(nn.Module):
    def __init__(self, parameter_groups, height_variable, directional_variables=(), discrete_variables=(), embedding_dim=32, token_dim=64, height_dim=32, pooling_dim=32, state_dim=16, lags=(0, 1, 2, 4), head_dim=32, dropout=0.3):
        super().__init__()
        self.frequencies = (1, 2, 4, 8, 16)
        self.parameter_groups = {frequency: tuple(parameter_groups.get(frequency, ())) for frequency in self.frequencies}
        self.variable_names = tuple(name for frequency in self.frequencies for name in self.parameter_groups[frequency])
        if not self.variable_names or len(set(self.variable_names)) != len(self.variable_names):
            raise ValueError("Variable names must be unique and non-empty.")
        if height_variable not in self.variable_names:
            raise ValueError("The height variable must be included in parameter_groups.")
        self.directional_variables = set(directional_variables)
        self.discrete_variables = set(discrete_variables)
        self.height_index = self.variable_names.index(height_variable)
        self.embedding_dim = embedding_dim
        self.encoders = nn.ModuleDict()
        for frequency, names in self.parameter_groups.items():
            for name in names:
                if name in self.directional_variables:
                    if frequency != 1:
                        raise ValueError("Directional variables must have frequency 1.")
                    self.encoders[name] = nn.Linear(2, embedding_dim)
                elif frequency == 1 or name in self.discrete_variables:
                    self.encoders[name] = nn.Linear(1, embedding_dim)
                else:
                    self.encoders[name] = DownsampleBlock(frequency, embedding_dim)
        self.interaction = InteractionBlock(embedding_dim, lags, head_dim, dropout)
        self.variable_fusion = nn.Sequential(nn.Linear(len(self.variable_names) * embedding_dim, token_dim), nn.LayerNorm(token_dim), nn.GELU(), nn.Dropout(dropout))
        self.temporal = MambaBlock(token_dim, state_dim)
        self.height_projection = nn.Linear(embedding_dim, height_dim)
        self.hidden_projection = nn.Linear(token_dim, pooling_dim)
        self.height_score = nn.Linear(height_dim, pooling_dim, bias=False)
        self.score = nn.Linear(pooling_dim, 1, bias=False)
        self.classifier = nn.Linear(token_dim, 1)

    def encode_frequency(self, x, frequency):
        names = self.parameter_groups[frequency]
        if x.shape[1] != len(names):
            raise ValueError(f"Expected {len(names)} variables at {frequency} Hz.")
        if not names:
            return x.new_zeros(x.shape[0], 0, x.shape[2], self.embedding_dim)
        features = []
        for index, name in enumerate(names):
            values = x[:, index]
            if name in self.directional_variables:
                radians = torch.deg2rad(values)
                values = torch.stack([radians.sin(), radians.cos()], dim=-1)
            elif name in self.discrete_variables:
                values = values[..., -1:]
            elif frequency == 1:
                values = values.unsqueeze(-1)
            features.append(self.encoders[name](values))
        return torch.stack(features, dim=1)

    def forward(self, *inputs, return_explanations=False):
        if len(inputs) != len(self.frequencies):
            raise ValueError("One input tensor is required for each supported frequency.")
        encoded = torch.cat([self.encode_frequency(x, frequency) for x, frequency in zip(inputs, self.frequencies)], dim=1).transpose(1, 2)
        height = self.height_projection(encoded[:, :, self.height_index])
        if return_explanations:
            interaction, routing, attention = self.interaction(encoded, return_attention=True)
        else:
            interaction = self.interaction(encoded)
        batch_size, steps, variables, embedding_dim = interaction.shape
        hidden = self.temporal(self.variable_fusion(interaction.reshape(batch_size, steps, variables * embedding_dim)))
        weights = torch.softmax(self.score(torch.tanh(self.hidden_projection(hidden) + self.height_score(height))), dim=1)
        logits = self.classifier((hidden * weights).sum(dim=1)).squeeze(-1)
        if return_explanations:
            relevance = (routing.unsqueeze(-1) * attention).sum(dim=3).mean(dim=(1, 2))
            return logits, {"variables": self.variable_names, "lag_weights": routing.detach(), "variable_attention": attention.detach(), "time_weights": weights.squeeze(-1).detach(), "variable_importance": relevance.detach()}
        return logits

