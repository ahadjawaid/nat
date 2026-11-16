# Copyright (c) 2026 Ahad Jawaid
"""NAT action tokenizer: 32-step action chunks to 14 time-ordered tokens per group of up to 7 action columns, and back.

    from nat_tokenizer import NATActionTokenizer
    tokenizer = NATActionTokenizer.from_pretrained()   # or AutoProcessor.from_pretrained(REPOSITORY, trust_remote_code=True)
    tokenizer.fit(dataset_actions)                 # per-column min and max, mapped to [-1, 1]
    tokens = tokenizer(action_chunks)              # (batch, 32, action_dim) or (32, action_dim) -> list of token lists
    actions = tokenizer.decode(tokens, action_dim=action_dim)   # -> (batch, 32, action_dim), in the original units

With normalize=False the actions must already lie in [-1, 1]. Tokens are ids in [0, vocab_size). Each group's tokens are
time-ordered, so a prefix of prefix_token_count(steps) tokens decodes the chunk's first steps (one group only)."""
import json
from pathlib import Path

import numpy as np
import torch

CONFIG_NAME, MODEL_NAME, REPOSITORY = "processor_config.json", "nat_tokenizer.pt", "ahad-j/nat-action-tokenizer"


class NATActionTokenizer:
    def __init__(self, folder, action_min=None, action_max=None, **config):
        self.folder, self.device, self.cuda_graphs = Path(folder), torch.device("cpu"), None
        self.model = torch.jit.load(str(self.folder / MODEL_NAME), map_location="cpu").eval()
        self.action_min = None if action_min is None else np.asarray(action_min, dtype=np.float32)
        self.action_max = None if action_max is None else np.asarray(action_max, dtype=np.float32)
        self.horizon, self.slots, self.vocab_size = config["time_horizon"], config["group_width"], config["vocab_size"]
        self.prefix_tokens = {int(steps): tokens for steps, tokens in config["prefix_token_counts"].items()}

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoProcessor"):
        """Called by transformers' AutoProcessor on remote-code classes; nothing to register."""

    @classmethod
    def from_pretrained(cls, name_or_path=REPOSITORY, **keywords):
        folder = Path(name_or_path)
        if not folder.is_dir():
            from huggingface_hub import snapshot_download
            folder = Path(snapshot_download(str(name_or_path), token=keywords.get("token"), revision=keywords.get("revision")))
        config = json.loads((folder / CONFIG_NAME).read_text())
        config.update({name: value for name, value in keywords.items() if name in ("action_min", "action_max")})
        return cls(folder, **{name: value for name, value in config.items() if name not in ("auto_map", "processor_class")})

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in (MODEL_NAME, Path(__file__).name, "README.md"):
            if (self.folder / name).exists() and self.folder != directory:
                (directory / name).write_bytes((self.folder / name).read_bytes())
        config = json.loads((self.folder / CONFIG_NAME).read_text())
        config.update(action_min=None if self.action_min is None else self.action_min.tolist(),
                      action_max=None if self.action_max is None else self.action_max.tolist())
        (directory / CONFIG_NAME).write_text(json.dumps(config, indent=2))

    def to(self, device, cuda_graphs=False):
        """Runs the encoder and decoder on device (e.g. "cuda"); inputs and outputs stay NumPy. With cuda_graphs each
        is captured once per batch shape as a CUDA graph and replayed (fewer kernel launches, same results)."""
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.cuda_graphs = {} if cuda_graphs else None
        return self

    def fit(self, actions):
        """Per-column min and max of a dataset's actions (any shape ending in action_dim), used to normalize."""
        actions = np.asarray(actions, dtype=np.float32).reshape(-1, np.shape(actions)[-1])
        self.action_min, self.action_max = actions.min(0), actions.max(0)
        return self

    def tokens_per_chunk(self, action_dim):
        return len(self._groups(action_dim)) * self.prefix_tokens[self.horizon]

    def prefix_token_count(self, steps):
        """Tokens that decode a single group's first steps (steps must end a latent: 8, 16, 24 or 32)."""
        return self.prefix_tokens[steps]

    def __call__(self, actions, normalize=True):
        actions = np.asarray(actions, dtype=np.float32)
        actions = actions[None] if actions.ndim == 2 else actions
        if actions.ndim != 3 or actions.shape[1] != self.horizon:
            raise ValueError(f"Expected action chunks of shape (batch, {self.horizon}, action_dim)")
        if not np.isfinite(actions).all():
            raise ValueError("Actions contain NaN or infinite values")
        if normalize:
            low, high = self._limits(actions)
            actions = np.where(high > low, 2 * (actions - low) / np.where(high > low, high - low, 1) - 1, 0)
        packed = self._packed(np.clip(actions, -1, 1))
        with torch.no_grad():
            tokens = self._run("encode", torch.from_numpy(packed).to(self.device))
        return tokens.reshape(len(actions), -1).tolist()

    def decode(self, tokens, action_dim, normalize=True):
        """Token lists (each a whole chunk, or for one group a prefix of whole latents) to actions."""
        groups = self._groups(action_dim)
        tokens = np.asarray(tokens, dtype=np.int64).reshape(len(tokens), -1)
        steps = {count: steps for steps, count in self.prefix_tokens.items()}.get(tokens.shape[1])
        if len(groups) == 1 and steps is not None and steps < self.horizon:
            tokens = np.pad(tokens, ((0, 0), (0, self.prefix_tokens[self.horizon] - tokens.shape[1])))
        elif tokens.shape[1] != self.tokens_per_chunk(action_dim):
            raise ValueError(f"Expected {self.tokens_per_chunk(action_dim)} tokens per chunk for {action_dim} columns")
        with torch.no_grad():
            packed = self._run("decode", torch.from_numpy(tokens.reshape(-1, self.prefix_tokens[self.horizon])).to(self.device))
        packed = np.clip(packed.cpu().numpy().reshape(len(tokens), len(groups), self.horizon, self.slots), -1, 1)
        actions = np.concatenate([packed[:, index, :, :len(group)] for index, group in enumerate(groups)], axis=-1)
        actions = actions[:, :steps or self.horizon]
        if not normalize:
            return actions
        low, high = self._limits(actions)
        return (actions + 1) / 2 * (high - low) + low

    def _run(self, method, inputs):
        if self.cuda_graphs is None:
            return getattr(self.model, method)(inputs)
        if (method, inputs.shape) not in self.cuda_graphs:
            static, stream = inputs.clone(), torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    getattr(self.model, method)(static)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = getattr(self.model, method)(static)
            self.cuda_graphs[method, inputs.shape] = static, graph, output
        static, graph, output = self.cuda_graphs[method, inputs.shape]
        static.copy_(inputs)
        graph.replay()
        return output.clone()

    def _groups(self, action_dim):
        return [range(start, min(start + self.slots, action_dim)) for start in range(0, action_dim, self.slots)]

    def _packed(self, actions):
        groups = self._groups(actions.shape[-1])
        packed = np.zeros((len(actions), len(groups), self.horizon, self.slots), dtype=np.float32)
        for index, group in enumerate(groups):
            packed[:, index, :, :len(group)] = actions[..., group.start:group.stop]
        return packed.reshape(-1, self.horizon, self.slots)

    def _limits(self, actions):
        if self.action_min is None or len(self.action_min) != actions.shape[-1]:
            raise ValueError("normalize=True needs the actions' per-column min and max: call fit(dataset_actions), pass "
                             "action_min/action_max to from_pretrained, or pass normalize=False for actions in [-1, 1]")
        return self.action_min, self.action_max
