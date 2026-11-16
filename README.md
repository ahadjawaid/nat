# NAT action tokenizer

Turns robot action chunks into discrete tokens for a vision-language-action model, and tokens back into actions.
Every 32-step chunk of up to 7 action columns becomes **14 tokens**, in time order, so a model can stream actions
before it has generated the whole chunk.

## Quick start

```bash
pip install "git+https://github.com/ahadjawaid/nat.git"
```

The weights and config download from the Hugging Face repository
[ahad-j/nat-action-tokenizer](https://huggingface.co/ahad-j/nat-action-tokenizer) on first use. While the GitHub and
Hugging Face repositories are private, install and load with an account that can read them (`huggingface-cli login`).

```python
import numpy as np
from nat_tokenizer import NATActionTokenizer

tokenizer = NATActionTokenizer.from_pretrained()   # downloads ahad-j/nat-action-tokenizer

# 1. Fit the normalization once, on your dataset's actions (any shape ending in action_dim).
tokenizer.fit(dataset_actions)

# 2. Tokenize 32-step chunks: (batch, 32, action_dim) or one chunk (32, action_dim).
tokens = tokenizer(action_chunks)            # list (one per chunk) of lists of ints in [0, 25600)

# 3. Decode back to actions in your original units.
actions = tokenizer.decode(tokens, action_dim=action_chunks.shape[-1])   # (batch, 32, action_dim)

# 4. Save the fitted tokenizer (keeps your min and max) and load it later the same way.
tokenizer.save_pretrained("my-nat-tokenizer")
tokenizer = NATActionTokenizer.from_pretrained("my-nat-tokenizer")
```

Without installing the package, `transformers` can load the same tokenizer from the Hugging Face repository's code
(like `physical-intelligence/fast`):

```python
from transformers import AutoProcessor
tokenizer = AutoProcessor.from_pretrained("ahad-j/nat-action-tokenizer", trust_remote_code=True)
```

Tokens and reconstructions are identical to the original NAT research code this tokenizer was trained and evaluated
with (checked on CPU and GPU); this package is a standalone export of it.

## Normalization (on by default)

The tokenizer was trained on actions in [-1, 1]. With `normalize=True` (the default) every column is mapped from
its min and max to [-1, 1] before encoding (values outside are clipped) and decoded actions are mapped back, so you
work in your own units throughout. Set the min and max with `fit(dataset_actions)`, or pass them directly:

```python
tokenizer = AutoProcessor.from_pretrained("ahad-j/nat-action-tokenizer", trust_remote_code=True,
                                          action_min=[...], action_max=[...])
```

If your actions are already in [-1, 1], pass `normalize=False` to both calls:
`tokenizer(chunks, normalize=False)` and `tokenizer.decode(tokens, action_dim=d, normalize=False)`.
A column whose min equals its max is encoded as 0 and decoded back to its constant value.

## Streaming (causal decoding)

Within a group the 14 tokens are time ordered: the first 4 decode the chunk's first 8 steps, the first 7 its first 16,
the first 11 its first 24, and all 14 the whole chunk. A prefix decodes exactly the same first steps as the full chunk,
so a model can execute actions while it is still generating:

```python
for steps in (8, 16, 24, 32):
    prefix = [chunk[:tokenizer.prefix_token_count(steps)] for chunk in tokens]
    first_steps = tokenizer.decode(prefix, action_dim=7)      # (batch, steps, 7)
```

Streaming works for actions of up to 7 columns (one group).

## Action dimensions

Columns are split into consecutive groups of up to 7 (8 columns = 7 + 1, 14 = 7 + 7, 21 = 7 + 7 + 7). Each group gets
its own 14 tokens and a chunk's tokens are the groups' tokens one after another: `tokenizer.tokens_per_chunk(action_dim)`
gives the count (14 per group). Chunks must be exactly 32 steps long.

## GPU and CUDA graphs

```python
tokenizer = tokenizer.to("cuda")                      # eager on the GPU
tokenizer = tokenizer.to("cuda", cuda_graphs=True)    # captures the encoder and decoder once per batch shape
```

Inputs and outputs stay NumPy. GPU results match the CPU up to floating point (rarely a token on a quantization
boundary differs). The first few calls are slower while TorchScript optimizes the graph.

## API

| call | does |
|---|---|
| `NATActionTokenizer.from_pretrained(repo_or_folder, action_min=None, action_max=None)` | load (optionally with your min and max) |
| `tokenizer.fit(actions)` | per-column min and max of your actions (returns the tokenizer) |
| `tokenizer(chunks, normalize=True)` | `(batch, 32, d)` or `(32, d)` actions to a list of token lists |
| `tokenizer.decode(tokens, action_dim, normalize=True)` | token lists (whole chunks, or one group's prefix) to `(batch, steps, d)` actions |
| `tokenizer.prefix_token_count(steps)` | tokens that decode the first `steps` (8, 16, 24 or 32) |
| `tokenizer.tokens_per_chunk(action_dim)` | 14 per group of 7 columns |
| `tokenizer.to(device, cuda_graphs=False)` | run on a device, optionally with CUDA graphs |
| `tokenizer.save_pretrained(directory)` | save, with the fitted min and max |
| `tokenizer.vocab_size` | 25,600 (4 scales x 6,400 codes) |

## Model

4 latents per chunk, 4 FSQ scales of 6,400 codes (one coarse token per two latents, three finer tokens per latent),
trained on a large corpus of robot and human action data (every window once, batch 4,096). The model is TorchScript, so
loading needs only torch and numpy.

## Reconstruction on robot benchmarks, and speed

Random 32-step chunks of each benchmark's demonstrations, each normalized by its own min and max (MetaWorld's actions clipped to the [-1, 1] its simulator executes). Errors are mean absolute differences between the input and its reconstruction. "Original NAT code" is the research code this tokenizer was trained with, run on the same chunks; this package reproduces it exactly.

| data | chunks x shape | MAE in [-1, 1] units | MAE | max error | original NAT code MAE | identical tokens |
|---|---|---|---|---|---|---|
| LIBERO-10 | 1000 x (32, 7) | 0.0304 | 0.0210 | 1.9762 | 0.0210 | 100.0% |
| MetaWorld mt4 (clipped) | 1000 x (32, 4) | 0.0651 | 0.0651 | 1.7578 | 0.0651 | 100.0% |
| RoboMimic mt4 | 1000 x (32, 7) | 0.0466 | 0.0404 | 1.5712 | 0.0404 | 100.0% |
| RoboCasa mt4 | 1000 x (32, 12) | 0.0130 | 0.0107 | 0.9970 | 0.0107 | 100.0% |

| implementation | device | batch of 32 x 7 chunks | encode ms | decode ms |
|---|---|---|---|---|
| nat-tokenizer | cpu, eager | 1 | 10.07 ± 0.03 | 19.47 ± 0.39 |
| original NAT code | cpu, eager | 1 | 12.14 ± 0.10 | 23.72 ± 0.26 |
| nat-tokenizer | cpu, eager | 64 | 318.17 ± 0.08 | 243.77 ± 0.62 |
| original NAT code | cpu, eager | 64 | 321.37 ± 0.63 | 251.12 ± 0.62 |
| nat-tokenizer | cuda, eager | 1 | 2.72 ± 0.07 | 4.68 ± 0.13 |
| original NAT code | cuda, eager | 1 | 5.84 ± 0.32 | 8.79 ± 0.20 |
| nat-tokenizer | cuda, eager | 64 | 2.71 ± 0.04 | 4.99 ± 0.12 |
| original NAT code | cuda, eager | 64 | 5.58 ± 0.13 | 8.95 ± 0.16 |
| nat-tokenizer | cuda, CUDA graphs | 1 | 0.44 ± 0.02 | 2.07 ± 0.02 |
| nat-tokenizer | cuda, CUDA graphs | 64 | 0.74 ± 0.01 | 3.49 ± 0.03 |

## Development

```bash
pip install -e ".[test]"
pytest
```

## License

MIT, see [LICENSE](LICENSE).
