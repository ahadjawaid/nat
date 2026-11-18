# Copyright (c) 2026 Ahad Jawaid
import numpy as np
import pytest
import torch

from nat_tokenizer import NATActionTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return NATActionTokenizer.from_pretrained()


def chunks(action_dim, low=-1.0, high=1.0, seed=0):
    walk = np.cumsum(np.random.default_rng(seed).normal(0, 0.08, (5, 32, action_dim)), axis=1)
    return (low + (high - low) * (np.tanh(walk) + 1) / 2).astype(np.float32)


@pytest.mark.parametrize("action_dim", [1, 6, 7, 8, 14])
def test_round_trip_shapes_and_token_range(tokenizer, action_dim):
    actions = chunks(action_dim)
    tokens = tokenizer(actions, normalize=False)
    assert len(tokens) == 5 and all(len(chunk) == tokenizer.tokens_per_chunk(action_dim) for chunk in tokens)
    assert 0 <= min(map(min, tokens)) and max(map(max, tokens)) < tokenizer.vocab_size
    decoded = tokenizer.decode(tokens, action_dim=action_dim, normalize=False)
    assert decoded.shape == actions.shape and np.abs(decoded - actions).mean() < 0.1


def test_normalization_round_trips_in_original_units(tokenizer):
    raw = chunks(7, low=-3.0, high=7.5)
    tokenizer.fit(raw)
    decoded = tokenizer.decode(tokenizer(raw), action_dim=7)
    assert decoded.min() >= raw.min() - 1e-5 and decoded.max() <= raw.max() + 1e-5
    assert np.abs(decoded - raw).mean() < 0.1 * (raw.max() - raw.min())


def test_constant_column_decodes_to_its_value(tokenizer):
    raw = np.concatenate([chunks(6), np.full((5, 32, 1), 4.0, dtype=np.float32)], axis=-1)
    tokenizer.fit(raw)
    np.testing.assert_allclose(tokenizer.decode(tokenizer(raw), action_dim=7)[..., -1], 4.0)


@pytest.mark.parametrize("steps", [8, 16, 24])
def test_streamed_prefix_decodes_the_full_chunks_first_steps(tokenizer, steps):
    tokens = tokenizer(chunks(7), normalize=False)
    prefix = [chunk[:tokenizer.prefix_token_count(steps)] for chunk in tokens]
    np.testing.assert_allclose(tokenizer.decode(prefix, action_dim=7, normalize=False),
                               tokenizer.decode(tokens, action_dim=7, normalize=False)[:, :steps], atol=1e-6)


def test_missing_statistics_and_bad_input_raise(tokenizer):
    fresh = NATActionTokenizer.from_pretrained()
    with pytest.raises(ValueError, match="fit"):
        fresh(chunks(7))
    bad = chunks(7)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fresh(bad, normalize=False)


def test_save_and_reload_keep_the_statistics(tokenizer, tmp_path):
    raw = chunks(7, low=0.0, high=2.0)
    tokenizer.fit(raw).save_pretrained(tmp_path)
    assert NATActionTokenizer.from_pretrained(tmp_path)(raw) == tokenizer(raw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_gpu_and_cuda_graphs_agree_with_the_cpu(tokenizer):
    actions = chunks(8)
    cpu_tokens = np.asarray(tokenizer(actions, normalize=False))
    for cuda_graphs in (False, True):
        gpu = NATActionTokenizer.from_pretrained().to("cuda", cuda_graphs=cuda_graphs)
        assert np.mean(np.asarray(gpu(actions, normalize=False)) == cpu_tokens) > 0.99


def test_the_downloaded_repository_holds_only_data(tokenizer):
    assert sorted(path.name for path in tokenizer.folder.iterdir()) == ["LICENSE", "README.md", "config.json",
                                                                        "model.safetensors"]


@pytest.mark.parametrize("normalize", [False, True])
def test_streaming_returns_each_latents_actions_as_its_tokens_arrive(tokenizer, normalize):
    raw = chunks(6, low=-2.0, high=3.0)
    tokenizer.fit(raw)
    actions = raw if normalize else np.clip(raw, -1, 1)
    tokens = tokenizer(actions[:1], normalize=normalize)[0]
    stream = tokenizer.stream(action_dim=6, normalize=normalize)
    arrived = [stream.push(token) for token in tokens]
    assert [index + 1 for index, latent in enumerate(arrived) if len(latent)] == [4, 7, 11, 14]
    assert {len(latent) for latent in arrived} == {0, 8}
    np.testing.assert_allclose(np.concatenate(arrived), tokenizer.decode([tokens], action_dim=6, normalize=normalize)[0],
                               atol=1e-6)
    with pytest.raises(ValueError, match="complete"):
        stream.push(tokens[0])


def test_streaming_needs_a_single_group(tokenizer):
    with pytest.raises(ValueError, match="single group"):
        tokenizer.stream(action_dim=8)


def test_a_chunk_starting_with_the_executing_actions_decodes_back_to_them(tokenizer):
    raw = chunks(7, low=-2.0, high=3.0)
    tokenizer.fit(raw)
    executing = raw[:, :16]
    prefix = tokenizer.tokens_starting_with(executing)
    assert [len(tokens) for tokens in prefix] == [tokenizer.prefix_token_count(16)] * len(raw)
    padded = np.concatenate([executing, np.repeat(executing[:, -1:], 16, axis=1)], axis=1)
    np.testing.assert_allclose(tokenizer.decode(prefix, action_dim=7),
                               tokenizer.decode(tokenizer(padded), action_dim=7)[:, :16], atol=1e-5)
    with pytest.raises(ValueError, match="steps"):
        tokenizer.tokens_starting_with(raw[:, :10])
