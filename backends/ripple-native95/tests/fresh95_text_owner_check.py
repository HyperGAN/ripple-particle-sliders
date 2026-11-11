"""Four NEW synthetic facade cases; UNRUN, no native model or optimizer update.

Explicit opt-in filename keeps ordinary source-tree CI discovery unchanged.
"""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from particle_sliders.inference import InferenceSession
from particle_sliders.lifecycle import Capabilities, LifecycleError
from ripple_native95_public import text_api as api


def _session(render, close=lambda: None):
    network = nn.Linear(2, 2)
    network.register_buffer("observed", torch.tensor([3.]))
    capabilities = Capabilities("routed_particle_gan", render=True,
        render_outputs=("text",), operations=("render",))
    preflights = []
    binding = SimpleNamespace(network=network, render=render, close=close,
                              observation=None)
    def preflight(*args):
        preflights.append(args)
    session = InferenceSession(binding, capabilities,
        {"selection": "raw", "network": {}}, "text", preflight)
    return session, preflights


def test_text_forwarding_filters_none_and_preserves_original_owner(monkeypatch):
    calls, opened, closed = [], [], []
    sentinel = object()
    original, preflights = _session(lambda **kw: calls.append(kw) or sentinel,
                                    lambda: closed.append(True))
    network = original.binding.network
    parameter_ids = [id(p) for p in network.parameters()]
    def open_original(artifact, **kw):
        opened.append((artifact, kw))
        return original
    monkeypatch.setattr(api, "_open_original", open_original)
    with api.open_text_inference("typed-RAW", options={"device": "cpu"},
            native_request={"ids": [1], "images": None}) as facade:
        assert facade._session is original
        assert facade.render(ids=[1], max_new_tokens=2, images=None,
                             strength=.5, seed=7) is sentinel
        assert facade.selection == "raw"
        assert [id(p) for p in network.parameters()] == parameter_ids
    assert closed == [True] and facade.closed
    assert opened == [("typed-RAW", {"options": {"device": "cpu"},
        "strength": 1., "native_request": {"ids": [1]}})]
    assert preflights == [("render", .5, False, "text", {"ids": [1], "max_new_tokens": 2})]
    assert calls == [{"strength": .5, "seed": 7, "output_type": "text",
                      "ids": [1], "max_new_tokens": 2}]


def test_non_none_references_refused_before_open_or_native_work(monkeypatch):
    opened, calls = [], []
    monkeypatch.setattr(api, "_open_original", lambda *a, **kw: opened.append(kw))
    for images in ([], [object()]):
        with pytest.raises(ValueError, match="references"):
            api.open_text_inference("RAW", images=images)
        with pytest.raises(ValueError, match="references"):
            api.open_text_inference("RAW", native_request={"images": images})
    assert opened == []
    original, preflights = _session(lambda **kw: calls.append(kw))
    facade = api.TextInferenceSession(original)
    with pytest.raises(ValueError, match="references"):
        facade.render(ids=[1], images=[])
    assert preflights == [] and calls == []
    facade.close()


def test_render_failure_restores_original_observation_and_closes():
    closed = []
    original = None
    def render(**kw):
        original.binding.network.observed.add_(9.)
        original.binding.network.train()
        torch.rand(3)
        raise RuntimeError("new text callback failure")
    original, _ = _session(render, lambda: closed.append(True))
    network = original.binding.network
    rng = torch.get_rng_state().clone()
    values = {k: v.clone() for k, v in network.state_dict().items()}
    was_training = network.training
    with pytest.raises(RuntimeError, match="new text callback failure"):
        with api.TextInferenceSession(original) as facade:
            facade.render(ids=[1], seed=11)
    assert torch.equal(torch.get_rng_state(), rng)
    assert network.training is was_training
    assert all(torch.equal(network.state_dict()[k], v) for k, v in values.items())
    assert closed == [True] and original.closed


def test_original_failed_close_blocks_work_and_remains_retryable():
    closes, calls = [], []
    def close():
        closes.append(True)
        if len(closes) == 1:
            raise RuntimeError("first close failed")
    original, _ = _session(lambda **kw: calls.append(kw), close)
    facade = api.TextInferenceSession(original)
    with pytest.raises(RuntimeError, match="first close failed"):
        facade.close()
    assert not facade.closed
    with pytest.raises(LifecycleError, match="requires cleanup"):
        facade.render(ids=[1])
    assert calls == []
    facade.close()
    facade.close()
    assert facade.closed and len(closes) == 2
