from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from ripple_native import finished_text
from ripple_native_public import scheduled_text


def test_step_gains_follow_output_positions_and_stop_at_eos(monkeypatch):
    monkeypatch.setattr(finished_text, "require_owner", lambda device: None)
    seen = []
    active_scale = 7.0

    @contextmanager
    def scaled(value):
        nonlocal active_scale
        previous = active_scale
        active_scale = value
        try:
            yield
        finally:
            active_scale = previous

    def hidden(context):
        seen.append((list(context), active_scale))
        return torch.tensor([[[1., 0.]]]) if len(seen) == 1 else torch.tensor([[[0., 1.]]])

    runtime = SimpleNamespace(
        device=torch.device("cpu"), hidden=hidden, identity={"fixture": True},
        load_tokenizer=lambda: SimpleNamespace(decode=lambda tokens, **kwargs: str(tokens)),
        store=SimpleNamespace(read=lambda *args: torch.eye(2)),
        model=SimpleNamespace(config=SimpleNamespace(vocab_size=2, eos_token_id=1)),
    )
    native = SimpleNamespace(runtime=runtime, network=SimpleNamespace(scaled=scaled))
    result = scheduled_text.generate(native, ids=[0], strength=0.5, max_new_tokens=3,
                                    step_gains=[1.5, 0.5, 1.0])
    assert seen == [([0], 0.75), ([0, 0], 0.25)]
    assert active_scale == 7.0
    assert result.native["new_token_ids"] == [0, 1]
    assert result.native["applied_step_strengths"] == [0.75, 0.25]


@pytest.mark.parametrize("gains", [[0.5], [0.5, 1.5], [False, 0.5], [1.0, float("nan")]])
def test_bad_or_out_of_range_step_gains_rejected_before_native_load(gains):
    with pytest.raises(ValueError):
        scheduled_text.step_strengths(gains, max_new_tokens=2, strength=1.0)
