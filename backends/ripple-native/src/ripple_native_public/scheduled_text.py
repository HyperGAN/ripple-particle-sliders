"""Schedule the captured Ripple generator without changing its RAW source identity."""
from dataclasses import replace
from sys import float_info
from types import SimpleNamespace

from ripple_native.finished_text import generate as generate_original, validate_request


def step_strengths(gains, *, strength, max_new_tokens):
    if not isinstance(gains, (tuple, list)) or len(gains) != max_new_tokens:
        raise ValueError("One step gain per planned output token required")
    if any(type(gain) not in (int, float) or not 0 <= gain <= float_info.max
           or strength * gain > 1 for gain in gains):
        raise ValueError("Step gains must be finite and keep effective strength in [0, 1]")
    return tuple(strength * gain for gain in gains)


def generate(native, *, ids, max_new_tokens=64, strength=1.0, seed=0,
             output_type="text", step_gains):
    validate_request(ids=ids, max_new_tokens=max_new_tokens, strength=strength,
                     seed=seed, output_type=output_type)
    strengths = step_strengths(step_gains, strength=strength, max_new_tokens=max_new_tokens)
    runtime = native.runtime
    applied = []

    def hidden(context):
        position = len(context) - len(ids)
        if position != len(applied) or position >= len(strengths):
            raise RuntimeError("Ripple generation departed from its native token order")
        scale = strengths[position]
        with native.network.scaled(scale):
            result = runtime.hidden(context)
        applied.append(scale)
        return result

    scheduled_runtime = SimpleNamespace(
        device=runtime.device, hidden=hidden, identity=runtime.identity,
        load_tokenizer=runtime.load_tokenizer, store=runtime.store, model=runtime.model,
    )
    scheduled = SimpleNamespace(network=native.network, runtime=scheduled_runtime)
    result = generate_original(scheduled, ids=ids, max_new_tokens=max_new_tokens,
                               strength=strength, seed=seed, output_type=output_type)
    if len(applied) != len(result.native["new_token_ids"]):
        raise RuntimeError("Ripple did not produce one token per applied strength")
    return replace(result, native=dict(result.native, requested_step_strengths=list(strengths),
                                      applied_step_strengths=applied))
