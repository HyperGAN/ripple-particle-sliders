"""Shared activation steering on Ripple's growing native full prefix."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from math import isfinite
from types import SimpleNamespace

import torch

from ripple_native.finished_text import generate as generate_original, validate_request
from .steering_core import CORE_REVISION, SteeringCore, require_steering_core


@dataclass(frozen=True, kw_only=True)
class PrefixGate:
    """A calibrated shared gate on one Ripple decoder input."""

    core: SteeringCore
    layer: int
    method: str
    parameter: float
    intervention: str
    evidence_ref: str
    direction: torch.Tensor | None = None
    seed: int | None = None
    clabs: object = None
    preserve_norm: bool = False
    projection_bounds: tuple[float, float] | None = None


@dataclass(frozen=True, kw_only=True)
class PrefixPromptSteering:
    """A fitted shared S-PSR module for one Ripple decoder input."""

    core: SteeringCore
    layer: int
    fitted: torch.nn.Module
    evidence_ref: str


def _unit_direction(value, width):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.shape != (width,):
        raise ValueError("Provide the calibrated projection direction at the selected Ripple layer")
    vector = value.detach().float()
    length = vector.norm().item()
    if not isfinite(length) or length == 0:
        raise ValueError("The calibrated direction must be finite and nonzero")
    return vector / length


def _policy(steering, layers, width):
    if not isinstance(steering, (PrefixGate, PrefixPromptSteering)):
        raise TypeError("Provide PrefixGate or PrefixPromptSteering")
    if (type(steering.layer) is not int or not 0 <= steering.layer < len(layers)
            or not isinstance(steering.evidence_ref, str) or not steering.evidence_ref.strip()):
        raise ValueError("Provide a native Ripple decoder layer and calibration or fit evidence")
    core = steering.core
    require_steering_core(core)
    if isinstance(steering, PrefixPromptSteering):
        fitted = steering.fitted
        if (not isinstance(fitted, core.prompt.PromptSteering)
                or fitted.probe.in_features != width
                or any(parameter.dtype != torch.float32 or not torch.isfinite(parameter).all().item()
                       for parameter in fitted.parameters())):
            raise ValueError("Use a finite fitted shared FP32 S-PSR module with Ripple's hidden width")
        return "s_psr", "residual", None, None, deepcopy(fitted).eval().requires_grad_(False)

    method, intervention = steering.method, steering.intervention
    if intervention not in ("residual", "raw_adapter"):
        raise ValueError("Select residual or raw_adapter intervention")
    if type(steering.preserve_norm) is not bool:
        raise ValueError("preserve_norm must be Boolean")
    bounds = steering.projection_bounds
    if bounds is not None:
        low, high = bounds
        if not isfinite(low) or not isfinite(high) or low > high:
            raise ValueError("Projection bounds must be finite and ordered")
    if intervention == "raw_adapter" and (
        steering.layer != 0 or steering.preserve_norm or bounds is not None
    ):
        raise ValueError("RAW gating reads layer 0 before the adapter and uses its original branches")
    direction = None
    if method != "random" or intervention == "residual":
        direction = _unit_direction(steering.direction, width)
    gate = core.adaptive.ActivationGate(method, steering.parameter, seed=steering.seed,
                                       clabs=steering.clabs)
    return method, intervention, gate, direction, None


def generate(native, *, ids, max_new_tokens=64, strength=1., seed=0,
             output_type="text", steering):
    """Use the original native generator with one adaptive decision per new position."""
    validate_request(ids=ids, max_new_tokens=max_new_tokens, strength=strength,
                     seed=seed, output_type=output_type)
    runtime = native.runtime
    layers = runtime.model.layers
    width = runtime.model.config.hidden_size
    method, intervention, gate, direction, fitted = _policy(steering, layers, width)
    prefix = len(ids) - 1  # The last prompt position predicts the first output.
    trace = []
    receipt = dict(core_revision=CORE_REVISION, method=method, intervention=intervention,
                   decoder_input_layer=steering.layer, evidence_ref=steering.evidence_ref,
                   nominal_strength=strength, prefill_only_positions=prefix, prefix_steps=trace)
    if gate is not None:
        receipt.update(parameter=steering.parameter, seed=steering.seed,
                       clabs=asdict(steering.clabs) if steering.clabs is not None else None,
                       preserve_norm=steering.preserve_norm,
                       projection_bounds=steering.projection_bounds)
    request = dict(ids=ids, max_new_tokens=max_new_tokens, seed=seed, output_type=output_type)
    if strength == 0:
        result = generate_original(native, strength=0., **request)
        receipt["effective_strengths"] = [0.] * len(result.native["new_token_ids"])
        if gate is not None:
            receipt["usage"] = asdict(gate.usage)
        return replace(result, native=dict(result.native, adaptive_steering=receipt))

    adaptive = steering.core.adaptive
    expected = None
    raw_mask = None
    scale = ExitStack()

    def edit_layer(_layer, args):
        nonlocal expected, raw_mask
        hidden = args[0]
        if expected is None or hidden.shape != (1, expected, width):
            raise RuntimeError("Ripple did not visit one full native residual per forward")
        index = expected - prefix - 1
        expected = None
        latest = hidden[:, -1:]
        record = dict(output_index=index, prefix_position=prefix + index)
        if fitted is not None:
            fitted.to(device=hidden.device)
            edited_latest, coefficient = fitted(latest, strength)
            effective = (coefficient * strength).item()
            if not isfinite(effective) or not torch.isfinite(edited_latest).all().item():
                raise ValueError("Nonfinite Ripple S-PSR intervention")
            # Reuse prior effective coefficients without evaluating the probe on old rows.
            older = [adaptive.steer_residual(hidden[:, pos:pos + 1], fitted.direction.weight[0],
                                              strength=step["effective_strength"], gate=True)
                     for pos, step in enumerate(trace, start=prefix)]
            edited = torch.cat((hidden[:, :prefix], *older, edited_latest), dim=1)
            record["coefficient"] = coefficient.item()
        else:
            score = None
            if method != "random":
                score = adaptive.activation_projection(latest, direction, strength=strength).item()
            decision = gate.step(score)
            choices = [False] * prefix + [step["fired"] for step in trace] + [decision.fired]
            effective = strength if decision.fired else 0.
            record.update(alignment=score, probability=decision.probability, fired=decision.fired)
            if intervention == "raw_adapter":
                edited = hidden
                if any(choices):
                    raw_mask = torch.tensor(choices, device=hidden.device, dtype=torch.bool)
                    scale.enter_context(native.network.scaled(strength))
            elif any(choices):
                mask = torch.tensor(choices, device=hidden.device, dtype=torch.bool)
                edited = adaptive.steer_residual(
                    hidden, direction, strength=strength, gate=mask,
                    preserve_norm=steering.preserve_norm,
                    projection_bounds=steering.projection_bounds,
                )
            else:
                edited = hidden
        if edited is not hidden and not torch.isfinite(edited).all().item():
            raise ValueError("Nonfinite Ripple residual after replay")
        trace.append(dict(record, effective_strength=effective))
        return (edited, *args[1:]) if edited is not hidden else None

    def mask_raw_delta(_graph, _args, delta):
        # The unchanged HNDL branch flattens [batch, prefix] in token order.
        if raw_mask is None or delta.ndim != 3 or delta.shape[:2] != (raw_mask.numel(), 1):
            raise RuntimeError("Ripple RAW projection departed from native prefix order")
        return delta.masked_fill(~raw_mask[:, None, None], 0)

    def hidden(context):
        nonlocal expected, raw_mask
        position = len(trace)
        if (position >= max_new_tokens or len(context) != len(ids) + position
                or context[:len(ids)] != list(ids)):
            raise RuntimeError("Ripple generation departed from its growing native prefix")
        expected = len(context)
        try:
            value = runtime.hidden(context)
        finally:
            scale.close()
            raw_mask = None
        if expected is not None:
            raise RuntimeError("Ripple did not visit its selected decoder layer")
        return value

    routed_runtime = SimpleNamespace(
        device=runtime.device, hidden=hidden, identity=runtime.identity,
        load_tokenizer=runtime.load_tokenizer, store=runtime.store, model=runtime.model,
    )
    routed = SimpleNamespace(network=native.network, runtime=routed_runtime)
    with ExitStack() as hooks:
        hooks.callback(layers[steering.layer].register_forward_pre_hook(edit_layer).remove)
        if intervention == "raw_adapter":
            adapters = tuple(native.network.adapters.values())
            if len(adapters) != 4 * len(layers):
                raise ValueError("Ripple RAW gating requires all native attention adapter branches")
            for adapter in adapters:
                hooks.callback(adapter.graph.register_forward_hook(mask_raw_delta).remove)
        result = generate_original(routed, strength=0., **request)
    if len(trace) != len(result.native["new_token_ids"]):
        raise RuntimeError("Ripple did not produce one native token per decision")
    receipt["effective_strengths"] = [step["effective_strength"] for step in trace]
    if gate is not None:
        receipt["usage"] = asdict(gate.usage)
    return replace(result, native=dict(result.native, adaptive_steering=receipt))
