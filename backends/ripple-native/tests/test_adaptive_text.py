from contextlib import contextmanager
import json
from os import environ
from random import Random
from types import MethodType, SimpleNamespace
from zipfile import ZipFile

import pytest
import torch

from ripple_native.adapter import Slider
from ripple_native.finished_text import generate as original
from ripple_native.fibo_hndl import HNDLParticleProjection, verified_builder
from ripple_native.streaming import StreamingModel
from ripple_native.vendor.configuration_ricursive import RicursiveConfig
from ripple_native.vendor.modeling_ricursive import RicursiveForCausalLM
from ripple_native_public.adaptive_text import PrefixGate, PrefixPromptSteering, generate
from ripple_native_public.steering_core import load_steering_core
from ripple_native_public.text_api import TextInferenceSession


@pytest.fixture
def core():
    wheel = environ.get("RIPPLE_TEST_STEERING_WHEEL")
    if not wheel:
        pytest.skip("Set RIPPLE_TEST_STEERING_WHEEL to the verified core #269 wheel")
    return load_steering_core(wheel)


@pytest.fixture
def native():
    # The real packed stream and vendor local/global attention on tiny CPU tensors.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        config = RicursiveConfig(
            vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            max_position_embeddings=128, sliding_window=4,
            layer_types=["sliding_attention", "full_attention"], use_gated_attention=True,
            gate_full_precision=True, use_valres=True, embed_scale=4., ffn_clamp_threshold=20.,
            use_cache=False, bos_token_id=1, eos_token_id=7, pad_token_id=0,
        )
        resident = RicursiveForCausalLM(config).eval().requires_grad_(False)
        cache = dict(resident.state_dict())
        # A stable greedy full-head output makes the mixed-gate prefix comparable.
        cache["lm_head.weight"] = torch.zeros_like(cache["lm_head.weight"])
        for index, layer in enumerate(resident.model.layers):
            for leaf in ("gate", "up", "down"):
                linear = getattr(layer.mlp, leaf + "_proj")
                prefix = f"model.layers.{index}.mlp.{leaf}_proj."
                cache[prefix + "weight"] = torch.randint(
                    0, 256, (linear.out_features, linear.in_features // 2), dtype=torch.uint8,
                )
                cache[prefix + "weight_scale"] = torch.ones(
                    linear.out_features, linear.in_features // 16,
                ).to(torch.float8_e4m3fn)
                cache[prefix + "weight_scale_2"] = torch.tensor(.04)
                cache[prefix + "input_scale"] = torch.tensor(.01)

        class Store:
            entries = cache

            def read(self, name, device):
                return cache[name].to(device)

            def rows(self, name, ids, device):
                return cache[name][ids].to(device)

            @contextmanager
            def block(self, index, device):
                prefix = f"model.layers.{index}."
                from ripple_native.streaming import CURRENT
                values = {key: value.to(device) for key, value in cache.items() if key.startswith(prefix)}
                token = CURRENT.set({"store": self, "tensors": values})
                try:
                    yield values
                finally:
                    CURRENT.reset(token)

        store = Store()
        device = torch.device("cpu")
        model = StreamingModel(config, store, device=device, chunk_rows=3).eval()
        builder, identity = verified_builder()
        bank = torch.randn(128, 4)
        adapters = {}
        for index, block in enumerate(model.layers):
            for leaf in ("q_proj", "k_proj", "v_proj", "o_proj"):
                host = getattr(block.native.self_attn, leaf)
                name = f"{index}.{leaf}"
                branch = HNDLParticleProjection(
                    name, host, bank, 8, 8, builder=builder, identity=identity,
                )
                with torch.no_grad():
                    branch.lora_up.weight.fill_(.25)
                adapters[name] = branch
    network = SimpleNamespace(adapters=adapters, closed=False)
    network.scaled = MethodType(Slider.scaled, network)
    runtime = SimpleNamespace(device=device, model=model, hidden=model.hidden, store=store,
                              identity={"fixture": True}, load_tokenizer=lambda:
                              SimpleNamespace(decode=lambda ids, **_: str(ids)))
    return SimpleNamespace(runtime=runtime, network=network)


def _request():
    return dict(ids=[1, 2], max_new_tokens=3, strength=.5)


def _capture_hidden(native, render):
    seen = []
    original_hidden = native.runtime.hidden

    def hidden(ids):
        value = original_hidden(ids)
        seen.append(value.clone())
        return value

    native.runtime.hidden = hidden
    try:
        return render(), seen
    finally:
        native.runtime.hidden = original_hidden


def test_cfp_replays_prior_residual_decisions_on_the_growing_prefix(native, core):
    model = native.runtime.model
    initial = model.store.rows("model.embed_tokens.weight", torch.tensor([[2, 0]]), "cpu")[0]
    direction = (initial[1] - initial[0]).float()
    direction /= direction.norm()
    scores = initial.float() @ direction * model.config.embed_scale
    threshold = scores.mean().item()
    policy = PrefixGate(core=core, layer=0, method="cfp", parameter=threshold,
                        direction=direction, evidence_ref="fixture", intervention="residual")
    result, actual = _capture_hidden(native, lambda: generate(native, steering=policy, **_request()))
    assert result.native["new_token_ids"] == [0, 0, 0]
    receipt = result.native["adaptive_steering"]
    assert [step["fired"] for step in receipt["prefix_steps"]] == [True, False, False]
    assert [step["prefix_position"] for step in receipt["prefix_steps"]] == [1, 2, 3]
    assert receipt["usage"] == {"fired": 1, "eligible": 3}

    def replay(_layer, args):
        hidden = args[0]
        mask = torch.arange(hidden.shape[1]) == 1
        return (core.adaptive.steer_residual(hidden, direction, strength=.5, gate=mask),)

    handle = model.layers[0].register_forward_pre_hook(replay)
    try:
        _, expected = _capture_hidden(native, lambda: original(native, **dict(_request(), strength=0.)))
    finally:
        handle.remove()
    assert all(torch.equal(left, right) for left, right in zip(actual, expected, strict=True))


def test_clabs_stops_at_native_eos_and_reuses_shared_state(native, core):
    native.runtime.model.config.eos_token_id = 0
    direction = torch.nn.functional.normalize(torch.ones(16), dim=0)
    parameters = core.adaptive.CLABSParameters(
        mean=0., std=1., epsilon=1e-6, setpoint=.5,
        proportional_gain=.5, integral_gain=.25, beta=.25,
    )
    gate = PrefixGate(core=core, layer=1, method="clabs", clabs=parameters,
                      parameter=.5, seed=10, direction=direction, evidence_ref="fixture",
                      intervention="residual")
    result = generate(native, steering=gate, **_request())
    receipt = result.native["adaptive_steering"]
    assert result.native["new_token_ids"] == [0]
    step, = receipt["prefix_steps"]
    controller = core.adaptive.ActivationGate("clabs", .5, clabs=parameters, seed=10)
    expected = controller.step(step["alignment"])
    assert (step["fired"], step["probability"]) == (expected.fired, expected.probability)


def test_random_raw_gate_masks_and_replays_graph_outputs_by_position(native, core):
    host = native.runtime.model.layers[0].native.self_attn.q_proj

    def record(render):
        values = []
        handle = host.register_forward_hook(lambda _host, _args, value: values.append(value.clone()))
        try:
            return render(), values
        finally:
            handle.remove()

    rng = Random(1)
    fired = [rng.random() < .5 for _ in range(3)]
    policy = PrefixGate(core=core, layer=0, method="random", parameter=.5, seed=1,
                        evidence_ref="fixture", intervention="raw_adapter")
    _, off = record(lambda: original(native, **dict(_request(), strength=0.)))
    _, fixed = record(lambda: original(native, **_request()))
    result, actual = record(lambda: generate(native, steering=policy, **_request()))
    receipt = result.native["adaptive_steering"]
    assert [step["fired"] for step in receipt["prefix_steps"]] == fired
    assert any(not torch.equal(left[:, 1], right[:, 1]) for left, right in zip(off, fixed, strict=True))
    for i, observed in enumerate(actual):
        mask = torch.tensor([False] + fired[:i+1]).view(1, -1, 1)
        expected = torch.where(mask, fixed[i], off[i])
        assert torch.equal(observed, expected)


@pytest.mark.parametrize("rate,strength,prompt", [(0., .5, [1, 2]), (1., .5, [2]), (1., 0., [1, 2])])
def test_raw_endpoints_keep_exact_native_hidden_and_no_hook_leaks(native, core, rate, strength, prompt):
    plan = PrefixGate(core=core, layer=0, method="random", parameter=rate, seed=1,
                      evidence_ref="fixture", intervention="raw_adapter")
    args = dict(ids=prompt, max_new_tokens=3, strength=strength)
    expected, base = _capture_hidden(native, lambda: original(native, **dict(args, strength=strength*rate)))
    result, actual = _capture_hidden(native, lambda: generate(native, steering=plan, **args))
    assert result.text == expected.text and result.native["new_token_ids"] == expected.native["new_token_ids"]
    assert all(torch.equal(left, right) for left, right in zip(actual, base, strict=True))
    assert not native.runtime.model.layers[0]._forward_pre_hooks
    assert all(not branch.graph._forward_hooks and branch.multiplier == 0
               for branch in native.network.adapters.values())


def test_s_psr_fits_one_new_row_and_reapplies_history(native, core, monkeypatch):
    fitted = core.prompt.PromptSteering(16)
    with torch.no_grad():
        fitted.direction.weight.fill_(.25)
        fitted.probe.weight.zero_()
        fitted.probe.bias.fill_(1.)
    sizes = []
    coefficients = core.prompt.PromptSteering.coefficients

    def record(self, tensor):
        sizes.append(tensor.shape[1])
        return coefficients(self, tensor)

    monkeypatch.setattr(core.prompt.PromptSteering, "coefficients", record)
    plan = PrefixPromptSteering(core=core, layer=1, fitted=fitted, evidence_ref="fixture fit")
    result, actual = _capture_hidden(native, lambda: generate(native, steering=plan, **_request()))
    assert sizes == [1, 1, 1]
    assert result.native["adaptive_steering"]["effective_strengths"] == [.5, .5, .5]
    direction = fitted.direction.weight[0].detach()
    strength = direction.norm().item() * .5
    gate = PrefixGate(core=core, layer=1, method="random", parameter=1., seed=1,
                      direction=direction, evidence_ref="fixture", intervention="residual")
    _, expected = _capture_hidden(native, lambda: generate(native, steering=gate,
                                                            **dict(_request(), strength=strength)))
    assert all(torch.equal(left, right) for left, right in zip(actual, expected, strict=True))


def test_adaptive_raw_failure_restores_hook_and_multiplier_state(native, core, monkeypatch):
    hidden = native.runtime.hidden

    def interrupted(ids):
        hidden(ids)
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(native.runtime, "hidden", interrupted)
    gate = PrefixGate(core=core, layer=0, method="random", parameter=1., seed=1,
                      evidence_ref="fixture", intervention="raw_adapter")
    with pytest.raises(RuntimeError, match="fixture failure"):
        generate(native, steering=gate, **_request())
    assert not native.runtime.model.layers[0]._forward_pre_hooks
    assert all(not branch.graph._forward_hooks and branch.multiplier == 0
               for branch in native.network.adapters.values())


def test_facade_uses_original_owner_and_filters_adaptive_policy(native, core):
    called = []

    @contextmanager
    def observation(seed, request):
        called.append((seed, request))
        yield

    class Owner:
        network = native.network
        runtime = native.runtime

        def response(self):
            raise AssertionError("The hidden-response proxy cannot generate finished text")

    owner = Owner()
    binding = SimpleNamespace(network=owner.network, response=owner.response)
    session = SimpleNamespace(binding=binding, _observation=observation,
                              capabilities=SimpleNamespace(check=lambda *_a, **_kw: None),
                              _open=lambda: None, _preflight=lambda *_a: None)
    facade = TextInferenceSession(session)
    plan = PrefixGate(core=core, layer=0, method="random", parameter=1., seed=1,
                      evidence_ref="fixture", intervention="raw_adapter")
    request = _request()
    with pytest.raises(ValueError, match="either"):
        facade.render(steering=plan, step_gains=[1., 1., 1.], **request)
    result = facade.render(steering=plan, **request)
    assert result.native["adaptive_steering"]["method"] == "random"
    assert called == [(0, {"ids": request["ids"], "max_new_tokens": request["max_new_tokens"]})]


def test_verified_module_loader_leaves_original_core_and_rejects_changed_wheel(core, tmp_path):
    import particle_sliders
    import sys

    original_path = tuple(sys.path)
    original_module = sys.modules["particle_sliders"]
    original_rng = torch.random.get_rng_state().clone()
    assert load_steering_core(environ["RIPPLE_TEST_STEERING_WHEEL"]) is core
    assert tuple(sys.path) == original_path
    assert sys.modules["particle_sliders"] is original_module is particle_sliders
    assert torch.equal(torch.random.get_rng_state(), original_rng)
    bad = tmp_path / "mutated.whl"
    path = "particle_sliders/adaptive_steering.py"
    with ZipFile(environ["RIPPLE_TEST_STEERING_WHEEL"]) as valid, ZipFile(bad, "w") as archive:
        clean = valid.read(path)
        archive.writestr(path, b"#" + clean[1:])
    with pytest.raises(ValueError, match="checksum differs"):
        load_steering_core(bad)
