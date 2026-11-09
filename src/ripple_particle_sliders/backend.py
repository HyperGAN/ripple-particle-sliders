"""Pinned official Ripple architecture; bounded CPU qualification only.

No checkpoint downloader, inferred quantization conversion, or BF16 fallback.
"""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import torch
from torch import nn

MODEL_ID = 'ricursiveAI/Ripple-65B-Preview-NVFP4'
REVISION = '99eea4477a001a41b86914e41feb4b2ed86105e5'
VENDOR = Path(__file__).parent / 'vendor'


def upstream_identity():
    return dict(model=MODEL_ID, revision=REVISION, quantization='ModelOpt NVFP4 MLP / BF16 attention',
                files={name: hashlib.sha256((VENDOR / name).read_bytes()).hexdigest()
                       for name in ('configuration_ricursive.py', 'modeling_ricursive.py', 'config.json')})


def make_fixture(width=32):
    """Run unmodified upstream code with both local and global attention.

    Random FP32 parameters exercise architecture, never emulate NVFP4 arithmetic.
    """
    if type(width) is not int or width not in (16, 32):
        raise ValueError('Ripple fixture width must be 16 or 32')
    if version('transformers') != '5.15.0':
        raise ValueError('Ripple architecture fixture requires transformers==5.15.0')
    from .vendor.configuration_ricursive import RicursiveConfig
    from .vendor.modeling_ricursive import RicursiveForCausalLM
    config = RicursiveConfig(
        vocab_size=256, hidden_size=width, intermediate_size=width * 2,
        num_hidden_layers=5, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=128, sliding_window=4,
        layer_types=['sliding_attention'] * 4 + ['full_attention'],
        use_gated_attention=True, gate_full_precision=True,
        use_valres=True, ffn_clamp_threshold=20.0,
        embed_scale=width ** .5, use_cache=False,
        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    return RicursiveForCausalLM(config).eval().requires_grad_(False)


def text_attention_targets(model):
    """Only Q/K/V/O in attention, which official NVFP4 excludes from quantization.

    Leave gates and MLP projections untouched. Reject altered/duplicate forwards
    before changing any native state; these are CPU fixture qualifications only.
    """
    from .vendor.modeling_ricursive import RicursiveAttention
    targets, seen = {}, set()
    for name, module in model.named_modules():
        if not isinstance(module, RicursiveAttention):
            continue
        for leaf in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            child = getattr(module, leaf, None)
            if (type(child) is not nn.Linear or id(child) in seen
                    or getattr(child.forward, '__func__', None) is not nn.Linear.forward):
                raise ValueError('expected unique, unwrapped Ripple attention Linear')
            seen.add(id(child))
            targets[f'{name}.{leaf}'] = child
    if not targets:
        raise ValueError('no Ripple text attention targets')
    return targets


class RippleBackend:
    def __init__(self, *, fixture_width=32):
        self.model = make_fixture(fixture_width)

    def hidden(self, ids):
        if (not isinstance(ids, (tuple, list)) or not 1 <= len(ids) <= 128
                or any(type(v) is not int or not 0 <= v < 256 for v in ids)):
            raise ValueError('fixture needs 1..128 token IDs in [0,256)')
        inputs = torch.tensor([ids], dtype=torch.long, device='cpu')
        return self.model.model(input_ids=inputs, use_cache=False,
                                return_dict=True).last_hidden_state
