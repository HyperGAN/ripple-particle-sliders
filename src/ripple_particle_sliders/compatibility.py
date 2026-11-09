"""Offline, allocation-free checkpoint recipe inspection; no training admission."""
import argparse
import fnmatch
import json
from pathlib import Path

from .backend import MODEL_ID, REVISION, VENDOR

WEIGHT_BYTES = 57_213_636_640


def inspect_config(config):
    expected = dict(model_type='ricursive', architectures=['RicursiveForCausalLM'],
                    hidden_size=5120, intermediate_size=40960, num_hidden_layers=80,
                    num_attention_heads=80, num_key_value_heads=8, head_dim=128,
                    vocab_size=100352, use_gated_attention=True, use_valres=True,
                    gate_full_precision=True, ffn_clamp_threshold=20.0,
                    sliding_window=1024, tie_word_embeddings=False)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('not the pinned Ripple-65B NVFP4 architecture')
    pattern = ['full_attention' if i % 5 == 4 else 'sliding_attention' for i in range(80)]
    if config.get('layer_types') != pattern:
        raise ValueError('Ripple hybrid local/global layer layout differs')
    quant = config.get('quantization_config', {})
    if quant.get('quant_method') != 'modelopt' or quant.get('quant_algo') != 'NVFP4':
        raise ValueError('official ModelOpt NVFP4 required; no BF16 fallback')
    groups = quant.get('config_groups', {})
    if len(groups) != 1:
        raise ValueError('one static W4A4 group required')
    group = next(iter(groups.values()))
    if group.get('targets') != ['Linear']:
        raise ValueError('NVFP4 recipe must target Linear')
    for kind in ('weights', 'input_activations'):
        if group.get(kind) != dict(dynamic=False, num_bits=4, type='float', group_size=16):
            raise ValueError('static NVFP4 weights and activations with group size 16 required')
    if quant.get('kv_cache_quant_algo') is not None or quant.get('kv_cache_scheme') is not None:
        raise ValueError('BF16 KV cache required')
    if quant.get('quantized_layers'):
        raise ValueError('per-layer quantization overrides are unqualified')
    ignored = quant.get('ignore', [])
    if not isinstance(ignored, list) or any(not isinstance(x, str) for x in ignored):
        raise ValueError('quantization exclusion list required')
    excluded = lambda name: any(fnmatch.fnmatchcase(name, p) for p in ignored)
    bf16 = ['lm_head', 'model.embed_tokens']
    targets = {}
    for i in range(80):
        stem = f'model.layers.{i}'
        for leaf in ('q', 'k', 'v', 'o', 'g'):
            name = f'{stem}.self_attn.{leaf}_proj'
            bf16.append(name)
            if leaf != 'g':
                targets[name] = {'q': [10240, 5120], 'k': [1024, 5120],
                                 'v': [1024, 5120], 'o': [5120, 10240]}[leaf]
        for leaf in ('gate', 'up', 'down'):
            if excluded(f'{stem}.mlp.{leaf}_proj'):
                raise ValueError('MLP must remain NVFP4')
    if not all(excluded(name) for name in bf16):
        raise ValueError('attention, embeddings and LM head must remain BF16')
    return dict(model=MODEL_ID, revision=REVISION, checkpoint_bytes=WEIGHT_BYTES,
                architecture='dense causal text LLM; 64 local + 16 global attention blocks',
                native_response_shape=[1, 5120], attention_targets=targets,
                quantization='static ModelOpt W4A4, group 16, MLP only',
                official_spark_compute_capability=[12, 1],
                nvfp4_training_qualified=False,
                blockers=['Full-model streamed loader not qualified; one real packed projection has passed CPU VJPs',
                          'Official Spark runtime is inference-only and requires SM121',
                          'Local RTX A6000 GPUs are SM86; packed checkpoint exceeds one 48 GiB card',
                          'A6000 dequantized BF16 reference GEMMs are a candidate; GPU tests and scheduler admission pending'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=VENDOR / 'config.json')
    args = parser.parse_args()
    report = inspect_config(json.loads(args.config.read_text()))
    report['attention_target_count'] = len(report.pop('attention_targets'))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
