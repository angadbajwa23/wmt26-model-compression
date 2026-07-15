"""
Shared decoder-layer discovery used by sensitivity_sweep.py, solve_budget.py, and
prepare_model.py.

Mirrors submissions/awq/prepare_model.py::_build_awq_mappings's approach of
discovering exact module/tensor names at runtime rather than hardcoding a prefix
string (e.g. "language_model.model.layers") — stays correct even if the multimodal
wrapper's attribute nesting differs across transformers versions.
"""
import re
from typing import Iterable

SUBMODULE_SUFFIXES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

# Matches the "*.layers.N" segment common to both nn.Module dotted names
# (e.g. "language_model.model.layers.0.self_attn.q_proj") and safetensors tensor
# names (e.g. "language_model.model.layers.0.self_attn.q_proj.weight") — group(1)
# is the same layer-prefix regardless of which submodule under that layer matched.
_LAYER_SEG_RE = re.compile(r"^(.*\.layers\.(\d+))\.")


def discover_layer_prefixes(names: Iterable[str]) -> dict[int, str]:
    """Return {layer_idx: module_name_prefix} from any iterable of dotted names."""
    layers: dict[int, str] = {}
    for name in names:
        if "vision_tower" in name:
            continue
        m = _LAYER_SEG_RE.match(name)
        if m:
            layers[int(m.group(2))] = m.group(1)
    if not layers:
        raise RuntimeError("No decoder layers discovered (no '*.layers.N.*' names found)")
    return dict(sorted(layers.items()))


def discover_decoder_layers(model) -> dict[int, str]:
    return discover_layer_prefixes(name for name, _ in model.named_modules())


def layer_linear_names(prefix: str) -> list[str]:
    return [f"{prefix}.{suffix}" for suffix in SUBMODULE_SUFFIXES]
