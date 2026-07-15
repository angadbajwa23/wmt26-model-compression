#!/usr/bin/env python
"""
Stage 2 of the COMET-guided mixed-precision recipe: given per-layer INT4 sensitivity
scores (workdir/sensitivity.json, from sensitivity_sweep.py) and a size budget,
solve for the precision tier (BF16 / INT8 / INT4) of each decoder layer that
minimizes total expected COMET drop subject to staying under budget.

Since the Fast sweep only measures sensitivity at INT4 (not INT8), each tier's
expected COMET drop is estimated as:
    drop(layer, BF16) = 0                                  (unquantized, exact)
    drop(layer, INT8) = int8_recovery_factor * sensitivity  (default 0.5 -- INT8's
                         quantization step is half of INT4's, used as a linear
                         interpolation proxy; NOT directly measured)
    drop(layer, INT4) = sensitivity                          (measured)
This is a documented approximation -- see the new submission's README.

Solved exactly via a multi-choice knapsack DP: 48 layers x 3 tiers, size
discretized into small buckets, small enough to solve in milliseconds.

Output: workdir/tier_assignment.json
  {"bf16": [3, 7, ...], "int8": [...], "int4": [...], "estimated_size_bytes": ...}
"""
import argparse
import json
import struct
from pathlib import Path

from layer_utils import discover_layer_prefixes, layer_linear_names

GiB = 1024 ** 3
BF16_BYTES_PER_PARAM = 2
SCALE_ZERO_BYTES_PER_GROUP = 4  # fp16 scale + fp16 zero-point, per (output_channel, group)


def resolve_model_source(model_id: str, cache_dir: Path) -> Path:
    explicit_path = Path(model_id).expanduser()
    if explicit_path.exists():
        return explicit_path
    if explicit_path.is_absolute():
        raise FileNotFoundError(f"Model path does not exist: {explicit_path}")
    cached_model_dir = cache_dir.expanduser() / model_id
    if (cached_model_dir / "config.json").exists():
        return cached_model_dir
    raise FileNotFoundError(
        f"Model not found at {cached_model_dir}. Run sensitivity_sweep.py first "
        "(it downloads the base model), or point --cache-dir at an existing cache."
    )


def read_safetensors_shapes(model_dir: Path) -> dict[str, list[int]]:
    """Read tensor shapes from safetensors header(s) without loading weight data."""
    shapes: dict[str, list[int]] = {}
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = ["model.safetensors"]

    for shard in shard_files:
        path = model_dir / shard
        with open(path, "rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            shapes[name] = meta["shape"]
    return shapes


def layer_submodule_dims(shapes: dict[str, list[int]], prefix: str) -> list[tuple[int, int]]:
    """[(out_features, in_features), ...] for the 7 linear submodules of one layer."""
    dims = []
    for name in layer_linear_names(prefix):
        shape = shapes[f"{name}.weight"]
        out_features, in_features = shape[0], shape[1]
        dims.append((out_features, in_features))
    return dims


def tier_bytes(dims: list[tuple[int, int]], tier: str, group_size: int) -> int:
    total = 0
    for out_features, in_features in dims:
        params = out_features * in_features
        if tier == "bf16":
            total += params * BF16_BYTES_PER_PARAM
        elif tier == "int8":
            n_groups = -(-in_features // group_size)  # ceil
            total += params * 1 + out_features * n_groups * SCALE_ZERO_BYTES_PER_GROUP
        elif tier == "int4":
            n_groups = -(-in_features // group_size)
            total += params // 2 + out_features * n_groups * SCALE_ZERO_BYTES_PER_GROUP
        else:
            raise ValueError(tier)
    return total


def fixed_overhead_bytes(shapes: dict[str, list[int]], layer_prefixes: dict[int, str]) -> int:
    """Everything outside the 48 decoder layers' 7 linears: embeddings, norms,
    vision tower, multi-modal projector -- stays BF16/unquantized in every recipe
    (same ignore=["lm_head", "re:model.vision_tower.*"] convention as awq/gptq)."""
    layer_tensor_names = set()
    for prefix in layer_prefixes.values():
        for name in layer_linear_names(prefix):
            layer_tensor_names.add(f"{name}.weight")

    total = 0
    for name, shape in shapes.items():
        if name in layer_tensor_names:
            continue
        params = 1
        for d in shape:
            params *= d
        total += params * BF16_BYTES_PER_PARAM
    return total


def solve_knapsack(
    layers: list[int],
    sensitivity: dict[int, float],
    dims_by_layer: dict[int, list[tuple[int, int]]],
    group_size: int,
    int8_recovery_factor: float,
    budget_bytes: int,
    fixed_bytes: int,
    bucket_bytes: int = 10 * 1024 * 1024,  # 10MB discretization
) -> dict[int, str]:
    tiers = ["int4", "int8", "bf16"]  # cheapest first
    options: dict[int, list[tuple[str, int, float]]] = {}  # layer -> [(tier, size, drop)]
    for L in layers:
        dims = dims_by_layer[L]
        s = sensitivity.get(L, 0.0)
        opts = []
        for tier in tiers:
            size = tier_bytes(dims, tier, group_size)
            drop = {"int4": s, "int8": int8_recovery_factor * s, "bf16": 0.0}[tier]
            opts.append((tier, size, drop))
        options[L] = opts

    remaining_bytes = budget_bytes - fixed_bytes
    if remaining_bytes <= 0:
        raise ValueError(
            f"Fixed (unquantized) portion alone is {fixed_bytes / GiB:.2f} GiB, "
            f"already over the {budget_bytes / GiB:.2f} GiB budget."
        )
    n_buckets = remaining_bytes // bucket_bytes + 1

    NEG_INF = float("-inf")
    # dp[b] = max total (-drop) achievable using exactly buckets <= b, after processing layers so far
    dp = [NEG_INF] * (n_buckets + 1)
    dp[0] = 0.0
    choice: list[dict[int, int]] = []  # choice[i][b] = option index chosen for layer i at budget b

    for L in layers:
        opts = options[L]
        new_dp = [NEG_INF] * (n_buckets + 1)
        layer_choice = {}
        for b in range(n_buckets + 1):
            if dp[b] == NEG_INF:
                continue
            for oi, (tier, size, drop) in enumerate(opts):
                cost_buckets = -(-size // bucket_bytes)
                nb = b + cost_buckets
                if nb > n_buckets:
                    continue
                val = dp[b] - drop
                if val > new_dp[nb]:
                    new_dp[nb] = val
                    layer_choice[nb] = oi
        dp = new_dp
        choice.append(layer_choice)

    best_b = max(range(n_buckets + 1), key=lambda b: dp[b])
    if dp[best_b] == NEG_INF:
        raise RuntimeError("Knapsack DP found no feasible assignment (budget too small).")

    assignment: dict[int, str] = {}
    b = best_b
    for i in range(len(layers) - 1, -1, -1):
        L = layers[i]
        oi = choice[i][b]
        tier, size, _ = options[L][oi]
        assignment[L] = tier
        b -= -(-size // bucket_bytes)

    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve per-layer BF16/INT8/INT4 tier assignment under a size budget",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", default="google/gemma-3-12b-it")
    parser.add_argument("--cache-dir", type=Path,
                        default="/mnt/tg/data/projects/wmt26/model-compression/models")
    parser.add_argument("--sensitivity", type=Path,
                        default=Path(__file__).parent / "workdir/sensitivity.json")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "workdir/tier_assignment.json")
    parser.add_argument("--budget-gb", type=float, default=9.0,
                        help="Target total model size in GiB (binary GB, matches `du -h`)")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--int8-recovery-factor", type=float, default=0.5,
                        help="Fraction of measured INT4 COMET drop assumed recoverable at INT8 "
                             "(heuristic -- INT8 sensitivity is not directly measured by the Fast sweep)")
    args = parser.parse_args()

    source = resolve_model_source(args.model_id, args.cache_dir)
    shapes = read_safetensors_shapes(source)
    layer_prefixes = discover_layer_prefixes(shapes.keys())
    print(f"Discovered {len(layer_prefixes)} decoder layers from {source}")

    sens_data = json.loads(args.sensitivity.read_text())
    sensitivity = {int(k): v for k, v in sens_data["layers"].items()}
    missing = set(layer_prefixes) - set(sensitivity)
    if missing:
        raise ValueError(f"sensitivity.json is missing layers {sorted(missing)}; rerun sensitivity_sweep.py")

    dims_by_layer = {L: layer_submodule_dims(shapes, prefix) for L, prefix in layer_prefixes.items()}
    fixed_bytes = fixed_overhead_bytes(shapes, layer_prefixes)
    print(f"Fixed (unquantized) overhead: {fixed_bytes / GiB:.2f} GiB "
          f"(embeddings, norms, vision tower, multi-modal projector)")

    budget_bytes = int(args.budget_gb * GiB)
    layers = sorted(layer_prefixes)
    assignment = solve_knapsack(
        layers, sensitivity, dims_by_layer, args.group_size,
        args.int8_recovery_factor, budget_bytes, fixed_bytes,
    )

    tiers: dict[str, list[int]] = {"bf16": [], "int8": [], "int4": []}
    total_bytes = fixed_bytes
    for L in layers:
        tier = assignment[L]
        tiers[tier].append(L)
        total_bytes += tier_bytes(dims_by_layer[L], tier, args.group_size)

    print(f"Tier assignment: bf16={len(tiers['bf16'])} layers, int8={len(tiers['int8'])} layers, "
          f"int4={len(tiers['int4'])} layers")
    print(f"Estimated total size: {total_bytes / GiB:.2f} GiB (budget: {args.budget_gb:.2f} GiB)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump({**tiers, "estimated_size_bytes": total_bytes, "budget_gb": args.budget_gb}, fh, indent=2)
    print(f"Wrote tier assignment to {args.output}")


if __name__ == "__main__":
    main()
