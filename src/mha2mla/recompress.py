import torch

from patch_func import LowRankKVLinear, svd_low_rank_approx


def reconstruct_kv_weights(kv_proj):
    """
    Reconstruct full K and V weight matrices from a trained LowRankKVLinear module.

    For joint method:  k_full = up_k.weight @ down_kv.weight
                       v_full = up_v.weight @ down_kv.weight
    For split method:  k_full = up_k.weight @ down_k.weight
                       v_full = up_v.weight @ down_v.weight
    For only_key:      k_full = up_k.weight @ down_k.weight
                       v_full = up_v.weight  (full rank)
    For only_value:    k_full = up_k.weight  (full rank)
                       v_full = up_v.weight @ down_v.weight
    For none:          k_full = up_k.weight, v_full = up_v.weight

    Returns:
        (k_weight, k_bias, v_weight, v_bias) all in float32
    """
    has_joint = hasattr(kv_proj, "down_kv")
    has_down_k = hasattr(kv_proj, "down_k")
    has_down_v = hasattr(kv_proj, "down_v")

    # Reconstruct K weight
    if has_joint:
        # joint: k = up_k(down_kv(x)), so W_k = up_k.weight @ down_kv.weight
        k_weight = (
            kv_proj.up_k.weight.data.float() @ kv_proj.down_kv.weight.data.float()
        )
    elif has_down_k:
        # split or only_key: k = up_k(down_k(x))
        k_weight = (
            kv_proj.up_k.weight.data.float() @ kv_proj.down_k.weight.data.float()
        )
    else:
        # none or only_value: k = up_k(x), full rank
        k_weight = kv_proj.up_k.weight.data.float()

    # Reconstruct V weight
    if has_joint:
        v_weight = (
            kv_proj.up_v.weight.data.float() @ kv_proj.down_kv.weight.data.float()
        )
    elif has_down_v:
        v_weight = (
            kv_proj.up_v.weight.data.float() @ kv_proj.down_v.weight.data.float()
        )
    else:
        v_weight = kv_proj.up_v.weight.data.float()

    # Extract biases (None if not present)
    k_bias = (
        kv_proj.up_k.bias.data.float()
        if kv_proj.up_k.bias is not None
        else None
    )
    v_bias = (
        kv_proj.up_v.bias.data.float()
        if kv_proj.up_v.bias is not None
        else None
    )

    return k_weight, k_bias, v_weight, v_bias


def recompress_model(model, new_low_rank, num_kv_heads, svd_method):
    """
    Recompress all layers' kv_proj to a new (lower) rank.

    For each layer:
    1. Reconstruct full K_c and V weight matrices from the current kv_proj
    2. Re-decompose with SVD at the new rank
    3. Replace kv_proj with the new LowRankKVLinear

    Args:
        model: The patched model with kv_proj on each layer
        new_low_rank: Target rank per head for the new decomposition
        num_kv_heads: Number of KV heads
        svd_method: SVD method ('joint', 'split', 'only_key', 'only_value', 'none')
    """
    d_kv_mid = new_low_rank * num_kv_heads

    for layer_idx, layer in enumerate(model.model.layers):
        old_kv_proj = layer.self_attn.kv_proj

        # 1. Reconstruct full weight matrices
        k_weight, k_bias, v_weight, v_bias = reconstruct_kv_weights(old_kv_proj)

        # 2. Re-decompose with new rank
        new_kv_proj = svd_low_rank_approx(
            k_c_weight=k_weight,
            k_c_bias=k_bias,
            v_weight=v_weight,
            v_bias=v_bias,
            d_kv_mid=d_kv_mid,
            method=svd_method,
        )

        # 3. Replace kv_proj
        layer.self_attn.kv_proj = new_kv_proj

        print(f"Layer {layer_idx}: recompressed kv_proj to rank {new_low_rank}")
