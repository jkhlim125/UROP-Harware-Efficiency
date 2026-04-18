import argparse
import os
import random

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


def to_numpy(scores):
    if not HAVE_NUMPY:
        raise RuntimeError("numpy is required for this script")
    if HAVE_TORCH and isinstance(scores, torch.Tensor):
        return scores.detach().cpu().numpy()
    if isinstance(scores, np.ndarray):
        return scores
    return np.array(scores)


def load_tensor_like_from_file(path):
    if not HAVE_NUMPY:
        raise RuntimeError("numpy is required for loading and analyzing score tensors")
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        x = np.load(path, allow_pickle=True)
        if isinstance(x, np.ndarray) and x.shape == ():
            try:
                x = x.item()
            except Exception:
                pass
        if isinstance(x, dict):
            if "scores" in x:
                x = x["scores"]
        return to_numpy(x)

    if ext in [".pt", ".pth"]:
        if not HAVE_TORCH:
            raise RuntimeError("torch is needed to load .pt/.pth files")
        x = torch.load(path, map_location="cpu")
        if isinstance(x, dict):
            if "scores" in x:
                x = x["scores"]
            else:
                found_tensor = None
                for k, v in x.items():
                    if HAVE_TORCH and isinstance(v, torch.Tensor):
                        found_tensor = v
                        print("found tensor in dict under key:", k)
                        break
                if found_tensor is None:
                    raise RuntimeError("could not find a tensor in the .pt/.pth file")
                x = found_tensor
        return to_numpy(x)

    raise RuntimeError("unsupported file type, use .npy or .pt/.pth")


def load_scores_from_file(path):
    return load_tensor_like_from_file(path)


def load_valid_mask_from_file(path):
    mask = load_tensor_like_from_file(path)
    mask = to_numpy(mask).astype(bool)
    return mask


def make_debug_scores(shape=(16, 12, 6), seed=0):
    if not HAVE_NUMPY:
        raise RuntimeError("numpy is required for generating debug score tensors")
    rng = np.random.default_rng(seed)

    # Start with repeated / quantized values on purpose.
    repeated_pool = np.array([
        0.00, 0.00, 0.00,
        0.05, 0.05, 0.05,
        0.10, 0.10, 0.10,
        0.12, 0.12,
        0.15, 0.15,
        0.20, 0.20,
        0.25, 0.25,
        0.30, 0.30,
        0.35,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
    ], dtype=np.float32)

    total = int(np.prod(shape))
    scores = rng.choice(repeated_pool, size=total, replace=True).astype(np.float32)

    # Inject some mild structure so some outputs get pruned much harder.
    if len(shape) == 3:
        scores = scores.reshape(shape)
        for out_idx in range(shape[0]):
            if out_idx % 4 == 0:
                scores[out_idx] = scores[out_idx] * 0.30
            elif out_idx % 4 == 1:
                scores[out_idx] = scores[out_idx] * 0.60
            elif out_idx % 4 == 2:
                scores[out_idx] = scores[out_idx] * 1.00
            else:
                scores[out_idx] = scores[out_idx] * 1.40

        # Make some exact ties around values that often become thresholds.
        if shape[0] > 3:
            scores[0, :, :] = 0.10
            scores[1, :3, :] = 0.20
            scores[2, :2, :] = 0.30
            scores[3, :, :2] = 0.05
    else:
        scores = scores.reshape(shape)

    return scores.astype(np.float32)


def flatten_scores(scores_np):
    return scores_np.reshape(-1).astype(np.float64)


def flatten_valid_mask(valid_mask, shape):
    if valid_mask is None:
        return None
    valid_mask_np = to_numpy(valid_mask).astype(bool)
    if valid_mask_np.shape != shape:
        raise RuntimeError(f"valid_mask shape {valid_mask_np.shape} does not match scores shape {shape}")
    return valid_mask_np.reshape(-1)


def compute_target_prune_count(total_valid, sparsity):
    return int(total_valid * sparsity)


def quantile_threshold(flat_scores, sparsity, valid_flat_mask=None):
    if valid_flat_mask is None:
        valid_vals = flat_scores
    else:
        valid_vals = flat_scores[valid_flat_mask]

    if valid_vals.size == 0:
        raise RuntimeError("no valid elements found for threshold calculation")

    return float(np.quantile(valid_vals, sparsity))


def threshold_partition(flat, threshold, atol=1e-8, valid_flat_mask=None):
    is_tie = np.isclose(flat, threshold, atol=atol)
    below = (flat < threshold) & (~is_tie)
    above = (flat > threshold) & (~is_tie)

    if valid_flat_mask is not None:
        below = below & valid_flat_mask
        is_tie = is_tie & valid_flat_mask
        above = above & valid_flat_mask

    return below, is_tie, above


def improved_keep_mask(scores_np, threshold, target_prune_count, seed=0, random_ties=False,
                       valid_mask=None, atol=1e-8):
    flat = flatten_scores(scores_np)
    total = flat.size

    keep = np.ones(total, dtype=bool)
    valid_flat_mask = flatten_valid_mask(valid_mask, scores_np.shape)

    below, equal, _ = threshold_partition(flat, threshold, atol=atol, valid_flat_mask=valid_flat_mask)

    keep[below] = False

    num_below = int(below.sum())
    need_more = target_prune_count - num_below

    eq_idx = np.where(equal)[0]

    if need_more > 0 and len(eq_idx) > 0:
        if random_ties:
            random.seed(seed)
            eq_idx = eq_idx.copy()
            random.shuffle(eq_idx)
        prune_eq_idx = eq_idx[:need_more]
        keep[prune_eq_idx] = False

    return keep.reshape(scores_np.shape)


def prune_ratio_from_keep(keep_mask):
    pruned = (~keep_mask).astype(np.float32)
    return float(pruned.mean())


def summarize_basic_stats(flat):
    print("\n" + "=" * 90)
    print("PART 5: BASIC SCORE STATS")
    print("=" * 90)
    print(f"num elements: {flat.size}")
    print(f"min:  {flat.min():.6f}")
    print(f"max:  {flat.max():.6f}")
    print(f"mean: {flat.mean():.6f}")
    print(f"std:  {flat.std():.6f}")

    print("\nsome sample values:")
    sample_n = min(20, flat.size)
    print(flat[:sample_n])

    unique_vals, counts = np.unique(flat, return_counts=True)
    top_order = np.argsort(counts)[::-1]

    print("\nmost repeated exact values:")
    for i in top_order[:10]:
        print(f"  value={unique_vals[i]:.6f}, count={counts[i]}")

    duplicate_elements = int(counts[counts > 1].sum())
    duplicate_ratio = duplicate_elements / float(flat.size)
    print(f"\nexact-duplicate element ratio: {100.0 * duplicate_ratio:.2f}%")
    if duplicate_ratio > 0.10:
        print("WARNING: many values are exactly repeated -> quantization / discretization effect likely")
    if len(unique_vals) < max(20, flat.size * 0.05):
        print("WARNING: unique value count is pretty small compared with tensor size")


def analyze_naive(flat, scores_np, sparsity, valid_mask=None, atol=1e-8):
    print("\n" + "=" * 90)
    print("PART 1: NAIVE GLOBAL THRESHOLD LOGIC")
    print("=" * 90)

    total = flat.size
    valid_flat_mask = flatten_valid_mask(valid_mask, scores_np.shape)
    total_valid = int(valid_flat_mask.sum()) if valid_flat_mask is not None else total
    target_prune_count = compute_target_prune_count(total_valid, sparsity)
    threshold = quantile_threshold(flat, sparsity, valid_flat_mask=valid_flat_mask)
    keep_naive = np.ones_like(scores_np, dtype=bool)

    below, equal, above = threshold_partition(
        flat,
        threshold,
        atol=atol,
        valid_flat_mask=valid_flat_mask,
    )
    keep_naive_flat = keep_naive.reshape(-1)
    keep_naive_flat[below] = False
    keep_naive_flat[above | equal] = True

    actual_keep_count = total_valid - int(below.sum())
    actual_prune_count = int(below.sum())
    actual_keep_ratio = actual_keep_count / float(total_valid) if total_valid > 0 else 0.0
    actual_prune_ratio = actual_prune_count / float(total_valid) if total_valid > 0 else 0.0
    diff = actual_prune_count - target_prune_count

    print(f"target sparsity: {sparsity:.4f}")
    print(f"threshold from quantile: {threshold:.8f}")
    print(f"total tensor elements: {total}")
    print(f"valid elements used for global pruning: {total_valid}")
    print("naive rule: values tied at threshold are kept, so only values strictly below threshold are pruned")
    print(f"target prune count: {target_prune_count}")
    print(f"naive global pruning prune count: {actual_prune_count}")
    print(f"naive global pruning prune ratio: {actual_prune_ratio:.4f}")
    print(f"naive global pruning keep ratio: {actual_keep_ratio:.4f}")
    print(f"target prune-count error: {diff:+d}")

    if actual_prune_count == target_prune_count:
        print("target sparsity satisfied exactly by naive logic")
    else:
        print("WARNING: naive thresholding does NOT hit the requested prune count")

    return {
        "threshold": threshold,
        "target_prune_count": target_prune_count,
        "keep_naive": keep_naive,
        "actual_prune_count": actual_prune_count,
        "actual_keep_count": actual_keep_count,
        "total_valid": total_valid,
    }


def analyze_threshold_equality(flat, threshold, valid_mask=None, scores_shape=None, atol=1e-8):
    print("\n" + "=" * 90)
    print("PART 2: EQUALITY-AT-THRESHOLD PROBLEM")
    print("=" * 90)

    valid_flat_mask = None
    if valid_mask is not None:
        if scores_shape is None:
            raise RuntimeError("scores_shape is needed when valid_mask is provided")
        valid_flat_mask = flatten_valid_mask(valid_mask, scores_shape)
    below_mask, equal_mask, above_mask = threshold_partition(flat, threshold, atol=atol, valid_flat_mask=valid_flat_mask)
    below = int(below_mask.sum())
    equal = int(equal_mask.sum())
    above = int(above_mask.sum())

    print(f"threshold: {threshold:.8f}")
    if valid_mask is None:
        print(f"elements below threshold: {below}")
        print(f"elements tied at threshold: {equal}")
        print(f"elements above threshold: {above}")
    else:
        print(f"valid elements below threshold: {below}")
        print(f"valid elements tied at threshold: {equal}")
        print(f"valid elements above threshold: {above}")

    if equal > 0:
        print(f"there are {equal} threshold ties (using np.isclose)")
    else:
        print("no exact equality at threshold in this tensor")

    valid_total = int(valid_flat_mask.sum()) if valid_flat_mask is not None else flat.size
    eq_ratio = equal / float(valid_total) if valid_total > 0 else 0.0
    if equal >= 5:
        print("WARNING: many elements tied at threshold -> naive global pruning may be unstable")
    if eq_ratio > 0.05:
        print(f"WARNING: threshold-tie ratio is large ({100.0 * eq_ratio:.2f}%)")

    return {
        "below": below,
        "equal": equal,
        "above": above,
    }


def analyze_improved(scores_np, threshold, target_prune_count, naive_prune_count, seed=0, random_ties=False,
                     valid_mask=None, atol=1e-8):
    print("\n" + "=" * 90)
    print("PART 3: IMPROVED EXACT-COUNT LOGIC")
    print("=" * 90)

    keep_improved = improved_keep_mask(
        scores_np,
        threshold,
        target_prune_count=target_prune_count,
        seed=seed,
        random_ties=random_ties,
        valid_mask=valid_mask,
        atol=atol,
    )

    valid_flat_mask = flatten_valid_mask(valid_mask, scores_np.shape)
    improved_pruned_flat = (~keep_improved).reshape(-1)
    if valid_flat_mask is not None:
        improved_prune_count = int((improved_pruned_flat & valid_flat_mask).sum())
    else:
        improved_prune_count = int(improved_pruned_flat.sum())
    diff_improved = improved_prune_count - target_prune_count

    print(f"naive global pruning count:        {naive_prune_count}")
    print(f"improved tie-broken pruning count: {improved_prune_count}")
    print(f"target prune count:                {target_prune_count}")
    print(f"naive target error:                {naive_prune_count - target_prune_count:+d}")
    print(f"improved target error:             {diff_improved:+d}")

    if diff_improved == 0:
        print("improved tie-breaking matches the requested prune count exactly")
    else:
        print("WARNING: improved tie-breaking still missed the exact target")
        print("this usually means the bookkeeping is off or the target is impossible somehow")

    return {
        "keep_improved": keep_improved,
        "improved_prune_count": improved_prune_count,
    }


def print_bad_channels(name, ratios, mean_ratio):
    for i, ratio in enumerate(ratios):
        if mean_ratio > 0 and ratio > 2.0 * mean_ratio:
            print(f"WARNING: {name} {i} prune ratio is very high: {ratio:.4f} vs mean {mean_ratio:.4f}")
        if mean_ratio > 0 and ratio < 0.5 * mean_ratio:
            print(f"WARNING: {name} {i} prune ratio is very low:  {ratio:.4f} vs mean {mean_ratio:.4f}")


def analyze_structure(scores_np, keep_naive, keep_improved, valid_mask=None):
    print("\n" + "=" * 90)
    print("PART 4: PRUNING CONSISTENCY ACROSS STRUCTURE")
    print("=" * 90)

    if scores_np.ndim == 1:
        print("1D tensor only, so there is no output/layer structure to compare")
        return None

    if scores_np.ndim == 2:
        valid_by_out = None if valid_mask is None else valid_mask.reshape(scores_np.shape[0], -1).astype(bool)
        pruned_naive = (~keep_naive).reshape(scores_np.shape[0], -1)
        pruned_improved = (~keep_improved).reshape(scores_np.shape[0], -1)
        if valid_by_out is None:
            by_out_naive = pruned_naive.mean(axis=1)
            by_out_improved = pruned_improved.mean(axis=1)
        else:
            by_out_naive = []
            by_out_improved = []
            for i in range(scores_np.shape[0]):
                valid_i = valid_by_out[i]
                if valid_i.sum() == 0:
                    by_out_naive.append(0.0)
                    by_out_improved.append(0.0)
                else:
                    by_out_naive.append(pruned_naive[i][valid_i].mean())
                    by_out_improved.append(pruned_improved[i][valid_i].mean())
            by_out_naive = np.array(by_out_naive)
            by_out_improved = np.array(by_out_improved)
        label = "row"
    else:
        valid_by_out = None if valid_mask is None else valid_mask.reshape(scores_np.shape[0], -1).astype(bool)
        pruned_naive = (~keep_naive).reshape(scores_np.shape[0], -1)
        pruned_improved = (~keep_improved).reshape(scores_np.shape[0], -1)
        if valid_by_out is None:
            by_out_naive = pruned_naive.mean(axis=1)
            by_out_improved = pruned_improved.mean(axis=1)
        else:
            by_out_naive = []
            by_out_improved = []
            for i in range(scores_np.shape[0]):
                valid_i = valid_by_out[i]
                if valid_i.sum() == 0:
                    by_out_naive.append(0.0)
                    by_out_improved.append(0.0)
                else:
                    by_out_naive.append(pruned_naive[i][valid_i].mean())
                    by_out_improved.append(pruned_improved[i][valid_i].mean())
            by_out_naive = np.array(by_out_naive)
            by_out_improved = np.array(by_out_improved)
        label = "output channel"

    print("naive per-output prune ratio stats:")
    print(f"  min:  {by_out_naive.min():.4f}")
    print(f"  max:  {by_out_naive.max():.4f}")
    print(f"  mean: {by_out_naive.mean():.4f}")
    print(f"  std:  {by_out_naive.std():.4f}")

    print("improved per-output prune ratio stats:")
    print(f"  min:  {by_out_improved.min():.4f}")
    print(f"  max:  {by_out_improved.max():.4f}")
    print(f"  mean: {by_out_improved.mean():.4f}")
    print(f"  std:  {by_out_improved.std():.4f}")

    print("\nfirst few per-structure ratios (naive -> improved):")
    for i in range(min(12, len(by_out_naive))):
        print(f"  {label} {i}: {by_out_naive[i]:.4f} -> {by_out_improved[i]:.4f}")

    print()
    print_bad_channels(label, by_out_naive, float(by_out_naive.mean()))

    if scores_np.ndim == 3:
        per_lut_naive = (~keep_naive).astype(np.float32)
        per_lut_improved = (~keep_improved).astype(np.float32)
        if valid_mask is not None:
            valid_float = valid_mask.astype(np.float32)
            denom = np.maximum(valid_float.sum(axis=2), 1.0)
            per_lut_naive = (per_lut_naive * valid_float).sum(axis=2) / denom
            per_lut_improved = (per_lut_improved * valid_float).sum(axis=2) / denom
        else:
            per_lut_naive = per_lut_naive.mean(axis=2)
            per_lut_improved = per_lut_improved.mean(axis=2)

        print("\nextra 3D view: per-[out, L] prune ratio")
        print(f"  naive min/max/mean:    {per_lut_naive.min():.4f} / {per_lut_naive.max():.4f} / {per_lut_naive.mean():.4f}")
        print(f"  improved min/max/mean: {per_lut_improved.min():.4f} / {per_lut_improved.max():.4f} / {per_lut_improved.mean():.4f}")

        very_dead = np.argwhere(per_lut_naive > 0.95)
        very_alive = np.argwhere(per_lut_naive < 0.05)
        if len(very_dead) > 0:
            print(f"WARNING: {len(very_dead)} [out, L] blocks are almost completely pruned in naive mask")
            print("  examples:", very_dead[:10])
        if len(very_alive) > 0:
            print(f"WARNING: {len(very_alive)} [out, L] blocks are almost untouched in naive mask")
            print("  examples:", very_alive[:10])

    return {
        "by_out_naive": by_out_naive,
        "by_out_improved": by_out_improved,
    }


def plot_everything(flat, threshold, target_prune_count, naive_prune_count, improved_prune_count,
                    structure_info=None, out_prefix="pruning_debug"):
    if not HAVE_MPL:
        print("\nmatplotlib not installed, skipping plots")
        return

    print("\n" + "=" * 90)
    print("PART 6: VISUALIZATION")
    print("=" * 90)

    fig1 = plt.figure(figsize=(8, 5))
    plt.hist(flat, bins=40, alpha=0.85, color="steelblue", edgecolor="black")
    plt.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"threshold={threshold:.4f}")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Score Histogram for Global Pruning Debug")
    plt.legend()
    plt.tight_layout()
    hist_path = out_prefix + "_hist.png"
    plt.savefig(hist_path, dpi=150)
    plt.close(fig1)
    print("saved:", hist_path)

    fig2 = plt.figure(figsize=(7, 5))
    names = ["target", "naive", "improved"]
    vals = [target_prune_count, naive_prune_count, improved_prune_count]
    colors = ["gray", "tomato", "seagreen"]
    plt.bar(names, vals, color=colors)
    plt.ylabel("Pruned Valid Element Count")
    plt.title("Naive Global Pruning vs Improved Tie-Broken Pruning")
    plt.tight_layout()
    count_path = out_prefix + "_counts.png"
    plt.savefig(count_path, dpi=150)
    plt.close(fig2)
    print("saved:", count_path)

    if structure_info is not None:
        fig3 = plt.figure(figsize=(10, 4.5))
        x = np.arange(len(structure_info["by_out_naive"]))
        width = 0.38
        plt.bar(x - width / 2, structure_info["by_out_naive"], width=width, label="naive")
        plt.bar(x + width / 2, structure_info["by_out_improved"], width=width, label="improved")
        plt.xlabel("Output / Row Index")
        plt.ylabel("Prune Ratio")
        plt.title("Per-Output Prune Ratio Imbalance")
        plt.legend()
        plt.tight_layout()
        struct_path = out_prefix + "_per_output.png"
        plt.savefig(struct_path, dpi=150)
        plt.close(fig3)
        print("saved:", struct_path)


def run_analysis(scores, sparsity, seed=0, random_ties=False, out_prefix="pruning_debug",
                 valid_mask=None, atol=1e-8):
    summary_lines = []
    scores_np = to_numpy(scores).astype(np.float32)
    valid_mask_np = None if valid_mask is None else to_numpy(valid_mask).astype(bool)

    if scores_np.ndim not in [1, 2, 3]:
        raise RuntimeError(f"expected 1D/2D/3D scores, got shape {scores_np.shape}")
    if not (0.0 <= sparsity <= 1.0):
        raise RuntimeError("sparsity should be between 0 and 1")
    if valid_mask_np is not None and valid_mask_np.shape != scores_np.shape:
        raise RuntimeError(f"valid_mask shape {valid_mask_np.shape} does not match scores shape {scores_np.shape}")

    flat = flatten_scores(scores_np)
    valid_flat_mask = flatten_valid_mask(valid_mask_np, scores_np.shape)
    valid_flat = flat if valid_flat_mask is None else flat[valid_flat_mask]

    print("\n" + "#" * 90)
    print("CHECK PRUNING CONSISTENCY")
    print("#" * 90)
    print("shape:", scores_np.shape)
    print("dtype:", scores_np.dtype)
    print("sparsity target:", sparsity)
    if valid_mask_np is None:
        print("valid mask: none (all tensor elements are treated as valid)")
    else:
        print("valid mask: provided")
        print("valid elements:", int(valid_mask_np.sum()))
    print("tie-break mode:", "random among threshold-equal elements" if random_ties else "first N threshold-equal elements")

    summary_lines.append("CHECK PRUNING CONSISTENCY")
    summary_lines.append(f"tensor shape: {scores_np.shape}")
    summary_lines.append(f"total tensor elements: {flat.size}")
    summary_lines.append(f"sparsity target: {sparsity}")
    summary_lines.append(f"valid elements used for pruning: {valid_flat.size}")

    summarize_basic_stats(flat)

    naive_info = analyze_naive(flat, scores_np, sparsity, valid_mask=valid_mask_np, atol=atol)
    tie_info = analyze_threshold_equality(
        flat,
        naive_info["threshold"],
        valid_mask=valid_mask_np,
        scores_shape=scores_np.shape,
        atol=atol,
    )
    improved_info = analyze_improved(
        scores_np,
        naive_info["threshold"],
        naive_info["target_prune_count"],
        naive_info["actual_prune_count"],
        seed=seed,
        random_ties=random_ties,
        valid_mask=valid_mask_np,
        atol=atol,
    )
    structure_info = analyze_structure(scores_np, naive_info["keep_naive"], improved_info["keep_improved"], valid_mask=valid_mask_np)

    print("\n" + "=" * 90)
    print("FINAL TAKEAWAY")
    print("=" * 90)
    naive_error = naive_info["actual_prune_count"] - naive_info["target_prune_count"]
    improved_error = improved_info["improved_prune_count"] - naive_info["target_prune_count"]
    print(f"naive prune-count error:    {naive_error:+d}")
    print(f"improved prune-count error: {improved_error:+d}")
    if naive_error != 0:
        print("This is the exact kind of bug this script is supposed to make obvious.")
        print("Naive quantile + strict '>' can miss the intended global sparsity when ties are present.")
    else:
        print("Naive prune count happened to match here, but still inspect threshold ties and structural imbalance.")

    if structure_info is not None:
        spread = float(structure_info["by_out_naive"].max() - structure_info["by_out_naive"].min())
        print(f"naive per-structure prune spread: {spread:.4f}")
        if spread > 0.50:
            print("WARNING: pruning is very uneven across outputs / layers")
    else:
        spread = None

    summary_lines.append(f"threshold value: {naive_info['threshold']:.8f}")
    summary_lines.append(f"target prune count: {naive_info['target_prune_count']}")
    summary_lines.append(f"naive prune count: {naive_info['actual_prune_count']}")
    summary_lines.append(f"improved prune count: {improved_info['improved_prune_count']}")
    summary_lines.append(f"threshold tie count: {tie_info['equal']}")
    summary_lines.append(f"naive prune-count error: {naive_error:+d}")
    summary_lines.append(f"improved prune-count error: {improved_error:+d}")
    if spread is not None:
        summary_lines.append(f"per-output prune spread: {spread:.6f}")

    plot_everything(
        flat=valid_flat,
        threshold=naive_info["threshold"],
        target_prune_count=naive_info["target_prune_count"],
        naive_prune_count=naive_info["actual_prune_count"],
        improved_prune_count=improved_info["improved_prune_count"],
        structure_info=structure_info,
        out_prefix=out_prefix,
    )

    summary_path = out_prefix + "_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        for line in summary_lines:
            f.write(line + "\n")
    print("saved summary:", summary_path)


def parse_shape(text):
    pieces = text.split(",")
    shape = tuple(int(x.strip()) for x in pieces if x.strip())
    if len(shape) == 0:
        raise RuntimeError("bad shape string")
    return shape


def main():
    parser = argparse.ArgumentParser(description="Debug global pruning threshold consistency")
    parser.add_argument("--file", type=str, default="", help="optional .npy or .pt/.pth file containing scores")
    parser.add_argument("--valid-mask-file", type=str, default="", help="optional .npy or .pt/.pth file containing valid mask")
    parser.add_argument("--sparsity", type=float, default=0.60, help="target sparsity")
    parser.add_argument("--shape", type=str, default="16,12,6", help="shape for generated debug tensor if no file")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--random-ties", action="store_true", help="randomly choose equal-threshold elements in improved simulation")
    parser.add_argument("--out-prefix", type=str, default="pruning_debug", help="prefix for saved plot files")
    parser.add_argument("--tie-atol", type=float, default=1e-8, help="atol used for threshold tie detection")
    args = parser.parse_args()

    if not HAVE_NUMPY:
        raise RuntimeError(
            "This script needs numpy. In a normal research env that usually comes with torch too."
        )

    if args.file:
        print("loading scores from:", args.file)
        scores = load_scores_from_file(args.file)
    else:
        shape = parse_shape(args.shape)
        print("no file given, generating debug tensor with repeated values")
        scores = make_debug_scores(shape=shape, seed=args.seed)

    valid_mask = None
    if args.valid_mask_file:
        print("loading valid mask from:", args.valid_mask_file)
        valid_mask = load_valid_mask_from_file(args.valid_mask_file)

    run_analysis(
        scores=scores,
        sparsity=args.sparsity,
        seed=args.seed,
        random_ties=args.random_ties,
        out_prefix=args.out_prefix,
        valid_mask=valid_mask,
        atol=args.tie_atol,
    )


if __name__ == "__main__":
    main()
