"""
Verify Assumption 2 for Language Models (GPT-2, LLaMA)

This script tests whether gradient-based component importance (s_i) 
correlates with actual ablation effects (Δ(i)) for language models.

Usage:
    python verify_ass2_lm.py \
        --model_path wandb:goodfire/spd/runs/abc123xyz \
        --text "Once upon a time" \
        --tokenizer_name SimpleStories/test-SimpleStories-gpt2-1.25M \
        --max_components 100 \
        --save_dir experiments/assumption2
"""

import random
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from scipy.stats import pearsonr, spearmanr

from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import make_mask_infos
from spd.configs import SamplingType


def build_mask_infos_with_ablation(
    ci_outputs,
    ablate_module: str,
    ablate_idx: int,
):
    """
    Build mask_infos with one component ablated.
    
    Args:
        ci_outputs: Causal importance outputs from model
        ablate_module: Which module to ablate a component from
        ablate_idx: Index of component to ablate
    """
    component_masks = {}
    
    for module_name, mask_vals in ci_outputs.lower_leaky.items():
        comp_mask = mask_vals.clone()
        
        # Ablate the specific component
        # Handle both 2D (batch, C) and 3D (batch, seq_len, C) tensors
        if module_name == ablate_module:
            if comp_mask.ndim == 2:
                # Shape: (batch, C)
                comp_mask[:, ablate_idx] = 0.0
            else:
                # Shape: (batch, seq_len, C) or similar
                comp_mask[..., ablate_idx] = 0.0
        
        component_masks[module_name] = comp_mask
    
    # Use make_mask_infos helper (standard pattern in codebase)
    # routing_masks="all" routes all positions to components
    return make_mask_infos(
        component_masks=component_masks,
        routing_masks="all",
        weight_deltas_and_masks=None
    )


def compute_gradient(
    model: ComponentModel,
    input_ids: torch.Tensor,
    ci_outputs,
    module_name: str,
    comp_idx: int,
    loss_fn: Callable,
    device: torch.device
) -> float:
    """
    Compute gradient of loss w.r.t. a specific component mask.
    
    Returns:
        Gradient value (averaged over all leading dimensions)
    """
    component_masks = {}
    
    for mod_name, mask_vals in ci_outputs.lower_leaky.items():
        comp_mask = mask_vals.clone().detach()
        
        # Enable gradient only for the module we're interested in
        if mod_name == module_name:
            comp_mask.requires_grad_(True)
        
        component_masks[mod_name] = comp_mask
    
    # Use make_mask_infos helper (standard pattern in codebase)
    # routing_masks="all" routes all positions to components
    mask_infos = make_mask_infos(
        component_masks=component_masks,
        routing_masks="all",
        weight_deltas_and_masks=None
    )
    
    # Forward with masked components
    out = model(input_ids, mask_infos=mask_infos)
    loss = loss_fn(out)
    
    # Get gradient
    comp_mask = mask_infos[module_name].component_mask
    
    if not comp_mask.requires_grad:
        print(f"Warning: component_mask for {module_name} does not require grad!")
        return 0.0
    
    grad = torch.autograd.grad(loss, comp_mask, allow_unused=True)[0]
    
    if grad is None:
        print(f"Warning: gradient is None for {module_name}!")
        return 0.0
    
    # Extract gradient for the specific component
    # Handle both 2D (batch, C) and 3D (batch, seq_len, C) tensors
    # Shape: (batch, C) for 2D, (batch, seq_len, C) or similar for 3D+
    grad_comp = grad[:, comp_idx] if grad.ndim == 2 else grad[..., comp_idx]
    
    # Average over all leading dimensions (batch, and seq_len if present)
    return grad_comp.mean().item()


def save_results(s_list, delta_list, labels, save_dir: Path):
    """Create plots and save statistics."""
    
    # Scatter plot: s_i vs Δ(i)
    plt.figure(figsize=(10, 6))
    plt.scatter(s_list, delta_list, s=15, alpha=0.5, edgecolors='black', linewidths=0.5)
    plt.xlabel("s_i = ∂L/∂m_i (gradient)", fontsize=12)
    plt.ylabel("Δ(i) = KL divergence increase", fontsize=12)
    plt.title("Assumption 2 Verification: Gradient vs Ablation Effect", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "assumption2_scatter.png", dpi=150)
    plt.close()
    
    # Scatter plot: |s_i| vs Δ(i)
    abs_s_list = [abs(s) for s in s_list]
    plt.figure(figsize=(10, 6))
    plt.scatter(abs_s_list, delta_list, s=15, alpha=0.5, edgecolors='black', linewidths=0.5)
    plt.xlabel("|s_i| = |∂L/∂m_i|", fontsize=12)
    plt.ylabel("Δ(i) = KL divergence increase", fontsize=12)
    plt.title("Assumption 2 Verification: Absolute Gradient vs Ablation Effect", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "assumption2_scatter_abs.png", dpi=150)
    plt.close()
    
    # Compute correlations
    pear = pearsonr(s_list, delta_list)
    spear = spearmanr(s_list, delta_list)
    pear_abs = pearsonr(abs_s_list, delta_list)
    spear_abs = spearmanr(abs_s_list, delta_list)
    
    # Save statistics
    with open(save_dir / "correlations.txt", "w") as f:
        f.write("=== Assumption 2 Verification Results ===\n\n")
        f.write("=== s_i vs Δ(i) ===\n")
        f.write(f"Pearson correlation: {pear.statistic:.6f}\n")
        f.write(f"  p-value: {pear.pvalue:.6e}\n")
        f.write(f"Spearman correlation: {spear.statistic:.6f}\n")
        f.write(f"  p-value: {spear.pvalue:.6e}\n\n")
        
        f.write("=== |s_i| vs Δ(i) ===\n")
        f.write(f"Pearson correlation: {pear_abs.statistic:.6f}\n")
        f.write(f"  p-value: {pear_abs.pvalue:.6e}\n")
        f.write(f"Spearman correlation: {spear_abs.statistic:.6f}\n")
        f.write(f"  p-value: {spear_abs.pvalue:.6e}\n\n")
        
        f.write("=== Statistics ===\n")
        f.write(f"Number of components tested: {len(labels)}\n")
        f.write(f"s_i range: [{min(s_list):.6f}, {max(s_list):.6f}]\n")
        f.write(f"|s_i| range: [{min(abs_s_list):.6f}, {max(abs_s_list):.6f}]\n")
        f.write(f"Δ(i) range: [{min(delta_list):.6f}, {max(delta_list):.6f}]\n")
        
        f.write("\n=== Component Labels (first 10) ===\n")
        for i, (mod, idx) in enumerate(labels[:10]):
            f.write(f"{i}: {mod}, component {idx}\n")
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Tested {len(labels)} components")
    print(f"\nCorrelations (s_i vs Δ(i)):")
    print(f"  Pearson:  {pear.statistic:7.4f} (p={pear.pvalue:.4e})")
    print(f"  Spearman: {spear.statistic:7.4f} (p={spear.pvalue:.4e})")
    print(f"\nCorrelations (|s_i| vs Δ(i)):")
    print(f"  Pearson:  {pear_abs.statistic:7.4f} (p={pear_abs.pvalue:.4e})")
    print(f"  Spearman: {spear_abs.statistic:7.4f} (p={spear_abs.pvalue:.4e})")
    print("="*70)
    
    # Interpretation
    print("\nInterpretation:")
    corr = pear_abs.statistic
    if abs(corr) > 0.7:
        print("  ✓ Strong correlation - Assumption 2 is well-supported!")
    elif abs(corr) > 0.5:
        print("  ~ Moderate correlation - Assumption 2 is partially supported")
    else:
        print("  ✗ Weak correlation - Assumption 2 may not hold well")


def verify_assumption2_lm(
    model: ComponentModel,
    input_ids: torch.Tensor,
    save_dir: str = "experiments/assumption2",
    max_components_per_module: int = 100,
    layer_filter: str | None = None,
    seed: int = 42,
):
    """
    Verify Assumption 2 for language models.
    
    Tests whether gradient-based importance s_i = ∂L/∂m_i predicts
    the actual ablation effect Δ(i) = L(model with component i ablated).
    
    Args:
        model: Trained ComponentModel
        input_ids: Token IDs, shape (batch, seq_len)
        save_dir: Directory to save results
        max_components_per_module: Maximum components to test per module
        layer_filter: Only test modules starting with this prefix (e.g., "h.0.")
        seed: Random seed for component sampling
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Starting Assumption 2 Verification for Language Model")
    print("="*70)
    print(f"Input shape: {input_ids.shape}")
    print(f"Device: {device}")
    print(f"Layer filter: {layer_filter or 'None (all layers)'}")
    print(f"Max components per module: {max_components_per_module}")
    
    # Get causal importance outputs
    print("\nComputing causal importances...")
    out_with_cache = model(input_ids, cache_type="input")
    ci_outputs = model.calc_causal_importances(
        out_with_cache.cache,
        sampling=SamplingType.uniform
    )
    
    # Get unmasked output for comparison
    print("Computing baseline (unmasked) output...")
    out_full = model(input_ids)
    
    # Define KL divergence loss function
    def kl_loss(logits):
        """KL divergence between masked and unmasked outputs."""
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        target_log_probs = torch.nn.functional.log_softmax(out_full, dim=-1)
        return torch.nn.functional.kl_div(
            log_probs,
            target_log_probs,
            reduction='batchmean',
            log_target=True
        )
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    s_list = []
    delta_list = []
    labels = []
    
    # Process each module
    modules_to_process = [
        (name, mask) for name, mask in ci_outputs.lower_leaky.items()
        if layer_filter is None or name.startswith(layer_filter)
    ]
    
    print(f"\nProcessing {len(modules_to_process)} modules:")
    for module_name, _ in modules_to_process:
        print(f"  - {module_name}")
    
    for module_name, mask_vals in modules_to_process:
        C = mask_vals.shape[-1]
        
        # Sample components if there are too many
        if C > max_components_per_module:
            component_indices = random.sample(range(C), max_components_per_module)
        else:
            component_indices = list(range(C))
        
        print(f"\n[{module_name}] Processing {len(component_indices)}/{C} components")
        
        for i, c in enumerate(component_indices):
            labels.append((module_name, c))
            
            # Compute Δ(i): ablation effect
            mask_infos = build_mask_infos_with_ablation(
                ci_outputs, module_name, c
            )
            out_ablated = model(input_ids, mask_infos=mask_infos)
            delta_i = kl_loss(out_ablated).item()
            delta_list.append(delta_i)
            
            # Compute s_i: gradient
            s_i = compute_gradient(
                model, input_ids, ci_outputs, module_name, c, kl_loss, device
            )
            s_list.append(s_i)
            
            # Progress update
            if (i + 1) % 25 == 0:
                print(f"  Progress: {i+1}/{len(component_indices)} components")
    
    print(f"\nTotal components processed: {len(labels)}")
    
    # Save results
    print(f"\nSaving results to {save_dir}...")
    save_results(s_list, delta_list, labels, save_dir)
    
    print(f"\nFiles created:")
    print(f"  - {save_dir}/assumption2_scatter.png")
    print(f"  - {save_dir}/assumption2_scatter_abs.png")
    print(f"  - {save_dir}/correlations.txt")


if __name__ == "__main__":
    import argparse

    from transformers import AutoTokenizer
    
    parser = argparse.ArgumentParser(
        description="Verify Assumption 2 for language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Test layer 0 only (faster)
  python verify_ass2_lm.py \\
      --model_path wandb:goodfire/spd/runs/abc123xyz \\
      --text "Once upon a time, there was a little girl." \\
      --layer_filter "h.0." \\
      --max_components 50

  # Test all layers
  python verify_ass2_lm.py \\
      --model_path wandb:goodfire/spd/runs/abc123xyz \\
      --max_components 100
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained ComponentModel (WandB path or local checkpoint)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Once upon a time, there was a little girl named Emma who loved to play.",
        help="Text to analyze (will be tokenized)"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="SimpleStories/test-SimpleStories-gpt2-1.25M",
        help="HuggingFace tokenizer name"
    )
    parser.add_argument(
        "--max_components",
        type=int,
        default=100,
        help="Maximum components to test per module (use smaller value for faster testing)"
    )
    parser.add_argument(
        "--layer_filter",
        type=str,
        default=None,
        help="Only test modules starting with this prefix (e.g., 'h.0.' for layer 0)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="experiments/assumption2",
        help="Directory to save results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for component sampling"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("ASSUMPTION 2 VERIFICATION FOR LANGUAGE MODELS")
    print("="*70)
    
    # Load model
    print(f"\n[1/4] Loading model from: {args.model_path}")
    run_info = SPDRunInfo.from_path(args.model_path)
    model = ComponentModel.from_run_info(run_info)
    model.eval()
    print(f"  ✓ Model loaded")
    print(f"  Components: {list(model.components.keys())[:3]}... ({len(model.components)} total)")
    
    # Load tokenizer
    print(f"\n[2/4] Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    print(f"  ✓ Tokenizer loaded")
    
    # Tokenize input
    print(f"\n[3/4] Tokenizing input text")
    print(f"  Text: '{args.text}'")
    tokens = tokenizer.encode(args.text, return_tensors="pt")
    print(f"  ✓ Tokens: {tokens.shape} = {tokens.tolist()[0][:10]}...")
    
    # Run verification
    print(f"\n[4/4] Running verification")
    verify_assumption2_lm(
        model=model,
        input_ids=tokens,
        save_dir=args.save_dir,
        max_components_per_module=args.max_components,
        layer_filter=args.layer_filter,
        seed=args.seed,
    )
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE!")
    print("="*70)
    print(f"\nView results in: {args.save_dir}/")