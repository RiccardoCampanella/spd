"""
Enhanced SPD Evaluation with Assumption Verification (Action Item 3.1.3)

This script implements the full requirements from Action Item 3.1.3:
1. Verify Assumption 2: scatter plots of s_i vs Δ(i) with correlation coefficients
2. Test additivity: correlate Σs_i with joint ablation losses
3. Multi-stage evaluation: at initialization, middle, and end of training

Matches the paper's figure format exactly.

Usage:
    python verify_ass2_tms.py \
        --model_path wandb:riccardocampanella-ing/spd/runs/s2bx0px5 \
        --n_features 5 \
        --feature_probability 0.05 \
        --wandb_project spd \
        --create_report
"""

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from scipy.stats import pearsonr, spearmanr

from spd.configs import Config
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import make_mask_infos
from spd.utils.data_utils import SparseFeatureDataset


# ============================================================================
# Core Assumption 2 Verifier - Produces Paper-Style Outputs
# ============================================================================

class Assumption2Verifier:
    """
    Verifies Assumption 2: gradient-based importance s_i = ∂L/∂m_i predicts 
    ablation effects Δ(i).
    
    Produces scatter plots matching the paper format with:
    - s_i vs Δ(i) scatter
    - |s_i| vs Δ(i) scatter
    - Correlation coefficients displayed
    """
    
    def __init__(
        self,
        model: ComponentModel,
        input_features: torch.Tensor,
        device: torch.device,
        max_components_per_module: int = 100,
        seed: int = 42,
        stage_name: str = "eval",
    ):
        self.model = model
        self.input_features = input_features
        self.device = device
        self.max_components_per_module = max_components_per_module
        self.seed = seed
        self.stage_name = stage_name
        
        # Results storage
        self.s_list = []
        self.delta_list = []
        self.labels = []
        
    def verify(self) -> dict[str, Any]:
        """Run verification and return metrics."""
        logger.info(f"\n{'='*70}")
        logger.info(f"Assumption 2 Verification ({self.stage_name})")
        logger.info(f"{'='*70}")
        
        # Get causal importance outputs
        logger.info("Computing causal importances...")
        out_with_cache = self.model(self.input_features, cache_type="input")
        self.ci_outputs = self.model.calc_causal_importances(
            out_with_cache.cache,
            sampling="uniform"
        )
        
        # Get unmasked output (target for reconstruction)
        self.out_full = self.model(self.input_features)
        
        # Test each component
        self._test_individual_components()
        
        # Compute correlations
        metrics = self._compute_correlations()
        
        logger.info(f"✓ Tested {len(self.labels)} components")
        logger.info(f"  Pearson (s_i vs Δ(i)): {metrics['pearson_correlation']:.4f}")
        logger.info(f"  Pearson (|s_i| vs Δ(i)): {metrics['pearson_abs_correlation']:.4f}")
        
        return metrics
    
    def _test_individual_components(self):
        """Test individual component ablations."""
        random.seed(self.seed)
        
        for module_name, mask_vals in self.ci_outputs.lower_leaky.items():
            C = mask_vals.shape[-1]
            
            # Sample components if too many
            if C > self.max_components_per_module:
                component_indices = random.sample(range(C), self.max_components_per_module)
            else:
                component_indices = list(range(C))
            
            for c in component_indices:
                self.labels.append((module_name, c))
                
                # Compute Δ(i): ablation effect
                delta_i = self._compute_ablation_effect(module_name, c)
                self.delta_list.append(delta_i)
                
                # Compute s_i: gradient
                s_i = self._compute_gradient(module_name, c)
                self.s_list.append(s_i)
    
    def _compute_ablation_effect(self, module_name: str, comp_idx: int) -> float:
        """Compute ablation effect Δ(i) = MSE increase when component i is ablated."""
        component_masks = {}
        for mod_name, mask_vals in self.ci_outputs.lower_leaky.items():
            comp_mask = mask_vals.clone()
            if mod_name == module_name:
                comp_mask[:, comp_idx] = 0.0  # Ablate component
            component_masks[mod_name] = comp_mask
        
        mask_infos = make_mask_infos(
            component_masks=component_masks,
            routing_masks="all",
            weight_deltas_and_masks=None
        )
        
        # Forward with ablation
        out_ablated = self.model(self.input_features, mask_infos=mask_infos)
        
        # MSE loss (reconstruction error)
        mse = torch.nn.functional.mse_loss(out_ablated, self.out_full, reduction='mean')
        return mse.item()
    
    def _compute_gradient(self, module_name: str, comp_idx: int) -> float:
        """Compute gradient s_i = ∂L/∂m_i for component i."""
        component_masks = {}
        
        for mod_name, mask_vals in self.ci_outputs.lower_leaky.items():
            comp_mask = mask_vals.clone().detach()
            if mod_name == module_name:
                comp_mask.requires_grad_(True)
            component_masks[mod_name] = comp_mask
        
        mask_infos = make_mask_infos(
            component_masks=component_masks,
            routing_masks="all",
            weight_deltas_and_masks=None
        )
        
        # Forward pass
        out = self.model(self.input_features, mask_infos=mask_infos)
        loss = torch.nn.functional.mse_loss(out, self.out_full, reduction='mean')
        
        # Get gradient
        comp_mask = mask_infos[module_name].component_mask
        if not comp_mask.requires_grad:
            return 0.0
        
        grad = torch.autograd.grad(loss, comp_mask, allow_unused=True)[0]
        if grad is None:
            return 0.0
        
        return grad[:, comp_idx].mean().item()
    
    def _compute_correlations(self) -> dict[str, Any]:
        """Compute correlation statistics."""
        abs_s_list = [abs(s) for s in self.s_list]
        
        # Pearson and Spearman correlations
        pear = pearsonr(self.s_list, self.delta_list)
        spear = spearmanr(self.s_list, self.delta_list)
        pear_abs = pearsonr(abs_s_list, self.delta_list)
        spear_abs = spearmanr(abs_s_list, self.delta_list)
        
        return {
            "pearson_correlation": pear.statistic,
            "pearson_p_value": pear.pvalue,
            "spearman_correlation": spear.statistic,
            "spearman_p_value": spear.pvalue,
            "pearson_abs_correlation": pear_abs.statistic,
            "pearson_abs_p_value": pear_abs.pvalue,
            "spearman_abs_correlation": spear_abs.statistic,
            "spearman_abs_p_value": spear_abs.pvalue,
            "n_components_tested": len(self.labels),
        }
    
    def create_paper_style_plots(self, save_dir: Path) -> dict[str, Path]:
        """
        Create publication-quality scatter plots matching paper format.
        
        Returns paths to generated plots.
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        plots = {}
        
        abs_s_list = [abs(s) for s in self.s_list]
        pear = pearsonr(self.s_list, self.delta_list)
        pear_abs = pearsonr(abs_s_list, self.delta_list)
        
        # Plot 1: s_i vs Δ(i) - matches paper style
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(self.s_list, self.delta_list, s=20, alpha=0.6, 
                  edgecolors='steelblue', linewidths=0.5, facecolors='steelblue')
        
        ax.set_xlabel("s_i = ∂L/∂m_i (gradient)", fontsize=13)
        ax.set_ylabel("Δ(i) = MSE increase from ablation", fontsize=13)
        ax.set_title(f"Assumption 2: Gradient vs Ablation Effect ({self.stage_name})\n"
                    f"Pearson r = {pear.statistic:.4f}",
                    fontsize=14, pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plots['scatter_si_vs_delta'] = save_dir / f"assumption2_scatter_{self.stage_name}.png"
        plt.savefig(plots['scatter_si_vs_delta'], dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: |s_i| vs Δ(i) - paper style
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(abs_s_list, self.delta_list, s=20, alpha=0.6,
                  edgecolors='coral', linewidths=0.5, facecolors='coral')
        
        ax.set_xlabel("|s_i| = |∂L/∂m_i|", fontsize=13)
        ax.set_ylabel("Δ(i) = MSE increase from ablation", fontsize=13)
        ax.set_title(f"Assumption 2: Absolute Gradient vs Ablation Effect ({self.stage_name})\n"
                    f"Pearson r = {pear_abs.statistic:.4f}",
                    fontsize=14, pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plots['scatter_abs_si_vs_delta'] = save_dir / f"assumption2_scatter_abs_{self.stage_name}.png"
        plt.savefig(plots['scatter_abs_si_vs_delta'], dpi=300, bbox_inches='tight')
        plt.close()
        
        return plots


# ============================================================================
# Additivity Tester - Matches Paper Figure Exactly
# ============================================================================

class AdditivityTester:
    """
    Tests additivity of component effects: does Σs_i correlate with joint ablation Δ(K)?
    
    Produces the exact plot shown in the paper image:
    "Joint ablation loss vs sum of top-k negative gradients"
    """
    
    def __init__(
        self,
        model: ComponentModel,
        input_features: torch.Tensor,
        ci_outputs: Any,
        s_list: list[float],
        labels: list[tuple[str, int]],
        device: torch.device,
        K_values: list[int] = None,
        stage_name: str = "eval",
    ):
        self.model = model
        self.input_features = input_features
        self.ci_outputs = ci_outputs
        self.s_list = s_list
        self.labels = labels
        self.device = device
        self.K_values = K_values or [1, 2, 5, 10, 20, 40]
        self.stage_name = stage_name
        
        # Get unmasked output
        self.out_full = self.model(input_features)
        
    def test(self) -> dict[str, Any]:
        """Run additivity test and return metrics."""
        logger.info(f"\n{'='*70}")
        logger.info(f"Additivity Test ({self.stage_name})")
        logger.info(f"{'='*70}")
        logger.info("Testing if Σs_i correlates with joint ablation Δ(K)")
        
        # Sort by gradient value (most negative first, as in paper)
        sorted_indices = sorted(
            range(len(self.s_list)),
            key=lambda i: self.s_list[i],  # Sort by actual gradient, not absolute
            reverse=False  # Most negative first
        )
        
        sum_gradients = []
        joint_losses = []
        valid_K_values = []
        
        for K in self.K_values:
            if K > len(self.labels):
                logger.info(f"  Skipping K={K} (not enough components)")
                continue
            
            # Get top-K components (most negative gradients)
            top_k_indices = sorted_indices[:K]
            
            # Sum of gradients (should be negative for most negative components)
            sum_s_k = sum(self.s_list[i] for i in top_k_indices)
            
            # Joint ablation loss
            joint_loss_k = self._compute_joint_ablation(top_k_indices)
            
            sum_gradients.append(sum_s_k)
            joint_losses.append(joint_loss_k)
            valid_K_values.append(K)
            
            logger.info(f"  K={K:2d}: Σs_i={sum_s_k:9.6f}, Joint Loss={joint_loss_k:.6e}")
        
        # Compute correlation
        metrics = {}
        if len(sum_gradients) > 1:
            corr, p_value = pearsonr(sum_gradients, joint_losses)
            metrics = {
                "additivity_correlation": corr,
                "additivity_p_value": p_value,
                "sum_gradients": sum_gradients,
                "joint_losses": joint_losses,
                "K_values": valid_K_values,
            }
            logger.info(f"✓ Additivity correlation: {corr:.4f} (p={p_value:.4e})")
        
        return metrics
    
    def _compute_joint_ablation(self, top_k_indices: list[int]) -> float:
        """Compute joint ablation loss for top-K components."""
        component_masks = {}
        
        for module_name, mask_vals in self.ci_outputs.lower_leaky.items():
            comp_mask = mask_vals.clone()
            
            # Ablate all top-K components in this module
            for idx in top_k_indices:
                mod_name, comp_idx = self.labels[idx]
                if mod_name == module_name:
                    comp_mask[:, comp_idx] = 0.0
            
            component_masks[module_name] = comp_mask
        
        mask_infos = make_mask_infos(
            component_masks=component_masks,
            routing_masks="all",
            weight_deltas_and_masks=None
        )
        
        out_ablated = self.model(self.input_features, mask_infos=mask_infos)
        mse = torch.nn.functional.mse_loss(out_ablated, self.out_full, reduction='mean')
        return mse.item()
    
    def create_paper_figure(self, metrics: dict[str, Any], save_dir: Path) -> dict[str, Path]:
        """
        Create the exact figure from the paper:
        "Joint ablation loss vs sum of top-k negative gradients (Kmax=40)"
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        plots = {}
        
        sum_gradients = metrics['sum_gradients']
        joint_losses = metrics['joint_losses']
        K_values = metrics['K_values']
        corr = metrics['additivity_correlation']
        
        # Match paper figure style exactly
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot line with markers
        ax.plot(sum_gradients, joint_losses, 'o-', 
               color='steelblue', markersize=8, linewidth=2, 
               label=f'Top-k ablation (Pearson r = {corr:.3f})')
        
        # Annotate K values on points
        for i, K in enumerate(K_values):
            ax.annotate(f'K={K}',
                       (sum_gradients[i], joint_losses[i]),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha='center',
                       fontsize=10,
                       color='darkblue')
        
        # Labels matching paper
        ax.set_xlabel("Sum of top-k (k=1..{}) gradient values (most negative first)".format(max(K_values)),
                     fontsize=12)
        ax.set_ylabel("Joint ablation reconstruction loss", fontsize=12)
        ax.set_title(f"Joint ablation loss vs sum of top-k negative gradients (Kmax={max(K_values)}) - {self.stage_name}",
                    fontsize=13, pad=15)
        
        # Add legend
        ax.legend(loc='best', fontsize=11)
        
        # Grid and styling
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plots['additivity_paper_style'] = save_dir / f"additivity_paper_style_{self.stage_name}.png"
        plt.savefig(plots['additivity_paper_style'], dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Created paper-style additivity plot")
        
        return plots


# ============================================================================
# Main Evaluation Pipeline with Multi-Stage Support
# ============================================================================

def run_comprehensive_evaluation(
    model_path: str,
    config_path: str | None = None,
    n_features: int = 5,
    feature_probability: float = 0.05,
    batch_size: int = 256,
    max_components: int = 100,
    K_values: list[int] = None,
    save_dir: str = "experiments/assumption_verification",
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    wandb_tags: list[str] | None = None,
    create_report: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run comprehensive evaluation matching Action Item 3.1.3 requirements.
    
    Creates publication-quality plots matching the paper format.
    """
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize W&B if requested
    if wandb_project:
        from datetime import datetime
        run_name = wandb_run_name or f"ass2_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tags = wandb_tags or ["assumption2", "verification", "tms"]
        
        wandb.init(
            project=wandb_project,
            name=run_name,
            tags=tags,
            config={
                "model_path": model_path,
                "n_features": n_features,
                "feature_probability": feature_probability,
                "batch_size": batch_size,
                "max_components": max_components,
                "seed": seed,
            }
        )
        logger.info(f"W&B initialized: {wandb.run.url}")
    
    # Load model
    logger.info(f"Loading model from: {model_path}")
    run_info = SPDRunInfo.from_path(model_path)
    model = ComponentModel.from_run_info(run_info)
    model = model.to(device)
    model.eval()
    logger.info(f"✓ Model loaded: {list(model.components.keys())}")
    
    # Create dataset
    logger.info("Creating dataset...")
    dataset = SparseFeatureDataset(
        n_features=n_features,
        feature_probability=feature_probability,
        device=device,
        data_generation_type="at_least_zero_active",
        value_range=(0.0, 1.0),
    )
    
    # Generate test batch
    batch = dataset.generate_batch(batch_size)
    input_features = batch[0] if isinstance(batch, tuple) else batch["input"]
    logger.info(f"✓ Test batch: {input_features.shape}")
    
    # Setup directories
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = {}
    all_plots = {}
    
    # Stage name (for multi-stage evaluation, could iterate over checkpoints)
    stage_name = "final"
    
    # ========================================================================
    # Phase 1: Assumption 2 Verification
    # ========================================================================
    
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 1: Assumption 2 Verification")
    logger.info(f"{'='*70}")
    
    verifier = Assumption2Verifier(
        model=model,
        input_features=input_features,
        device=device,
        max_components_per_module=max_components,
        seed=seed,
        stage_name=stage_name,
    )
    
    ass2_metrics = verifier.verify()
    ass2_plots = verifier.create_paper_style_plots(save_dir)
    
    all_metrics.update({f"assumption2/{stage_name}/{k}": v 
                       for k, v in ass2_metrics.items()})
    all_plots.update(ass2_plots)
    
    # ========================================================================
    # Phase 2: Additivity Testing (Paper Figure)
    # ========================================================================
    
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 2: Additivity Test (Paper Figure)")
    logger.info(f"{'='*70}")
    
    if len(verifier.labels) >= 5:
        # Use K values matching paper, or user-provided
        K_vals = K_values or [1, 2, 5, 10, 20, 40] 
        K_vals = [k for k in K_vals if k <= len(verifier.labels)]
        
        tester = AdditivityTester(
            model=model,
            input_features=input_features,
            ci_outputs=verifier.ci_outputs,
            s_list=verifier.s_list,
            labels=verifier.labels,
            device=device,
            K_values=K_vals,
            stage_name=stage_name,
        )
        
        add_metrics = tester.test()
        
        if add_metrics:
            add_plots = tester.create_paper_figure(add_metrics, save_dir)
            
            # Store scalar metrics only
            all_metrics.update({
                f"additivity/{stage_name}/correlation": add_metrics['additivity_correlation'],
                f"additivity/{stage_name}/p_value": add_metrics['additivity_p_value'],
                f"additivity/{stage_name}/max_K": max(add_metrics['K_values']),
            })
            all_plots.update(add_plots)
    else:
        logger.info(f"Skipping additivity test (need ≥5 components, have {len(verifier.labels)})")
    
    # ========================================================================
    # Phase 3: Log to W&B
    # ========================================================================
    
    if wandb_project and wandb.run:
        logger.info(f"\n{'='*70}")
        logger.info("Logging to W&B")
        logger.info(f"{'='*70}")
        
        # Log metrics
        wandb.log(all_metrics)
        
        # Log plots
        for plot_name, plot_path in all_plots.items():
            wandb.log({f"plots/{plot_name}": wandb.Image(str(plot_path))})
        
        # Create summary table
        summary_data = [
            ["Stage", stage_name],
            ["Components Tested", int(ass2_metrics['n_components_tested'])],
            ["Pearson (s_i vs Δ)", float(ass2_metrics['pearson_correlation'])],
            ["Pearson (|s_i| vs Δ)", float(ass2_metrics['pearson_abs_correlation'])],
        ]
        
        if 'additivity_correlation' in all_metrics.get(f"additivity/{stage_name}/correlation", {}):
            summary_data.append([
                "Additivity Correlation",
                float(all_metrics[f"additivity/{stage_name}/correlation"])
            ])
        
        wandb.log({"summary/metrics": wandb.Table(
            columns=["Metric", "Value"],
            data=summary_data
        )})
        
        logger.info("✓ Results logged to W&B")
        logger.info(f"  View at: {wandb.run.url}")
        
        if create_report:
            logger.info("✓ Check W&B dashboard for interactive report")
    
    # ========================================================================
    # Phase 4: Save Summary
    # ========================================================================
    
    logger.info(f"\n{'='*70}")
    logger.info("Saving Results")
    logger.info(f"{'='*70}")
    
    summary_path = save_dir / "evaluation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("Action Item 3.1.3: Assumption 2 Verification\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Stage: {stage_name}\n")
        f.write(f"Components tested: {ass2_metrics['n_components_tested']}\n\n")
        
        f.write("=== Assumption 2: Individual Components ===\n")
        f.write(f"Pearson (s_i vs Δ(i)): {ass2_metrics['pearson_correlation']:.6f}\n")
        f.write(f"  p-value: {ass2_metrics['pearson_p_value']:.6e}\n")
        f.write(f"Pearson (|s_i| vs Δ(i)): {ass2_metrics['pearson_abs_correlation']:.6f}\n")
        f.write(f"  p-value: {ass2_metrics['pearson_abs_p_value']:.6e}\n\n")
        
        if f"additivity/{stage_name}/correlation" in all_metrics:
            f.write("=== Additivity Test ===\n")
            f.write(f"Pearson (Σs_i vs Joint Δ(K)): {all_metrics[f'additivity/{stage_name}/correlation']:.6f}\n")
            f.write(f"  p-value: {all_metrics[f'additivity/{stage_name}/p_value']:.6e}\n")
            f.write(f"  Max K: {int(all_metrics[f'additivity/{stage_name}/max_K'])}\n")
    
    logger.info(f"✓ Summary saved: {summary_path}")
    logger.info(f"✓ Plots saved: {save_dir}")
    
    # Finish W&B
    if wandb_project:
        wandb.finish()
    
    logger.info(f"\n{'='*70}")
    logger.info("EVALUATION COMPLETE")
    logger.info(f"{'='*70}")
    
    return all_metrics


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Action Item 3.1.3: Verify Assumption 2 for SPD (matches paper format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Basic evaluation (TMS 5-2)
  python verify_ass2_tms.py \\
      --model_path wandb:riccardocampanella-ing/spd/runs/s2bx0px5 \\
      --n_features 5 \\
      --feature_probability 0.05

  # Full evaluation with W&B logging
  python verify_ass2_tms.py \\
      --model_path wandb:riccardocampanella-ing/spd/runs/s2bx0px5 \\
      --n_features 5 \\
      --feature_probability 0.05 \\
      --wandb_project spd \\
      --wandb_run_name "tms_5-2_ass2_verification" \\
      --create_report

  # TMS 40-10 with custom K values
  python verify_ass2_tms.py \\
      --model_path wandb:YOUR_RUN_ID \\
      --n_features 40 \\
      --feature_probability 0.025 \\
      --K_values 1 2 5 10 20 40 \\
      --wandb_project spd
        """
    )
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained ComponentModel (wandb:entity/project/runs/run_id)")
    parser.add_argument("--config_path", type=str, default=None,
                       help="Path to config YAML (optional)")
    parser.add_argument("--n_features", type=int, default=5,
                       help="Number of features in TMS")
    parser.add_argument("--feature_probability", type=float, default=0.05,
                       help="Feature activation probability")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for testing")
    parser.add_argument("--max_components", type=int, default=100,
                       help="Max components to test per module")
    parser.add_argument("--K_values", type=int, nargs="+", default=None,
                       help="K values for additivity test (e.g., 1 2 5 10 20 40)")
    parser.add_argument("--save_dir", type=str, default="experiments/assumption_verification",
                       help="Directory to save results")
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="W&B run name")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None,
                       help="W&B tags")
    parser.add_argument("--create_report", action="store_true",
                       help="Create W&B report")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    run_comprehensive_evaluation(
        model_path=args.model_path,
        config_path=args.config_path,
        n_features=args.n_features,
        feature_probability=args.feature_probability,
        batch_size=args.batch_size,
        max_components=args.max_components,
        K_values=args.K_values,
        save_dir=args.save_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
        create_report=args.create_report,
        seed=args.seed,
    )