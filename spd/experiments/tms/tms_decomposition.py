"""Run spd on a TMS model.

Note that the first instance index is fixed to the identity matrix. This is done so we can compare
the losses of the "correct" solution during training.
"""

import json
from pathlib import Path
from typing import Any
import torch
import fire
import wandb

from spd.configs import Config
from spd.experiments.tms.configs import TMSTaskConfig
from spd.experiments.tms.models import TMSModel, TMSTargetRunInfo
from spd.log import logger
from spd.run_spd import optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import save_pre_run_info, set_seed
from spd.utils.run_utils import get_output_dir
from spd.utils.wandb_utils import init_wandb

def verify_assumption2(
    component_model,
    batch,
    target_out,
    ci,
    device,
    config,
    step: int,
    out_dir: Path | None,
    max_components_per_module: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Verify Assumption 2: Test if gradient s_i = ∂L/∂g_i predicts ablation effect Δ(i).
    
    Args:
        component_model: The component model
        batch: Input batch
        target_out: Target output (unmasked)
        ci: Causal importance outputs
        device: Device
        config: Config object
        step: Current training step
        out_dir: Output directory for saving plots
        max_components_per_module: Max components to test per module
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with correlation metrics and plot paths
    """
    import random
    from pathlib import Path
    
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image
    from scipy.stats import pearsonr, spearmanr
    
    from spd.models.components import make_mask_infos
    from spd.utils.component_utils import calc_stochastic_component_mask_info
    from spd.utils.general_utils import calc_sum_recon_loss_lm
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Assumption 2 Verification at Step {step}")
    logger.info(f"{'='*70}")
    
    # Set seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # FIX: Generate a single common random number (CRN) sample for the stochastic mask
    first_module_name = next(iter(ci.lower_leaky))
    C = ci.lower_leaky[first_module_name].shape[-1]
    batch_size = ci.lower_leaky[first_module_name].shape[0]
    
    # r ~ Uniform(0, 1) or {0, 1} depending on implementation, here we use Continuous Relaxation
    r_sample = torch.rand(batch_size, C, device=device)
    
    # Storage for results
    s_list: list[float] = []
    delta_list: list[float] = []
    labels: list[tuple[str, int]] = []
    
    # Get layerwise loss function - we need to compute L_stochastic-recon-layerwise^(l)
    # For each layer l, we'll compute the gradient w.r.t. components in that layer
    for module_name, mask_vals in ci.lower_leaky.items():
        C = mask_vals.shape[-1]
        
        # Sample components if too many
        if C > max_components_per_module:
            component_indices = random.sample(range(C), max_components_per_module)
        else:
            component_indices = list(range(C))
        
        logger.info(f"Processing {len(component_indices)}/{C} components in {module_name}")
        
        for c in component_indices:
            labels.append((module_name, c))
            
            # ====================================================================
            # Compute Δ(i): Ablation effect (Deterministic)
            # ====================================================================
            # This calculation uses the deterministic ablation mask (g_i = 0)
            component_masks_ablate = {}
            for mod_name, mask_vals_mod in ci.lower_leaky.items():
                comp_mask = mask_vals_mod.clone()
                if mod_name == module_name:
                    comp_mask[:, c] = 0.0  # Ablate component i
                component_masks_ablate[mod_name] = comp_mask
            
            mask_infos_ablate = make_mask_infos(
                component_masks=component_masks_ablate,
                routing_masks="all",
                weight_deltas_and_masks=None
            )
            
            # Forward pass with ablation
            out_ablated = component_model(batch, mask_infos=mask_infos_ablate)
            
            # Compute loss (MSE for TMS)
            delta_i = calc_sum_recon_loss_lm(
                pred=out_ablated,
                target=target_out,
                loss_type=config.output_loss_type
            ).item()
            delta_list.append(delta_i)
            
            # ====================================================================
            # Compute s_i: Gradient of layerwise loss w.r.t. g_i (Stochastic)
            # ====================================================================
            
            # 1. Create component importance tensor (g_c) and enable gradient tracking
            g_c_tensor = ci.lower_leaky[module_name].clone().detach()
            
            # Enable gradient for the single component being tested (g_i)
            g_c_tensor[:, c].requires_grad_(True)
            
            # 2. Manually compute the STOCHASTIC MASK m_c (Equation 13)
            # m_c = g_c + (1 - g_c) * r_c
            # The mask r_c is the same random sample across all components in the module
            m_c_tensor = g_c_tensor + (1 - g_c_tensor) * r_sample
            
            # 3. Create mask info structure for the stochastic layerwise forward pass
            # We use the component mask m_c_tensor for the current module, 
            # and effectively no mask (identity mask) for other modules
            mask_infos_stochastic = make_mask_infos(
                component_masks={module_name: m_c_tensor},
                routing_masks="all",
                weight_deltas_and_masks=None
            )

            # 4. Forward pass with only layer l masked using the STOCHASTIC mask
            # We must ensure only the layer 'module_name' uses the mask, others use identity (default)
            out_layerwise_stochastic = component_model(batch, mask_infos={module_name: mask_infos_stochastic[module_name]})
            
            # 5. Compute layerwise loss
            loss_layerwise_stochastic = calc_sum_recon_loss_lm(
                pred=out_layerwise_stochastic,
                target=target_out,
                loss_type=config.output_loss_type
            )
            
            # 6. Get gradient w.r.t. the importance score g_c[:, c]
            if not g_c_tensor[:, c].requires_grad:
                s_i = 0.0
            else:
                grad = torch.autograd.grad(
                    loss_layerwise_stochastic,
                    g_c_tensor[:, c], # Gradient w.r.t. g_i
                    allow_unused=True,
                    retain_graph=False
                )[0]
                if grad is None:
                    s_i = 0.0
                else:
                    # Extract gradient for component i and average
                    s_i = grad.mean().item()
            
            s_list.append(s_i)
    
    # ====================================================================
    # Compute correlations
    # ====================================================================
    # ... (Correlation calculation is standard)
    abs_s_list = [abs(s) for s in s_list]
    pear = pearsonr(s_list, delta_list)
    pear_abs = pearsonr(abs_s_list, delta_list)
    spear = spearmanr(s_list, delta_list)
    spear_abs = spearmanr(abs_s_list, delta_list)
    
    logger.info(f"✓ Tested {len(labels)} components")
    logger.info(f"  Pearson (s_i vs Δ(i)): {pear.statistic:.4f} (p={pear.pvalue:.4e})")
    logger.info(f"  Pearson (|s_i| vs Δ(i)): {pear_abs.statistic:.4f} (p={pear_abs.pvalue:.4e})")
    
    # ====================================================================
    # Create plots
    # ... (Plotting is standard)
    # ====================================================================
    stage_name = "initialization" if step == 0 else ("middle" if step < config.steps else "end")
    plots = {}
    
    # Plot 1: s_i vs Δ(i)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(s_list, delta_list, s=20, alpha=0.6, 
              edgecolors='steelblue', linewidths=0.5, facecolors='steelblue')
    ax.set_xlabel("s_i = ∂L/∂g_i (stochastic gradient)", fontsize=13)
    ax.set_ylabel("Δ(i) = MSE increase from ablation", fontsize=13)
    ax.set_title(f"Assumption 2: Stochastic Gradient vs Ablation Effect ({stage_name}, step {step})\n"
                f"Pearson r = {pear.statistic:.4f}",
                fontsize=14, pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    plot_path_si = out_dir / f"assumption2_scatter_si_step_{step}.png" if out_dir else None
    if plot_path_si:
        plt.savefig(plot_path_si, dpi=300, bbox_inches='tight')
        plots['scatter_si_vs_delta'] = plot_path_si
    plt.close()
    
    # Plot 2: |s_i| vs Δ(i)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(abs_s_list, delta_list, s=20, alpha=0.6,
              edgecolors='coral', linewidths=0.5, facecolors='coral')
    ax.set_xlabel("|s_i| = |∂L/∂g_i| (stochastic gradient)", fontsize=13)
    ax.set_ylabel("Δ(i) = MSE increase from ablation", fontsize=13)
    ax.set_title(f"Assumption 2: Absolute Stochastic Gradient vs Ablation Effect ({stage_name}, step {step})\n"
                f"Pearson r = {pear_abs.statistic:.4f}",
                fontsize=14, pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    plot_path_abs = out_dir / f"assumption2_scatter_abs_step_{step}.png" if out_dir else None
    if plot_path_abs:
        plt.savefig(plot_path_abs, dpi=300, bbox_inches='tight')
        plots['scatter_abs_si_vs_delta'] = plot_path_abs
    plt.close()
    
        # ====================================================================
    # Additivity Test: Joint ablation vs sum of top-k gradients
    # ====================================================================
    K_values = [1, 2, 5, 10, 20, 40]
    K_values = [k for k in K_values if k <= len(labels)]  # Only use K values we have enough components for
    
    if len(labels) >= 5:  # Need at least 5 components for meaningful test
        logger.info(f"\n{'='*70}")
        logger.info("Additivity Test: Joint ablation vs sum of top-k gradients")
        logger.info(f"{'='*70}")
        
        # Sort by gradient value (most negative first, as in paper)
        # This is correct: the most negative gradient corresponds to the component 
        # whose ablation provides the largest loss increase (if the correlation is positive).
        sorted_indices = sorted(
            range(len(s_list)),
            key=lambda i: s_list[i],  # Sort by actual gradient, not absolute
            reverse=False  # Most negative first
        )
        
        sum_gradients = []
        joint_losses = []
        valid_K_values = []
        
        for K in K_values:
            if K > len(labels):
                continue
            
            # Get top-K components (most negative gradients)
            top_k_indices = sorted_indices[:K]
            
            # Sum of gradients (should be negative for most negative components)
            sum_s_k = sum(s_list[i] for i in top_k_indices)
            
            # Joint ablation: ablate all top-K components together
            # This logic is correct for deterministic ablation
            component_masks_joint = {}
            for mod_name, mask_vals_mod in ci.lower_leaky.items():
                comp_mask = mask_vals_mod.clone()
                
                # Ablate all top-K components in this module
                for idx in top_k_indices:
                    mod_name_k, comp_idx_k = labels[idx]
                    if mod_name_k == mod_name:
                        comp_mask[:, comp_idx_k] = 0.0
                
                component_masks_joint[mod_name] = comp_mask
            
            mask_infos_joint = make_mask_infos(
                component_masks=component_masks_joint,
                routing_masks="all",
                weight_deltas_and_masks=None
            )
            
            # Forward pass with joint ablation
            out_joint = component_model(batch, mask_infos=mask_infos_joint)
            
            # Compute joint ablation loss (Deterministic)
            joint_loss_k = calc_sum_recon_loss_lm(
                pred=out_joint,
                target=target_out,
                loss_type=config.output_loss_type
            ).item()
            
            sum_gradients.append(sum_s_k)
            joint_losses.append(joint_loss_k)
            valid_K_values.append(K)
            
            logger.info(f"  K={K:2d}: Σs_i={sum_s_k:9.6f}, Joint Loss={joint_loss_k:.6e}")
        
        # Compute correlation
        if len(sum_gradients) > 1:
            additivity_corr, additivity_p = pearsonr(sum_gradients, joint_losses)
            logger.info(f"✓ Additivity correlation: {additivity_corr:.4f} (p={additivity_p:.4e})")
            
            # Plot 3: Joint ablation vs sum of top-k gradients
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot line with markers
            ax.plot(sum_gradients, joint_losses, 'o-', 
                   color='steelblue', markersize=8, linewidth=2, 
                   label=f'Top-k ablation (Pearson r = {additivity_corr:.3f})')
            
            # Annotate K values on points
            for i, K in enumerate(valid_K_values):
                ax.annotate(f'K={K}',
                           (sum_gradients[i], joint_losses[i]),
                           textcoords="offset points",
                           xytext=(0, 10),
                           ha='center',
                           fontsize=10,
                           color='darkblue')
            
            # Labels matching paper
            ax.set_xlabel(f"Sum of top-k (k=1..{max(valid_K_values)}) gradient values (most negative first)",
                         fontsize=12)
            ax.set_ylabel("Joint ablation reconstruction loss", fontsize=12)
            ax.set_title(f"Joint ablation loss vs sum of top-k negative gradients (Kmax={max(valid_K_values)}) - {stage_name}",
                        fontsize=13, pad=15)
            
            # Add legend
            ax.legend(loc='best', fontsize=11)
            
            # Grid and styling
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            
            plot_path_additivity = out_dir / f"assumption2_additivity_step_{step}.png" if out_dir else None
            if plot_path_additivity:
                plt.savefig(plot_path_additivity, dpi=300, bbox_inches='tight')
                plots['additivity_plot'] = plot_path_additivity
            plt.close()
            
            # Add additivity metrics to return dict
            additivity_metrics = {
                'additivity_correlation': additivity_corr,
                'additivity_p_value': additivity_p,
                'additivity_max_K': max(valid_K_values),
            }
        else:
            additivity_metrics = {}
    else:
        logger.info(f"Skipping additivity test (need ≥5 components, have {len(labels)})")
        additivity_metrics = {}
    
    # Return metrics and plot paths
    return {
        **plots,
        'pearson_correlation': pear.statistic,
        'pearson_p_value': pear.pvalue,
        'pearson_abs_correlation': pear_abs.statistic,
        'pearson_abs_p_value': pear_abs.pvalue,
        'spearman_correlation': spear.statistic,
        'spearman_p_value': spear.pvalue,
        'spearman_abs_correlation': spear_abs.statistic,
        'spearman_abs_p_value': spear_abs.pvalue,
        'n_components_tested': len(labels),
        'step': step,
        'stage': stage_name,
        **additivity_metrics,  # Include additivity metrics
    }

def main(
    config_path: Path | str | None = None,
    config_json: str | None = None,
    evals_id: str | None = None,
    sweep_id: str | None = None,
    sweep_params_json: str | None = None,
) -> None:
    assert (config_path is not None) != (config_json is not None), (
        "Need exactly one of config_path and config_json"
    )
    if config_path is not None:
        config = Config.from_file(config_path)
    else:
        assert config_json is not None
        config = Config(**json.loads(config_json.removeprefix("json:")))

    sweep_params = (
        None if sweep_params_json is None else json.loads(sweep_params_json.removeprefix("json:"))
    )

    device = get_device()
    logger.info(f"Using device: {device}")

    if config.wandb_project:
        tags = ["tms"]
        if evals_id:
            tags.append(evals_id)
        if sweep_id:
            tags.append(sweep_id)
        config = init_wandb(config, config.wandb_project, tags=tags)

    out_dir = get_output_dir(use_wandb_id=config.wandb_project is not None)

    task_config = config.task_config
    assert isinstance(task_config, TMSTaskConfig)

    set_seed(config.seed)
    logger.info(config)

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_run_info = TMSTargetRunInfo.from_path(config.pretrained_model_path)
    target_model = TMSModel.from_run_info(target_run_info)
    target_model = target_model.to(device)
    target_model.eval()

    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        if config.wandb_run_name:
            wandb.run.name = config.wandb_run_name

    save_pre_run_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        spd_config=config,
        sweep_params=sweep_params,
        target_model=target_model,
        train_config=target_model.config,
        task_name=config.task_config.task_name,
    )

    synced_inputs = target_run_info.config.synced_inputs
    dataset = SparseFeatureDataset(
        n_features=target_model.config.n_features,
        feature_probability=task_config.feature_probability,
        device=device,
        data_generation_type=task_config.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=synced_inputs,
    )
    train_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.microbatch_size, shuffle=False
    )
    eval_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.eval_batch_size, shuffle=False
    )

    tied_weights = None
    if target_model.config.tied_weights:
        tied_weights = [("linear1", "linear2")]

    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=out_dir,
        tied_weights=tied_weights,
    )

    # ====================================================================
    # Assumption 2 Verification
    # ====================================================================
    logger.info("\n" + "="*70)
    logger.info("Running Assumption 2 Verification")
    logger.info("="*70)
    
    # Load the final trained component model from checkpoint
    from spd.models.component_model import ComponentModel
    
    # Reconstruct component model (same way as in optimize())
    component_model = ComponentModel(
        target_model=target_model,
        target_module_patterns=config.all_module_patterns,
        C=config.C,
        ci_fn_type=config.ci_fn_type,
        ci_fn_hidden_dims=config.ci_fn_hidden_dims,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
        sigmoid_type=config.sigmoid_type,
    )
    component_model.to(device)
    
    # Load the final trained weights
    final_checkpoint_path = out_dir / f"model_{config.steps}.pth" if out_dir else None
    if final_checkpoint_path and final_checkpoint_path.exists():
        logger.info(f"Loading trained weights from {final_checkpoint_path}")
        component_model.load_state_dict(torch.load(final_checkpoint_path))
    else:
        logger.warning(f"Checkpoint not found at {final_checkpoint_path}, using untrained model")
    
    component_model.eval()
    
    # Generate a test batch with fixed seed for reproducibility
    set_seed(42)  # Use same seed as verification
    test_batch = dataset.generate_batch(config.eval_batch_size)
    test_input = test_batch[0] if isinstance(test_batch, tuple) else test_batch["input"]
    test_input = test_input.to(device)
    
    # Get target output (unmasked) and compute CI
    with torch.no_grad():
        target_output_cache = component_model(test_input, cache_type="input")
        target_output = target_output_cache.output if hasattr(target_output_cache, 'output') else target_output_cache
        ci = component_model.calc_causal_importances(
            pre_weight_acts=target_output_cache.cache,
            detach_inputs=False,
            sampling=config.sampling,
        )
    
    # Run verification
    verification_results = verify_assumption2(
        component_model=component_model,
        batch=test_input,
        target_out=target_output,
        ci=ci,
        device=device,
        config=config,
        step=config.steps,
        out_dir=out_dir,
        max_components_per_module=100,
        seed=42,
    )
    
    # Log results to wandb
    if config.wandb_project:
        wandb_verification_logs = {}
        for k, v in verification_results.items():
            if isinstance(v, (int, float)):
                wandb_verification_logs[f"assumption2/{k}"] = v
            elif isinstance(v, Path):
                # Log plot images to wandb
                wandb_verification_logs[f"assumption2/{k}"] = wandb.Image(str(v))
        wandb.log(wandb_verification_logs, step=config.steps)
        logger.info(f"✓ Logged Assumption 2 results to wandb")

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)