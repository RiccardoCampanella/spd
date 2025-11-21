"""Run spd on a TMS model.

Note that the first instance index is fixed to the identity matrix. This is done so we can compare
the losses of the "correct" solution during training.
"""

import json
from pathlib import Path
from typing import Any

import fire
from spd.spd.models.component_model import ComponentModel
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
    Verify Assumption 2: Test if gradient s_i = ∂L/∂m_i predicts ablation effect Δ(i).
    
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
            # Compute Δ(i): Ablation effect
            # ====================================================================
            # Create mask with only component i ablated (r^(i))
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
            # Compute s_i: Gradient of layerwise loss w.r.t. m_i
            # ====================================================================
            # For layerwise loss, we need to compute L_stochastic-recon-layerwise^(l)
            # where l is the layer containing component i
            
            # Create component masks with gradient enabled for component i
            component_masks_grad = {}
            for mod_name, mask_vals_mod in ci.lower_leaky.items():
                comp_mask = mask_vals_mod.clone().detach()
                if mod_name == module_name:
                    comp_mask.requires_grad_(True)
                component_masks_grad[mod_name] = comp_mask
            
            mask_infos_grad = make_mask_infos(
                component_masks=component_masks_grad,
                routing_masks="all",
                weight_deltas_and_masks=None
            )
            
            # Forward pass with only layer l masked (layerwise loss)
            # This matches L_stochastic-recon-layerwise^(l) from the paper
            out_layerwise = component_model(batch, mask_infos={module_name: mask_infos_grad[module_name]})
            
            # Compute layerwise loss
            loss_layerwise = calc_sum_recon_loss_lm(
                pred=out_layerwise,
                target=target_out,
                loss_type=config.output_loss_type
            )
            
            # Get gradient w.r.t. component mask m_i
            comp_mask = mask_infos_grad[module_name].component_mask
            if not comp_mask.requires_grad:
                s_i = 0.0
            else:
                grad = torch.autograd.grad(
                    loss_layerwise,
                    comp_mask,
                    allow_unused=True,
                    retain_graph=False
                )[0]
                if grad is None:
                    s_i = 0.0
                else:
                    # Extract gradient for component i and average
                    s_i = grad[:, c].mean().item()
            
            s_list.append(s_i)
    
    # ====================================================================
    # Compute correlations
    # ====================================================================
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
    # ====================================================================
    stage_name = "initialization" if step == 0 else ("middle" if step < config.steps else "end")
    plots = {}
    
    # Plot 1: s_i vs Δ(i)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(s_list, delta_list, s=20, alpha=0.6, 
              edgecolors='steelblue', linewidths=0.5, facecolors='steelblue')
    ax.set_xlabel("s_i = ∂L/∂m_i (gradient)", fontsize=13)
    ax.set_ylabel("Δ(i) = MSE increase from ablation", fontsize=13)
    ax.set_title(f"Assumption 2: Gradient vs Ablation Effect ({stage_name}, step {step})\n"
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
    ax.set_xlabel("|s_i| = |∂L/∂m_i|", fontsize=13)
    ax.set_ylabel("Δ(i) = MSE increase from ablation", fontsize=13)
    ax.set_title(f"Assumption 2: Absolute Gradient vs Ablation Effect ({stage_name}, step {step})\n"
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
    
    # Get the final model state (from the last checkpoint or current state)
    # We need to reload the model to ensure we have the trained weights
    if out_dir and (out_dir / f"model_{config.steps}.pth").exists():
        logger.info(f"Loading final model from {out_dir / f'model_{config.steps}.pth'}")
        final_model_state = load_file(out_dir / f"model_{config.steps}.pth")
        # Reconstruct component model and load state
        # (This depends on how your model saving/loading works)
    
    # For now, we'll use the model from optimize - but we need to access it
    # The cleanest way is to modify optimize() to return the model, or we can
    # reload it. Let's create a helper that runs verification on a model.
    
    # Generate a test batch with fixed seed for reproducibility
    set_seed(42)  # Use same seed as verification
    test_batch = dataset.generate_batch(config.eval_batch_size)
    test_input = test_batch[0] if isinstance(test_batch, tuple) else test_batch["input"]
    test_input = test_input.to(device)
    
    # Get target output (unmasked)
    with torch.no_grad():
        target_output = target_model(test_input)

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
