from abc import ABC, abstractmethod
import torch
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional


@dataclass
class LossOutput:
    """Container for loss calculation outputs"""
    loss: torch.Tensor
    metrics: Dict[str, torch.Tensor]


class BaseAuxiliaryLoss(ABC):
    """Abstract base class for auxiliary loss components"""
    
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    @abstractmethod
    def compute_loss(self, velocities: Dict[str, torch.Tensor], inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the auxiliary loss"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this auxiliary loss for logging"""
        pass
    
    @staticmethod
    def _reconstruct_clean_sample(scheduler, noisy_sample: torch.Tensor, model_prediction: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the original clean sample from the noisy sample, model prediction, and timestep,
        based on the FlowMatchScheduler's noise schedule and target definition.

        Args:
            scheduler: An instance of the FlowMatchScheduler.
            noisy_sample: The noisy sample (e.g., noisy_x_win).
            model_prediction: The model's prediction for the training target (e.g., velocity_win_pred).
            timestep: The timestep at which noise was added.

        Returns:
            The reconstructed clean sample.
        """
        # Ensure scheduler.timesteps is on the same device as timestep
        scheduler_timesteps = scheduler.timesteps.to(timestep.device)
        timestep_id = torch.argmin((scheduler_timesteps - timestep).abs())
        
        sigma = scheduler.sigmas[timestep_id]
        
        if noisy_sample.is_cuda:
            sigma = sigma.to(noisy_sample.device)

        x_clean = noisy_sample - sigma * model_prediction
        
        return x_clean


class StaticPenaltyLoss(BaseAuxiliaryLoss):
    """Static penalty auxiliary loss"""
    
    def get_name(self) -> str:
        return "static_penalty"
    
    def compute_loss(self, velocities: Dict[str, torch.Tensor], inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate static penalty loss based on reconstructed clean content.
        Args:
            velocities: Dictionary containing velocity predictions  
            inputs: Dictionary containing scheduler, noisy samples, timesteps
            
        Returns:
            Static penalty loss (higher penalty for more static content)
        """
        v_pred = velocities['win']  # [B, C, T, H, W]
        x_clean = self._reconstruct_clean_sample(
            inputs['scheduler'], 
            inputs['noisy_x_win'], 
            v_pred, 
            inputs['timestep']
        )
        
        # Temporal variance penalty (primary component)
        # Based on experiments: static ~0.045, dynamic ~0.194
        temporal_var = torch.var(x_clean, dim=2, unbiased=False)  # [B, C, H, W]
        temporal_penalty = -torch.mean(temporal_var)
        
        # Frame difference penalty (secondary component)
        # Based on experiments: static ~0.127, dynamic ~0.270
        if x_clean.size(2) > 1:
            frame_diffs = x_clean[:, :, 1:] - x_clean[:, :, :-1]  # [B, C, T-1, H, W]
            diff_magnitude = torch.mean(frame_diffs ** 2)
            frame_penalty = -diff_magnitude
        else:
            frame_penalty = torch.tensor(0.0, device=x_clean.device)
        
        # Combined penalty (70% temporal variance + 30% frame difference)
        # Weights based on experimental effectiveness
        penalty = 0.7 * temporal_penalty + 0.3 * frame_penalty
        
        return penalty




class BaseLossStrategy(ABC):
    """Abstract base class for loss calculation strategies"""
    
    def __init__(self, auxiliary_losses: Optional[List[BaseAuxiliaryLoss]] = None):
        self.auxiliary_losses = auxiliary_losses or []

    def _apply_auxiliary_losses(self, base_loss: torch.Tensor, velocities: Dict[str, torch.Tensor], 
                               inputs: Dict[str, torch.Tensor], base_metrics: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply all auxiliary losses to base loss"""
        final_loss = base_loss
        final_metrics = base_metrics.copy()
        
        for aux_loss in self.auxiliary_losses:
            if aux_loss.weight > 0:
                aux_loss_value = aux_loss.compute_loss(velocities, inputs)
                final_loss = final_loss + aux_loss.weight * aux_loss_value
                final_metrics[aux_loss.get_name()] = aux_loss_value
        
        return final_loss, final_metrics
    
    @abstractmethod
    def _calculate_base_loss(self, velocities: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate the core loss and base metrics based on the provided velocities and inputs.
        
        Args:
            velocities: Dictionary containing velocity tensors and metrics.
        
        Returns:
            A tuple containing the computed base_loss and a dictionary of base_metrics.
        """
        pass

    def calculate_loss(self, velocities: Dict[str, torch.Tensor], inputs: Dict[str, torch.Tensor]) -> LossOutput:
        """Calculate final loss including auxiliary losses.
        
        Args:
            velocities: Dictionary containing velocity tensors and metrics.
            inputs: Dictionary containing input tensors.
        
        Returns:
            LossOutput containing the computed final loss and relevant metrics.
        """
        base_loss, base_metrics = self._calculate_base_loss(velocities)
        final_loss, final_metrics = self._apply_auxiliary_losses(base_loss, velocities, inputs, base_metrics)
        return LossOutput(loss=final_loss, metrics=final_metrics)


class SFTLossStrategy(BaseLossStrategy):
    """Supervised Fine-Tuning loss strategy"""
    
    def _calculate_base_loss(self, velocities: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        base_loss = torch.nn.functional.mse_loss(
            velocities['win'].float(), 
            velocities['win_target'].float()
        )
        base_loss = base_loss.mean() * velocities["win_metric"].mean()
        
        return base_loss, {}


class DROLossStrategy(BaseLossStrategy):
    """Direct Reward Optimization loss strategy"""
    
    def _calculate_base_loss(self, velocities: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        model_win_err = (velocities['win'] - velocities['win_target']).pow(2).mean(dim=[1, 2, 3, 4])
        model_lose_err = (velocities['lose'] - velocities['lose_target']).pow(2).mean(dim=[1, 2, 3, 4])
        
        base_loss = model_win_err - model_lose_err
        base_loss = base_loss.mean()
        
        return base_loss, {}


class DPOLossStrategy(BaseLossStrategy):
    """Direct Preference Optimization loss strategy"""
    
    def __init__(self, beta: float = 500, auxiliary_losses: Optional[List[BaseAuxiliaryLoss]] = None):
        super().__init__(auxiliary_losses)
        self.beta = beta
    
    def _calculate_base_loss(self, velocities: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Calculate squared errors for model predictions
        model_win_err = (velocities['win'] - velocities['win_target']).pow(2).mean(dim=[1, 2, 3, 4])
        model_lose_err = (velocities['lose'] - velocities['lose_target']).pow(2).mean(dim=[1, 2, 3, 4])
        
        # Calculate squared errors for reference model
        ref_win_err = (velocities['win_ref'] - velocities['win_target']).pow(2).mean(dim=[1, 2, 3, 4])
        ref_lose_err = (velocities['lose_ref'] - velocities['lose_target']).pow(2).mean(dim=[1, 2, 3, 4])
        
        # Calculate differences
        win_diff = model_win_err - ref_win_err
        lose_diff = model_lose_err - ref_lose_err
        
        # Calculate DPO loss
        inside_term = -0.5 * self.beta * (win_diff - lose_diff)
        base_loss = -torch.nn.functional.logsigmoid(inside_term).mean()
        
        base_metrics = {
            "win_diff": win_diff.mean(),
            "lose_diff": lose_diff.mean()
        }
        
        return base_loss, base_metrics


def create_loss_strategy(strategy: str, beta: float = 500, static_penalty_lambda: float = 0.0, 
                        inlier_regression_lambda: float = 0.0, inlier_model_path: Optional[str] = None) -> BaseLossStrategy:
    """Factory function to create the appropriate loss strategy with auxiliary losses
    
    Args:
        strategy: One of 'sft', 'dro', or 'dpo'
        beta: Beta parameter for DPO loss (default: 500)
        static_penalty_lambda: Weight for static penalty term (default: 0.0)
        inlier_regression_lambda: Deprecated, not used
        inlier_model_path: Deprecated, not used
    
    Returns:
        An instance of the appropriate loss strategy
    """
    auxiliary_losses = []
    
    if static_penalty_lambda > 0:
        auxiliary_losses.append(StaticPenaltyLoss(weight=static_penalty_lambda))
    
    strategies = {
        'sft': lambda: SFTLossStrategy(auxiliary_losses=auxiliary_losses),
        'dro': lambda: DROLossStrategy(auxiliary_losses=auxiliary_losses),
        'dpo': lambda: DPOLossStrategy(beta=beta, auxiliary_losses=auxiliary_losses)
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Must be one of {list(strategies.keys())}")
    
    return strategies[strategy]() 
