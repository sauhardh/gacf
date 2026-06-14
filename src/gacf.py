import torch
import torch.nn as nn


class GACFController:
    """
    Gradient-Aware Channel Freezing (GACF) Controller.
    
    This class non-intrusively attaches to a PyTorch model to:
    1. Hook into backward passes to measure channel-wise gradient importance.
    2. Maintain an Exponential Moving Average (EMA) of these importances.
    3. Gradually freeze a percentage of channels by zeroing their weight gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        p_max: float = 0.5,
        t_freeze: int = 15,
        beta: float = 0.9,
    ):
        self.model = model
        self.p_max = p_max        # Maximum percentage of channels to freeze
        self.t_freeze = t_freeze  # Number of epochs over which to ramp up freezing
        self.beta = beta          # EMA decay factor

        self.conv_layers = {}
        self.ema_buffers = {}
        self.freeze_masks = {}    # Boolean masks: True means frozen
        self.hooks = []

        self._register_layers()

    def _register_layers(self):
        """Finds all Conv2d layers and registers backward hooks to capture gradients."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels > 1:
                self.conv_layers[name] = module
                # Initialize EMA buffer with zeros for each output channel
                self.ema_buffers[name] = torch.zeros(module.out_channels, device=module.weight.device)
                self.freeze_masks[name] = torch.zeros(module.out_channels, dtype=torch.bool, device=module.weight.device)
                
                # Register the backward hook to score importance
                hook = module.register_full_backward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, layer_name: str):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is None:
                return
            
            # grad_output[0] shape: (Batch, Channels, Height, Width)
            g = grad_output[0].detach()
            
            # 1. Importance Scorer: Squared L2 norm across spatial (H, W) and batch (N) dimensions
            # This leaves us with a 1D tensor of length = Channels
            current_score = (g ** 2).sum(dim=(0, 2, 3))
            
            # 2. EMA Buffer: Update running estimates
            # ema = beta * ema + (1 - beta) * current
            self.ema_buffers[layer_name] = (
                self.beta * self.ema_buffers[layer_name] + 
                (1.0 - self.beta) * current_score
            )
            
        return hook

    def step_epoch(self, current_epoch: int):
        """
        Called at the start of each epoch to determine which channels to freeze.
        Implements the linear ramp from 0% to Pmax over Tfreeze epochs.
        """
        # Calculate current target freeze percentage
        if current_epoch >= self.t_freeze:
            current_p = self.p_max
        else:
            # Linear ramp: e.g. epoch 0 -> 0%, epoch T -> Pmax
            current_p = self.p_max * (current_epoch / self.t_freeze)

        total_frozen = 0
        total_channels = 0

        # Rank and freeze channels per layer
        for name, module in self.conv_layers.items():
            num_channels = module.out_channels
            num_to_freeze = int(num_channels * current_p)
            
            total_channels += num_channels
            total_frozen += num_to_freeze

            if num_to_freeze == 0:
                self.freeze_masks[name].fill_(False)
                continue

            # Get EMA scores
            scores = self.ema_buffers[name]
            
            # Rank channels by score (ascending: lowest scores are least important)
            # We want to freeze the LEAST important channels
            _, sorted_indices = torch.sort(scores, descending=False)
            
            # Select the bottom 'num_to_freeze' channels
            channels_to_freeze = sorted_indices[:num_to_freeze]
            
            # Update the boolean mask
            mask = torch.zeros(num_channels, dtype=torch.bool, device=module.weight.device)
            mask[channels_to_freeze] = True
            self.freeze_masks[name] = mask

        print(f"[GACF] Epoch {current_epoch} | Target P: {current_p*100:.1f}% | "
              f"Frozen: {total_frozen}/{total_channels} channels")

    def zero_frozen_gradients(self):
        """
        Must be called IMMEDIATELY after `loss.backward()` and before `optimizer.step()`.
        This zeroes out the weight gradients of the frozen channels, preventing them from updating.
        """
        for name, module in self.conv_layers.items():
            mask = self.freeze_masks[name]
            
            # If no channels are frozen in this layer, skip
            if not mask.any():
                continue

            # Zero out gradients for the frozen output channels (dim 0 of weight tensor)
            if module.weight.grad is not None:
                module.weight.grad[mask, ...] = 0.0
                
            if module.bias is not None and module.bias.grad is not None:
                module.bias.grad[mask] = 0.0

    def remove_hooks(self):
        """Cleanup function to remove hooks if needed."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
