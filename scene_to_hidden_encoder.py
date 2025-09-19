"""
Scene-to-Hidden State Encoder for SPRING-HardNet Integration

This module provides the critical final piece of the SPRING architecture restoration.
Converts scene features into scene-specific GRU hidden state initializations,
replacing the shared hidden state that was preventing scene-specific spatial reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SceneToHiddenEncoder(nn.Module):
    """
    Neural network that converts scene features to scene-specific GRU hidden states.
    
    This solves the final architectural bottleneck where all samples shared the same
    hidden state initialization via .expand(), causing the GRU to ignore varied inputs
    and produce nearly identical outputs regardless of scene content.
    
    Architecture:
        scene_features [batch_size, scene_dim] 
        → linear projection [batch_size, num_layers * hidden_size]
        → reshape to [num_layers, batch_size, hidden_size]
        → normalize to reasonable hidden state ranges
    """
    
    def __init__(
        self,
        scene_feature_dim: int = 500,
        num_layers: int = 3,
        hidden_size: int = 500,
        initialization_scale: float = 0.1,
        activation: str = "tanh"
    ):
        super().__init__()
        
        self.scene_feature_dim = scene_feature_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.initialization_scale = initialization_scale
        
        # Single linear projection for simplicity and efficiency
        self.projection = nn.Linear(
            scene_feature_dim, 
            num_layers * hidden_size
        )
        
        # Optional activation for hidden state range control
        if activation == "tanh":
            self.activation = nn.Tanh()  # Keeps values in [-1, 1]
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()  # Keeps values in [0, 1]
        elif activation == "relu":
            self.activation = nn.ReLU()  # Keeps values >= 0
        else:
            self.activation = None  # No activation (linear)
            
        # Initialize for stable hidden state generation
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for stable hidden state generation."""
        # Small weight initialization to prevent extreme hidden states
        nn.init.xavier_uniform_(
            self.projection.weight, 
            gain=self.initialization_scale
        )
        
        # Zero bias initialization - let weights control the hidden state
        nn.init.zeros_(self.projection.bias)
        
    def forward(self, scene_features: torch.Tensor) -> torch.Tensor:
        """
        Convert scene features to GRU hidden state initialization.
        
        Args:
            scene_features: Scene encoding [batch_size, scene_feature_dim]
            
        Returns:
            Hidden states [num_layers, batch_size, hidden_size] for GRU initialization
        """
        batch_size = scene_features.shape[0]
        
        # Project scene features to hidden state space
        flat_hidden = self.projection(scene_features)
        
        # Apply activation if specified
        if self.activation is not None:
            flat_hidden = self.activation(flat_hidden)
            
        # Reshape to GRU hidden state format [num_layers, batch_size, hidden_size]
        hidden_states = flat_hidden.view(self.num_layers, batch_size, self.hidden_size)
        
        return hidden_states
        
    def get_debug_info(self, scene_features: torch.Tensor) -> dict:
        """Get debugging information about hidden state generation."""
        with torch.no_grad():
            batch_size = scene_features.shape[0]
            
            # Forward pass with intermediate values
            flat_hidden = self.projection(scene_features)
            if self.activation is not None:
                flat_hidden_activated = self.activation(flat_hidden)
            else:
                flat_hidden_activated = flat_hidden
                
            hidden_states = flat_hidden_activated.view(self.num_layers, batch_size, self.hidden_size)
            
            return {
                'input_stats': {
                    'mean': scene_features.mean().item(),
                    'std': scene_features.std().item(),
                    'range': [scene_features.min().item(), scene_features.max().item()]
                },
                'projection_stats': {
                    'mean': flat_hidden.mean().item(),
                    'std': flat_hidden.std().item(),
                    'range': [flat_hidden.min().item(), flat_hidden.max().item()]
                },
                'hidden_states_stats': {
                    'mean': hidden_states.mean().item(),
                    'std': hidden_states.std().item(),
                    'range': [hidden_states.min().item(), hidden_states.max().item()]
                },
                'diversity_across_batch': {
                    'layer_0_std': hidden_states[0].std().item(),
                    'layer_1_std': hidden_states[1].std().item(),
                    'layer_2_std': hidden_states[2].std().item(),
                },
                'hidden_state_shape': list(hidden_states.shape)
            }


def test_scene_to_hidden_encoder():
    """Test the scene-to-hidden encoder with various configurations."""
    print("=== Testing Scene-to-Hidden Encoder ===")
    
    # Test configuration
    batch_size = 16
    scene_feature_dim = 500
    num_layers = 3
    hidden_size = 500
    
    # Test different activation functions
    activations = [None, "tanh", "sigmoid", "relu"]
    
    for activation in activations:
        print(f"\n--- Testing with activation: {activation} ---")
        
        # Create encoder
        encoder = SceneToHiddenEncoder(
            scene_feature_dim=scene_feature_dim,
            num_layers=num_layers,
            hidden_size=hidden_size,
            initialization_scale=0.1,
            activation=activation
        )
        
        print(f"✓ Encoder created: {scene_feature_dim} → [{num_layers}, {hidden_size}] hidden states")
        
        # Test with realistic scene features (similar to what we observed)
        scene_features = torch.randn(batch_size, scene_feature_dim) * 0.5 + 0.02  # Similar to training data
        
        # Forward pass
        hidden_states = encoder(scene_features)
        debug_info = encoder.get_debug_info(scene_features)
        
        print(f"Input: {scene_features.shape} → Output: {hidden_states.shape}")
        print(f"Hidden state range: [{hidden_states.min():.3f}, {hidden_states.max():.3f}]")
        print(f"Hidden state diversity: std={hidden_states.std():.3f}")
        
        # Check diversity across batch (different samples should have different hidden states)
        sample_0_mean = hidden_states[:, 0, :].mean().item()
        sample_1_mean = hidden_states[:, 1, :].mean().item()
        print(f"Sample diversity: sample_0_mean={sample_0_mean:.3f}, sample_1_mean={sample_1_mean:.3f}")
        
        # Verify output properties
        assert hidden_states.shape == (num_layers, batch_size, hidden_size), f"Wrong output shape: {hidden_states.shape}"
        
        # Check that different scenes produce different hidden states
        scene_diff = (hidden_states[:, 0, :] - hidden_states[:, 1, :]).abs().mean().item()
        assert scene_diff > 0.001, f"Hidden states too similar across samples: diff={scene_diff}"
        
        # Check hidden state magnitudes are reasonable
        hidden_magnitude = hidden_states.abs().mean().item()
        assert 0.001 < hidden_magnitude < 10.0, f"Hidden state magnitude out of range: {hidden_magnitude}"
        
        print("✓ All assertions passed")
        print(f"✓ Batch diversity confirmed: cross-sample difference = {scene_diff:.4f}")
    
    print("\n Scene-to-Hidden Encoder Test Complete")
    print(" Ready for integration into SPRING architecture")


if __name__ == "__main__":
    test_scene_to_hidden_encoder()