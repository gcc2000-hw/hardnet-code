"""
Scene-to-Layout Encoder for SPRING-HardNet Integration

This module provides the critical missing link between scene understanding and spatial reasoning.
Converts ResNet18 scene features into layout coordinate predictions for the GRU spatial reasoning module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SceneToLayoutEncoder(nn.Module):
    """
    Neural network that converts scene features to initial layout coordinate predictions.
    
    This solves the critical architectural gap in SPRING-HardNet where scene features
    from ResNet18 were computed but never passed to the spatial reasoning module.
    
    Architecture:
        scene_features [batch_size, 512] 
        → projection_1 [batch_size, 1024] (expand feature space)
        → relu + dropout
        → projection_2 [batch_size, max_objects * 4] (coordinate predictions) 
        → reshape to [batch_size, max_objects, 4]
        → normalize to per-mille coordinate system [0, 1000]
    """
    
    def __init__(
        self,
        scene_feature_dim: int = 512,
        max_objects: int = 10,
        coordinate_dim: int = 4,
        hidden_dim: int = 1024,
        dropout_rate: float = 0.1,
        coordinate_system: str = "per_mille"
    ):
        super().__init__()
        
        self.scene_feature_dim = scene_feature_dim
        self.max_objects = max_objects
        self.coordinate_dim = coordinate_dim
        self.hidden_dim = hidden_dim
        self.coordinate_system = coordinate_system
        
        # Two-stage projection for sophisticated scene-to-layout mapping
        self.projection_1 = nn.Linear(scene_feature_dim, hidden_dim)
        self.projection_2 = nn.Linear(hidden_dim, max_objects * coordinate_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        
        # Initialize for coordinate prediction in per-mille system [0, 1000]
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for coordinate prediction in per-mille system."""
        # First projection: Standard Xavier for feature expansion
        nn.init.xavier_uniform_(self.projection_1.weight, gain=1.0)
        nn.init.zeros_(self.projection_1.bias)
        
        # Second projection: Specialized for coordinate outputs
        # Weight initialization for coordinate range [0, 1000]
        nn.init.xavier_uniform_(self.projection_2.weight, gain=0.5)  # Smaller gain for coordinate stability
        
        # Bias initialization: Center coordinates around middle of canvas
        # x, y centered around 500 (middle of 1000), w, h around 100 (reasonable object size)
        coordinate_bias = torch.zeros(self.max_objects * self.coordinate_dim)
        for i in range(self.max_objects):
            base_idx = i * 4
            coordinate_bias[base_idx:base_idx + 2] = 500.0  # x, y center
            coordinate_bias[base_idx + 2:base_idx + 4] = 100.0  # w, h reasonable size
            
        self.projection_2.bias.data = coordinate_bias
        
    def forward(self, scene_features: torch.Tensor) -> torch.Tensor:
        """
        Convert scene features to layout coordinate predictions.
        
        Args:
            scene_features: Scene encoding from ResNet18 [batch_size, scene_feature_dim]
            
        Returns:
            Layout coordinates [batch_size, max_objects, coordinate_dim] in per-mille system
        """
        batch_size = scene_features.shape[0]
        
        # Two-stage projection: scene features → expanded features → coordinates
        expanded_features = self.activation(self.projection_1(scene_features))
        expanded_features = self.dropout(expanded_features)
        
        # Project to coordinate space
        flat_coordinates = self.projection_2(expanded_features)
        
        # Reshape to object-coordinate format
        layout_coordinates = flat_coordinates.view(batch_size, self.max_objects, self.coordinate_dim)
        
        # Normalize to coordinate system range
        if self.coordinate_system == "per_mille":
            # Clamp to valid per-mille range [0, 1000]
            layout_coordinates = torch.clamp(layout_coordinates, min=0.0, max=1000.0)
        elif self.coordinate_system == "normalized":
            # Apply sigmoid for normalized range [0, 1]
            layout_coordinates = torch.sigmoid(layout_coordinates)
        else:
            # Raw coordinates - model decides the range
            pass
            
        return layout_coordinates
        
    def get_debug_info(self, scene_features: torch.Tensor) -> dict:
        """Get debugging information about the encoding process."""
        with torch.no_grad():
            batch_size = scene_features.shape[0]
            
            # Forward pass with intermediate values
            expanded_features = self.activation(self.projection_1(scene_features))
            flat_coordinates = self.projection_2(expanded_features)
            layout_coordinates = flat_coordinates.view(batch_size, self.max_objects, self.coordinate_dim)
            
            return {
                'input_stats': {
                    'mean': scene_features.mean().item(),
                    'std': scene_features.std().item(),
                    'range': [scene_features.min().item(), scene_features.max().item()]
                },
                'expanded_features_stats': {
                    'mean': expanded_features.mean().item(),
                    'std': expanded_features.std().item(),
                    'range': [expanded_features.min().item(), expanded_features.max().item()]
                },
                'output_stats': {
                    'mean': layout_coordinates.mean().item(),
                    'std': layout_coordinates.std().item(),
                    'range': [layout_coordinates.min().item(), layout_coordinates.max().item()]
                },
                'coordinate_diversity': {
                    'x_range': [layout_coordinates[:, :, 0].min().item(), layout_coordinates[:, :, 0].max().item()],
                    'y_range': [layout_coordinates[:, :, 1].min().item(), layout_coordinates[:, :, 1].max().item()],
                    'w_range': [layout_coordinates[:, :, 2].min().item(), layout_coordinates[:, :, 2].max().item()],
                    'h_range': [layout_coordinates[:, :, 3].min().item(), layout_coordinates[:, :, 3].max().item()]
                }
            }


def test_scene_to_layout_encoder():
    """Test the scene-to-layout encoder with various inputs."""
    print("=== Testing Scene-to-Layout Encoder ===")
    
    # Test configuration
    batch_size = 4
    scene_feature_dim = 512
    max_objects = 10
    
    # Create encoder
    encoder = SceneToLayoutEncoder(
        scene_feature_dim=scene_feature_dim,
        max_objects=max_objects,
        coordinate_system="per_mille"
    )
    
    print(f"✓ Encoder created: {scene_feature_dim} → {max_objects}×4 coordinates")
    
    # Test with various scene features
    test_cases = [
        ("Random scene features", torch.randn(batch_size, scene_feature_dim)),
        ("Zero scene features", torch.zeros(batch_size, scene_feature_dim)),  
        ("Positive scene features", torch.abs(torch.randn(batch_size, scene_feature_dim))),
        ("Negative scene features", -torch.abs(torch.randn(batch_size, scene_feature_dim)))
    ]
    
    for test_name, scene_features in test_cases:
        print(f"\n--- {test_name} ---")
        
        # Forward pass
        layout_coords = encoder(scene_features)
        debug_info = encoder.get_debug_info(scene_features)
        
        print(f"Input: {scene_features.shape} → Output: {layout_coords.shape}")
        print(f"Output range: [{layout_coords.min():.1f}, {layout_coords.max():.1f}]")
        print(f"Output diversity: std={layout_coords.std():.1f}")
        print(f"Coordinate ranges: x[{debug_info['coordinate_diversity']['x_range'][0]:.0f}, {debug_info['coordinate_diversity']['x_range'][1]:.0f}], "
              f"y[{debug_info['coordinate_diversity']['y_range'][0]:.0f}, {debug_info['coordinate_diversity']['y_range'][1]:.0f}]")
        
        # Verify output properties
        assert layout_coords.shape == (batch_size, max_objects, 4), f"Wrong output shape: {layout_coords.shape}"
        assert layout_coords.min() >= 0.0, f"Coordinates below 0: {layout_coords.min()}"
        assert layout_coords.max() <= 1000.0, f"Coordinates above 1000: {layout_coords.max()}"
        
        print("✓ All assertions passed")
    
    print("\n Scene-to-Layout Encoder Test Complete")


if __name__ == "__main__":
    test_scene_to_layout_encoder()