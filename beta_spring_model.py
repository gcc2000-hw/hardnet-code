"""
Beta SPRING Model Configuration and Wrapper

This provides the model interface compatible with the hardnet inference pipeline
while using the Beta SPRING probabilistic spatial reasoning approach.

Key Components:
- BetaSpringConfig: Configuration matching SpringHybridConfig interface
- BetaSpringModel: Model wrapper around BetaSpatialReasonerComplete
- DeploymentMode/SpatialReasoningMode: Enums for compatibility

Architecture:
- Wraps BetaSpatialReasonerComplete with SpringHybridModel-compatible interface
- Handles coordinate system conversion from Beta [0,1] to pixels
- Integrates perception module (DETR) for background object detection
- Provides same forward() interface as SpringHybridModel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional
import logging
from PIL import Image
import numpy as np

# Beta SPRING core components
from beta_spatial_complete import BetaSpatialReasonerComplete

# Create simplified DETR wrapper to avoid import issues
try:
    from transformers import DetrFeatureExtractor, DetrForObjectDetection
    DETR_AVAILABLE = True
except ImportError:
    DETR_AVAILABLE = False
    print("Warning: DETR not available. Perception module will be disabled.")

logger = logging.getLogger(__name__)


class SimpleDETRWrapper:
    """
    Simplified DETR wrapper for perception module
    
    This provides a basic object detection interface without dependencies
    on missing coco_dataset functions.
    """
    
    def __init__(self):
        if not DETR_AVAILABLE:
            raise ImportError("DETR (transformers) not available")
            
        self.feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.model.eval()
    
    def __call__(self, images: torch.Tensor) -> Dict[str, Any]:
        """
        Detect objects in images
        
        Args:
            images: [batch_size, 3, H, W] tensor
            
        Returns:
            Dict with detection results matching expected format
        """
        batch_size = images.size(0)
        device = images.device
        
        # Convert tensor to PIL images for DETR
        pil_images = []
        for i in range(batch_size):
            # Convert tensor [3, H, W] to PIL
            img_tensor = images[i].cpu()
            img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
            pil_img = Image.fromarray(img_array)
            pil_images.append(pil_img)
        
        results = {
            'detected_objects': torch.zeros(batch_size, 5, 4),  # [batch, max_objects, 4]
            'detection_info': {
                'per_sample': []
            }
        }
        
        try:
            with torch.no_grad():
                for i, pil_img in enumerate(pil_images):
                    # Process single image
                    inputs = self.feature_extractor(images=pil_img, return_tensors="pt")
                    outputs = self.model(**inputs)
                    
                    # Post-process detections
                    target_sizes = torch.tensor([pil_img.size[::-1]])
                    detections = self.feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]
                    
                    # Filter confident detections
                    confidence_threshold = 0.7
                    confident_mask = detections["scores"] > confidence_threshold
                    
                    if confident_mask.sum() > 0:
                        boxes = detections["boxes"][confident_mask]
                        scores = detections["scores"][confident_mask]
                        labels = detections["labels"][confident_mask]
                        
                        # Convert to our format: [x, y, w, h]
                        num_detections = min(len(boxes), 5)  # Max 5 objects
                        for j in range(num_detections):
                            x1, y1, x2, y2 = boxes[j].tolist()
                            results['detected_objects'][i, j] = torch.tensor([x1, y1, x2-x1, y2-y1])
                        
                        # Store detection info
                        results['detection_info']['per_sample'].append({
                            'confidence_scores': scores[:5].tolist(),
                            'category_names': [f"object_{label.item()}" for label in labels[:5]]
                        })
                    else:
                        # No confident detections
                        results['detection_info']['per_sample'].append({
                            'confidence_scores': [],
                            'category_names': []
                        })
        
        except Exception as e:
            logger.warning(f"DETR detection failed: {e}")
            # Return empty results
            for i in range(batch_size):
                results['detection_info']['per_sample'].append({
                    'confidence_scores': [],
                    'category_names': []
                })
        
        return results


class DeploymentMode(Enum):
    """Deployment mode enum matching hardnet interface"""
    TRAINING = "training"
    INFERENCE = "inference"
    VALIDATION = "validation"


class SpatialReasoningMode(Enum):
    """Spatial reasoning mode enum matching hardnet interface"""
    DIFFERENTIABLE = "differentiable"
    DISCRETE = "discrete"
    HYBRID = "hybrid"


class BetaSpringConfig:
    """
    Configuration class matching SpringHybridConfig interface
    
    This provides all the same parameters and attributes as SpringHybridConfig
    but configured for Beta SPRING's probabilistic approach.
    """
    
    def __init__(self,
                 deployment_mode: DeploymentMode = DeploymentMode.INFERENCE,
                 device: str = "cuda",
                 image_size: Tuple[int, int] = (512, 512),
                 max_objects: int = 5,
                 mixed_precision: bool = False,
                 gradient_checkpointing: bool = False,
                 srm_mode: SpatialReasoningMode = SpatialReasoningMode.DIFFERENTIABLE,
                 enable_perception_module: bool = True,
                 enable_veg_module: bool = False,
                 # Beta SPRING specific parameters
                 scene_dim: int = 512,
                 hidden_dim: int = 256,
                 num_beta_samples: int = 100,
                 constraint_loss_weight: float = 1.0,
                 coordinate_loss_weight: float = 1.0,
                 uncertainty_threshold: float = 0.1):
        
        # Core parameters matching SpringHybridConfig interface
        self.deployment_mode = deployment_mode
        self.device = torch.device(device)
        self.image_size = image_size
        self.max_objects = max_objects
        self.mixed_precision = mixed_precision
        self.gradient_checkpointing = gradient_checkpointing
        self.srm_mode = srm_mode
        self.enable_perception_module = enable_perception_module
        self.enable_veg_module = enable_veg_module
        
        # Beta SPRING specific parameters
        self.scene_dim = scene_dim
        self.hidden_dim = hidden_dim
        self.num_beta_samples = num_beta_samples
        self.constraint_loss_weight = constraint_loss_weight
        self.coordinate_loss_weight = coordinate_loss_weight
        self.uncertainty_threshold = uncertainty_threshold
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if not isinstance(self.image_size, (tuple, list)) or len(self.image_size) != 2:
            raise ValueError(f"image_size must be tuple/list of 2 integers, got {self.image_size}")
        
        if self.max_objects <= 0:
            raise ValueError(f"max_objects must be positive, got {self.max_objects}")
        
        if self.scene_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError(f"scene_dim and hidden_dim must be positive")
        
        if self.num_beta_samples <= 0:
            raise ValueError(f"num_beta_samples must be positive, got {self.num_beta_samples}")
        
        logger.info(f"BetaSpringConfig validated: image_size={self.image_size}, "
                   f"max_objects={self.max_objects}, deployment_mode={self.deployment_mode.value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            'deployment_mode': self.deployment_mode.value,
            'device': str(self.device),
            'image_size': self.image_size,
            'max_objects': self.max_objects,
            'mixed_precision': self.mixed_precision,
            'gradient_checkpointing': self.gradient_checkpointing,
            'srm_mode': self.srm_mode.value,
            'enable_perception_module': self.enable_perception_module,
            'enable_veg_module': self.enable_veg_module,
            'scene_dim': self.scene_dim,
            'hidden_dim': self.hidden_dim,
            'num_beta_samples': self.num_beta_samples,
            'constraint_loss_weight': self.constraint_loss_weight,
            'coordinate_loss_weight': self.coordinate_loss_weight,
            'uncertainty_threshold': self.uncertainty_threshold
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BetaSpringConfig':
        """Create config from dictionary"""
        # Convert string enums back to enum objects
        deployment_mode = DeploymentMode(config_dict.get('deployment_mode', 'inference'))
        srm_mode = SpatialReasoningMode(config_dict.get('srm_mode', 'differentiable'))
        
        config_dict = config_dict.copy()
        config_dict['deployment_mode'] = deployment_mode
        config_dict['srm_mode'] = srm_mode
        
        return cls(**config_dict)


class BetaSpringModel(nn.Module):
    """
    Model wrapper around BetaSpatialReasonerComplete
    
    This provides the same interface as SpringHybridModel while using
    Beta SPRING's probabilistic spatial reasoning approach.
    
    Key Features:
    - Compatible forward() method matching SpringHybridModel
    - Coordinate system conversion from Beta [0,1] to pixels
    - Integrated perception module for background object detection
    - Constraint processing and satisfaction calculation
    - Proper gradient flow for end-to-end training
    """
    
    def __init__(self, config: BetaSpringConfig):
        super().__init__()
        
        self.config = config
        logger.info(f"Initializing BetaSpringModel with config: {config.to_dict()}")
        
        # Core Beta spatial reasoner
        self.beta_reasoner = BetaSpatialReasonerComplete(
            scene_dim=config.scene_dim,
            hidden_dim=config.hidden_dim,
            num_objects=config.max_objects,
            num_heads=8,
            input_channels=3,
            input_size=config.image_size[0],  # Use config image size
            coord_weight=config.coordinate_loss_weight,
            constraint_weight=config.constraint_loss_weight,
            uncertainty_weight=0.01,
            boundary_weight=0.05
        )
        
        # Perception module for background object detection (optional)
        if config.enable_perception_module and DETR_AVAILABLE:
            try:
                self.perception_module = SimpleDETRWrapper()
                logger.info("Perception module (SimpleDETRWrapper) enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize perception module: {e}")
                self.perception_module = None
        else:
            self.perception_module = None
            if config.enable_perception_module and not DETR_AVAILABLE:
                logger.info("Perception module requested but DETR not available")
            else:
                logger.info("Perception module disabled")
        
        # Move to device
        self.to(config.device)
        
        logger.info(f"BetaSpringModel initialized on {config.device}")
    
    def forward(self, 
                images: torch.Tensor,
                constraints: Optional[List[List]] = None,
                return_intermediate: bool = False,
                **kwargs) -> Dict[str, Any]:
        """
        Forward pass matching SpringHybridModel interface
        
        Args:
            images: Background images [batch_size, 3, H, W]
            constraints: List of constraint lists for each batch item
            return_intermediate: Whether to return intermediate results
            
        Returns:
            Dict containing:
                - coordinates: Generated object coordinates [batch_size, num_objects, 4]
                - constraint_satisfaction: Constraint satisfaction score
                - layout_results: Layout generation results (if return_intermediate)
                - perception_results: Background object detection results (if perception enabled)
        """
        batch_size = images.size(0)
        device = images.device
        
        results = {}
        
        # 1. Background object detection (if enabled)
        perception_results = None
        if self.perception_module is not None:
            try:
                with torch.no_grad():  # Perception is frozen during inference
                    perception_results = self.perception_module(images)
                    results['perception_results'] = perception_results
                    logger.debug(f"Perception detected objects in {batch_size} images")
            except Exception as e:
                logger.warning(f"Perception module failed: {e}")
                perception_results = None
        
        # 2. Beta spatial reasoning
        logger.debug(f"Running Beta spatial reasoning on {batch_size} images")
        
        # Convert constraints to Beta reasoner format if provided
        processed_constraints = None
        if constraints is not None:
            processed_constraints = self._process_constraints(constraints)
        
        # Generate coordinates using Beta reasoner
        beta_outputs = self.beta_reasoner(
            input_data=images,
            constraints=processed_constraints
        )
        
        # Extract coordinates and handle shape issues BEFORE conversion
        beta_coordinates = beta_outputs['coordinates']  # Expected: [batch_size, num_objects, 4]
        
        logger.debug(f"Raw Beta coordinates shape: {beta_coordinates.shape}")
        logger.debug(f"Raw Beta coordinates range: [{beta_coordinates.min():.3f}, {beta_coordinates.max():.3f}]")
        
        # CRITICAL FIX: Handle sampling dimension - BetaSpatialReasonerComplete returns [n_samples, batch, objects, 4]
        if beta_coordinates.dim() == 4 and beta_coordinates.size(0) > 1:
            logger.debug(f"Extracting single sample from {beta_coordinates.size(0)} Beta samples")
            # Take the first sample: [n_samples, batch, objects, 4] -> [batch, objects, 4]
            beta_coordinates = beta_coordinates[0]  
            logger.debug(f"Coordinates shape after sampling: {beta_coordinates.shape}")
        
        # Handle coordinate dimension mismatch (fallback for unusual cases)
        if beta_coordinates.size(-1) != 4:
            logger.warning(f"Beta coordinates have {beta_coordinates.size(-1)} dimensions, expected 4 [x,y,w,h]")
            
            if beta_coordinates.size(-1) == 3:
                # Common case: missing one dimension, likely width or height
                batch_size, num_objects, _ = beta_coordinates.shape
                # Add a 4th dimension (assume it's width/height with reasonable default)
                padding = torch.ones(batch_size, num_objects, 1, device=beta_coordinates.device) * 0.1  # 10% of space
                beta_coordinates = torch.cat([beta_coordinates, padding], dim=-1)
                logger.info(f"Added missing dimension: {beta_coordinates.shape}")
            elif beta_coordinates.size(-1) == 2:
                # Only x,y provided - add w,h
                batch_size, num_objects, _ = beta_coordinates.shape
                wh_padding = torch.ones(batch_size, num_objects, 2, device=beta_coordinates.device) * 0.1
                beta_coordinates = torch.cat([beta_coordinates, wh_padding], dim=-1) 
                logger.info(f"Added width/height dimensions: {beta_coordinates.shape}")
            else:
                raise ValueError(f"Cannot handle coordinate shape: {beta_coordinates.shape}")
        
        logger.debug(f"Final Beta coordinates shape: {beta_coordinates.shape}")
        
        # Now safely convert to pixel coordinates
        pixel_coordinates = self._convert_to_pixel_coordinates(beta_coordinates, images.shape[-2:])
        
        results['coordinates'] = pixel_coordinates
        results['predicted_coordinates'] = pixel_coordinates  # Alias for compatibility
        
        # 3. Constraint satisfaction calculation
        constraint_satisfaction = 0.0
        if constraints is not None and processed_constraints is not None:
            try:
                constraint_satisfaction = self._calculate_constraint_satisfaction(
                    beta_coordinates, processed_constraints
                )
            except Exception as e:
                logger.warning(f"Constraint satisfaction calculation failed: {e}")
                constraint_satisfaction = 0.0
        
        results['constraint_satisfaction'] = constraint_satisfaction
        
        # 4. Layout results (if requested)
        if return_intermediate:
            layout_results = {
                'final_layout': pixel_coordinates,
                'beta_coordinates': beta_coordinates,
                'beta_distributions': beta_outputs.get('distributions', None),
                'uncertainty': beta_outputs.get('uncertainty', None),
                'constraint_violations': beta_outputs.get('constraint_violations', None)
            }
            results['layout_results'] = layout_results
        
        logger.debug(f"BetaSpringModel forward complete: coordinates shape {pixel_coordinates.shape}")
        return results
    
    def _process_constraints(self, constraints: List[List]) -> List[List]:
        """
        Process constraints for Beta reasoner
        
        This converts the constraint format from the evaluation pipeline
        to the format expected by BetaSpatialReasonerComplete.
        """
        processed = []
        
        for batch_constraints in constraints:
            batch_processed = []
            
            for constraint in batch_constraints:
                # Convert constraint to Beta reasoner format
                # This will be expanded based on constraint_language_v2 integration
                try:
                    beta_constraint = self._convert_constraint_to_beta_format(constraint)
                    if beta_constraint is not None:
                        batch_processed.append(beta_constraint)
                except Exception as e:
                    logger.warning(f"Failed to convert constraint {constraint}: {e}")
                    continue
            
            processed.append(batch_processed)
        
        return processed
    
    def _convert_constraint_to_beta_format(self, constraint) -> Optional[Dict]:
        """
        Convert single constraint to Beta reasoner format
        
        This is a placeholder that will be expanded to handle all constraint types
        from constraint_language_v2.py
        """
        # TODO: Implement full constraint conversion
        # For now, return a basic format that Beta reasoner can handle
        
        if hasattr(constraint, '__dict__'):
            return {
                'type': constraint.__class__.__name__,
                'params': constraint.__dict__
            }
        elif isinstance(constraint, dict):
            return constraint
        else:
            logger.warning(f"Unknown constraint format: {type(constraint)}")
            return None
    
    def _convert_to_pixel_coordinates(self, 
                                    beta_coords: torch.Tensor, 
                                    image_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Convert Beta coordinates from per-mille [0,1000] to normalized [0,1] coordinates
        
        CRITICAL FIX: BetaSpatialReasonerComplete outputs coordinates in per-mille space [0,1000]
        but the pipeline expects normalized [0,1] coordinates.
        
        Args:
            beta_coords: [batch_size, num_objects, 4] in [0,1000] per-mille range
            image_shape: (height, width) of images
            
        Returns:
            Normalized coordinates [batch_size, num_objects, 4] in [0,1] range
        """
        height, width = image_shape
        
        # Clone to avoid modifying original
        normalized_coords = beta_coords.clone()
        
        # CRITICAL FIX: Convert from per-mille [0,1000] to normalized [0,1]
        normalized_coords = normalized_coords / 1000.0
        
        # Clamp to ensure valid [0,1] range
        normalized_coords = torch.clamp(normalized_coords, min=0.0, max=1.0)
        
        logger.debug(f"Converted coordinates from per-mille [0,1000] to normalized [0,1]: "
                    f"range [{normalized_coords.min():.3f}, {normalized_coords.max():.3f}]")
        
        return normalized_coords
    
    def _calculate_constraint_satisfaction(self, 
                                         coordinates: torch.Tensor,
                                         constraints: List[List]) -> float:
        """
        Calculate constraint satisfaction rate
        
        Args:
            coordinates: [batch_size, num_objects, 4] in [0,1] range
            constraints: Processed constraints
            
        Returns:
            Constraint satisfaction rate [0,1]
        """
        if not constraints or len(constraints) == 0:
            return 1.0
        
        total_constraints = 0
        satisfied_constraints = 0
        
        try:
            # Use Beta reasoner's constraint satisfaction calculation
            satisfaction_results = self.beta_reasoner.evaluate_constraints(coordinates, constraints)
            
            if isinstance(satisfaction_results, dict):
                return satisfaction_results.get('average_satisfaction', 0.0)
            elif isinstance(satisfaction_results, (float, int)):
                return float(satisfaction_results)
            else:
                logger.warning(f"Unknown constraint satisfaction format: {type(satisfaction_results)}")
                return 0.0
                
        except Exception as e:
            logger.warning(f"Constraint satisfaction calculation failed: {e}")
            return 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for debugging and logging"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'BetaSpringModel',
            'config': self.config.to_dict(),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(next(self.parameters()).device),
            'perception_enabled': self.perception_module is not None
        }


# Compatibility aliases for hardnet interface
SpringHybridConfig = BetaSpringConfig
SpringHybridModel = BetaSpringModel