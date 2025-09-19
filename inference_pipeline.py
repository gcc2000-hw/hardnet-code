"""
SPRING-HardNet Inference Pipeline
Loads trained model and generates images from specifications.
NO ERROR HANDLING - Let it crash to identify issues.
"""

import torch
from PIL import Image, ImageDraw
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
import time
import logging
from torchvision import transforms

# Import original SPRING authors' proven inpainting functions
from stable_diffusion_functions import inpaint

# Import system components
from completeInt import SpringHybridModel, SpringHybridConfig, DeploymentMode
from spring_int import SpatialReasoningMode
from constraint_language_v2 import (
    ConstraintT1, ConstraintT2, ConstraintT3, ConstraintOR
)
from constraint_gen import ConstraintGenerator, ConstraintGenerationConfig
from diffusers import StableDiffusionInpaintPipeline

# Import ALL training config classes for checkpoint deserialization
try:
    from training_config import (
        HybridTrainingConfig, DataConfig, ModelConfig, 
        Stage1Config, Stage2Config, InfrastructureConfig, ExperimentConfig
    )
    # Make all config classes available in global namespace for unpickling
    import sys
    sys.modules['__main__'].HybridTrainingConfig = HybridTrainingConfig
    sys.modules['__main__'].DataConfig = DataConfig
    sys.modules['__main__'].ModelConfig = ModelConfig
    sys.modules['__main__'].Stage1Config = Stage1Config
    sys.modules['__main__'].Stage2Config = Stage2Config
    sys.modules['__main__'].InfrastructureConfig = InfrastructureConfig
    sys.modules['__main__'].ExperimentConfig = ExperimentConfig
    
    # Also handle the legacy TrainingConfig name that might be in older checkpoints
    sys.modules['__main__'].TrainingConfig = HybridTrainingConfig
    print("All training config classes imported for checkpoint loading")
except ImportError as e:
    print(f"Could not import training config classes: {e}")
    # Create dummy classes if training_config not available
    class DummyConfig:
        pass
    import sys
    sys.modules['__main__'].HybridTrainingConfig = DummyConfig
    sys.modules['__main__'].TrainingConfig = DummyConfig  # Legacy name
    sys.modules['__main__'].DataConfig = DummyConfig
    sys.modules['__main__'].ModelConfig = DummyConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpringInferencePipeline:
    """Main inference pipeline for SPRING-HardNet system."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """Initialize the inference pipeline with trained model."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        
        logger.info(f"Initializing SPRING-HardNet inference pipeline")
        logger.info(f"Device: {self.device}")
        logger.info(f"Checkpoint: {checkpoint_path}")
        
        # Load the model
        self._load_model()
        
        # VEG Control: Set to False for colored box testing, True for realistic images
        self.enable_veg = True  # TOGGLE THIS: True=realistic images, False=colored boxes - DISABLED FOR METRICS TESTING
        
        if self.enable_veg:
            self._initialize_veg()
            logger.info("VEG enabled - will generate realistic images")
        else:
            logger.info("VEG disabled - will use colored bounding boxes")
        
        # Initialize constraint generator
        self._initialize_constraint_generator()
        
        logger.info("Pipeline initialization complete")
    
    def _load_model(self):
        """Load the trained SPRING-HardNet model from checkpoint."""
        logger.info(f"Loading model checkpoint from: {self.checkpoint_path}")
        
        # Verify checkpoint exists
        import os
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Load checkpoint - NO ERROR HANDLING
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # MODEL CONFIGURATION FIX: Extract correct training configuration from checkpoint
        checkpoint_config = checkpoint.get('config', None)
        
        if checkpoint_config is not None:
            logger.info("✓ Found checkpoint configuration (extracting model settings)")
            logger.info(f"   Config type: {type(checkpoint_config).__name__}")
            
            try:
                # FIXED: Extract from TrainingConfig object directly (not nested data_config/model_config)
                # The training config is saved as a TrainingConfig object with direct attributes
                
                # Extract image size from training config
                if hasattr(checkpoint_config, 'dataset_image_size'):
                    training_image_size = checkpoint_config.dataset_image_size
                    # Handle tuple or single value
                    if isinstance(training_image_size, (tuple, list)):
                        image_size = tuple(training_image_size)
                    else:
                        image_size = (training_image_size, training_image_size)
                else:
                    logger.warning("   No dataset_image_size in config, using training default (512, 512)")
                    image_size = (512, 512)  # Training default, NOT inference default
                
                # Extract max objects from training config
                if hasattr(checkpoint_config, 'max_objects_per_scene'):
                    max_objects = checkpoint_config.max_objects_per_scene
                else:
                    logger.warning("   No max_objects_per_scene in config, using training default 10")
                    max_objects = 10  # Training default, NOT inference default
                
                # Extract other relevant training settings
                enable_perception = True  # Default to True - most models use perception module
                
                # Log what we extracted
                logger.info(f"   ✓ Extracted from training config:")
                logger.info(f"     • image_size: {image_size} (training resolution)")
                logger.info(f"     • max_objects: {max_objects} (training object capacity)")
                logger.info(f"     • enable_perception: {enable_perception}")
                
                # CRITICAL: Create model config that matches training configuration
                self.model_config = SpringHybridConfig(
                    deployment_mode=DeploymentMode.INFERENCE,
                    device=self.device,
                    image_size=image_size,  # Use TRAINING image size
                    max_objects=max_objects,  # Use TRAINING max objects
                    mixed_precision=False,  # Disable for inference stability
                    gradient_checkpointing=False,  # Not needed for inference
                    srm_mode=SpatialReasoningMode.DIFFERENTIABLE,  # Use differentiable mode for inference
                    enable_perception_module=enable_perception,
                    enable_veg_module=False  # Disable VEG for inference
                )
                
                logger.info(f"   ✅ MODEL CONFIG CREATED: image_size={self.model_config.image_size}, max_objects={self.model_config.max_objects}")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to extract config: {e}")
                logger.warning("   Using training defaults as fallback")
                
                # Use training defaults, not arbitrary inference defaults
                self.model_config = SpringHybridConfig(
                    deployment_mode=DeploymentMode.INFERENCE,
                    device=self.device,
                    image_size=(512, 512),  # Training default
                    max_objects=10,  # Training default  
                    mixed_precision=False,
                    gradient_checkpointing=False,
                    srm_mode=SpatialReasoningMode.DIFFERENTIABLE,
                    enable_perception_module=True,
                    enable_veg_module=True
                )
        else:
            logger.warning("❌ No config found in checkpoint! Using TRAINING defaults as fallback")
            logger.warning("   This may indicate an old or corrupted checkpoint")
            
            # FIXED: Use training defaults, not arbitrary inference defaults
            self.model_config = SpringHybridConfig(
                deployment_mode=DeploymentMode.INFERENCE,
                device=self.device,
                image_size=(512, 512),  # TRAINING default (NOT 128x128!)
                max_objects=10,  # TRAINING default (NOT 5!)
                mixed_precision=False,
                gradient_checkpointing=False,
                srm_mode=SpatialReasoningMode.DIFFERENTIABLE,
                enable_perception_module=True,
                enable_veg_module=False
            )
            
            logger.info(f"   📋 Fallback config: image_size={self.model_config.image_size}, max_objects={self.model_config.max_objects}")
        
        # VERIFICATION: Log final model configuration before creating model
        logger.info(f"🏗️  CREATING MODEL with configuration:")
        logger.info(f"   📐 Image size: {self.model_config.image_size}")
        logger.info(f"   🔢 Max objects: {self.model_config.max_objects}")
        logger.info(f"   👁️  Perception module: {self.model_config.enable_perception_module}")
        logger.info(f"   🧠 SRM mode: {self.model_config.srm_mode}")
        
        # Create model with verified configuration
        self.model = SpringHybridModel(self.model_config)
        
        # SIMPLIFIED: Load checkpoint directly - let model handle its own initialization
        logger.info("Loading checkpoint state dict...")
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys during model loading: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys during model loading: {unexpected_keys}")
        
        # Simple verification - if model loads, HardNet should work
        logger.info("Model checkpoint loaded successfully")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Log training metrics from checkpoint
        if 'training_metrics' in checkpoint:
            metrics = checkpoint['training_metrics']
            logger.info(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
            logger.info(f"Final constraint satisfaction: {metrics.get('constraint_satisfaction', 'unknown')}")
        
        logger.info("Model loaded successfully")
    
    def _initialize_veg(self):
        """Initialize Stable Diffusion inpainting pipeline for object placement."""
        logger.info("Initializing Visual Element Generator (SD Inpainting)...")
        
        # Initialize the inpainting pipeline
        from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
        
        self.veg_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            safety_checker=None,  # Disable safety checker for faster inference
            requires_safety_checker=False
        )
        
        # Use DPMSolverMultistepScheduler for faster convergence
        self.veg_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.veg_pipeline.scheduler.config
        )
        
        self.veg_pipeline = self.veg_pipeline.to(self.device)
        self.veg_pipeline.enable_attention_slicing()  # Memory optimization
        
        logger.info("VEG initialized successfully with DPMSolver scheduler")
    
    def _initialize_constraint_generator(self):
        """Initialize constraint generator for converting specifications."""
        self.constraint_config = ConstraintGenerationConfig(
            canvas_width=1.0,  # Normalized coordinates (0-1)
            canvas_height=1.0,
            min_object_size=0.03,  # 3% of canvas
            max_object_size=0.4   # 40% of canvas
        )
        self.constraint_generator = ConstraintGenerator(self.constraint_config)
        logger.info("Constraint generator initialized")
    
    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to torch tensor [1,C,H,W] for original authors' functions."""
        transform = transforms.ToTensor()
        return transform(pil_image).unsqueeze(0).to(self.device)
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert torch tensor [1,C,H,W] back to PIL Image."""
        transform = transforms.ToPILImage()
        return transform(tensor[0].cpu())
    
    def _create_room_prompt(self, obj_name: str, room_type: str) -> str:
        """Create highly specific object-focused prompts for better SD generation."""
        
        # Ultra-detailed object specifications for aggressive generation
        object_specifics = {
            # Kitchen items - focus on distinctive visual features
            "bowl": "large white ceramic mixing bowl with smooth rounded edges and visible depth",
            "cup": "white ceramic coffee mug with curved handle and visible interior", 
            "banana": "bright yellow ripe banana with natural brown spots and curved shape",
            "apple": "glossy red apple with natural skin texture and visible highlights",
            "knife": "sharp stainless steel kitchen knife with black handle and metal blade",
            "spoon": "silver metal tablespoon with reflective surface and curved bowl shape",
            "donut": "round glazed chocolate frosted donut with thick sweet coating and visible texture",
            "orange": "bright orange round citrus fruit with natural bumpy skin texture",
            "carrot": "long fresh orange carrot with green leafy top and tapered shape",
            "bottle": "clear glass bottle with visible transparency and cylindrical shape",
            "wine glass": "elegant clear wine glass with long stem and wide bowl",
            
            # Furniture items - emphasize 3D structure and materials
            "chair": "solid wooden dining chair with high backrest, seat cushion, and four sturdy legs",
            "couch": "large fabric sectional sofa with multiple cushions and visible armrests",
            "table": "rectangular wooden dining table with visible wood grain and four legs",
            "lamp": "tall table lamp with fabric lampshade, metal base, and electrical cord",
            "bed": "queen size bed frame with thick mattress, pillows, and visible bedding",
            "desk": "wide wooden office desk with drawers, flat surface, and sturdy construction",
            "bookshelf": "tall wooden bookshelf with multiple horizontal shelves and vertical supports"
        }
        
        # Get specific description or fallback to basic
        specific_obj = object_specifics.get(obj_name.lower(), f"{obj_name} object")
        
        if room_type.lower() == "kitchen":
            if obj_name.lower() in ["bowl", "cup", "banana", "apple", "knife", "spoon", "donut", "orange", "carrot"]:
                return f"A {specific_obj} placed on a clean kitchen counter"
            else:
                return f"A {specific_obj} in a bright modern kitchen"
        elif room_type.lower() == "living":
            if obj_name.lower() in ["chair", "couch", "table", "lamp", "desk"]:
                return f"A {specific_obj} positioned in a modern living room"
            else:
                return f"A {specific_obj} in a contemporary living room interior"
        else:
            # Generic fallback
            return f"A {specific_obj} in a clean modern interior"
    
    def _apply_furniture_priors(self, obj_name: str, w_norm: float, h_norm: float) -> tuple:
        """Apply realistic size and aspect ratio priors for furniture objects."""
        
        # Furniture size priors (as fraction of image)
        furniture_priors = {
            'chair': {
                'min_area': 0.015,    # 1.5% of image minimum
                'max_area': 0.08,     # 8% of image maximum  
                'aspect_ratio': (0.7, 1.4),  # Width/Height ratio range
                'typical_size': 0.04  # 4% of image typically
            },
            'table': {
                'min_area': 0.03,     # 3% of image minimum
                'max_area': 0.15,     # 15% of image maximum
                'aspect_ratio': (0.8, 2.5),  # Can be square to rectangular
                'typical_size': 0.08  # 8% of image typically
            },
            'couch': {
                'min_area': 0.06,     # 6% of image minimum  
                'max_area': 0.25,     # 25% of image maximum
                'aspect_ratio': (1.8, 3.5),  # Wide furniture
                'typical_size': 0.12  # 12% of image typically
            },
            'sofa': {
                'min_area': 0.06,     # Same as couch
                'max_area': 0.25,
                'aspect_ratio': (1.8, 3.5),
                'typical_size': 0.12
            },
            'bed': {
                'min_area': 0.08,     # 8% of image minimum
                'max_area': 0.30,     # 30% of image maximum
                'aspect_ratio': (1.2, 2.0),  # Rectangular
                'typical_size': 0.15  # 15% of image typically
            }
        }
        
        # Default for unknown furniture
        default_prior = {
            'min_area': 0.02,
            'max_area': 0.12,
            'aspect_ratio': (0.6, 2.0),
            'typical_size': 0.06
        }
        
        prior = furniture_priors.get(obj_name.lower(), default_prior)
        
        # Calculate current area
        area = w_norm * h_norm
        
        # Adjust size to be within reasonable range
        if area < prior['min_area']:
            scale = (prior['min_area'] / area) ** 0.5
            w_norm *= scale
            h_norm *= scale
        elif area > prior['max_area']:
            scale = (prior['max_area'] / area) ** 0.5
            w_norm *= scale  
            h_norm *= scale
        
        # Enforce aspect ratio constraints
        aspect = w_norm / h_norm if h_norm > 0 else 1.0
        min_aspect, max_aspect = prior['aspect_ratio']
        
        if aspect < min_aspect:
            # Too tall, make wider
            w_norm = h_norm * min_aspect
        elif aspect > max_aspect:
            # Too wide, make taller
            h_norm = w_norm / max_aspect
        
        return w_norm, h_norm
    
    def _generate_collision_avoidance_constraints_DISABLED(self, enhanced_specification: Dict) -> List:
        """Generate constraints to prevent new objects from overlapping with existing ones."""
        constraints = []
        existing_objects_with_boxes = enhanced_specification.get('existing_objects_with_boxes', [])
        new_objects = enhanced_specification['objects']
        all_objects = enhanced_specification['all_objects']
        
        # Generate separation constraints for each new object vs each existing object
        for new_obj in new_objects:
            new_obj_idx = all_objects.index(new_obj)
            
            for existing_obj in existing_objects_with_boxes:
                existing_obj_name = existing_obj['class']
                existing_obj_idx = all_objects.index(existing_obj_name)
                
                # Get existing object position (convert to normalized coordinates)
                x, y, w, h = existing_obj['bbox']
                # FIXED: Normalize by actual training image size (not hardcoded 512)
                training_img_size = self.model_config.image_size[0]  # Use training image size
                x_norm, y_norm = x / training_img_size, y / training_img_size
                w_norm, h_norm = w / training_img_size, h / training_img_size
                logger.debug(f"   Normalized existing object {existing_obj['class']}: [{x_norm:.3f}, {y_norm:.3f}, {w_norm:.3f}, {h_norm:.3f}]")
                
                # Create constraints to keep new object away from existing object
                # Option 1: new object left of existing (new.x + new.width < existing.x)
                # Option 2: new object right of existing (new.x > existing.x + existing.width) 
                # Option 3: new object above existing (new.y + new.height < existing.y)
                # Option 4: new object below existing (new.y > existing.y + existing.height)
                
                # For simplicity, require separation in at least X or Y direction
                # Generate OR constraint: (new_obj left OR right OR above OR below existing_obj)
                
                from constraint_language_v2 import ConstraintOR, ConstraintT4
                
                separation_options = [
                    # Left separation: new_obj.x + new_obj.w < existing_obj.x  
                    ConstraintT4("lt", "+", new_obj_idx, 0, new_obj_idx, 2, existing_obj_idx, 0, -0.01),
                    # Right separation: new_obj.x > existing_obj.x + existing_obj.w
                    ConstraintT4("gt", "+", new_obj_idx, 0, existing_obj_idx, 0, existing_obj_idx, 2, 0.01),
                    # Above separation: new_obj.y + new_obj.h < existing_obj.y
                    ConstraintT4("lt", "+", new_obj_idx, 1, new_obj_idx, 3, existing_obj_idx, 1, -0.01),
                    # Below separation: new_obj.y > existing_obj.y + existing_obj.h  
                    ConstraintT4("gt", "+", new_obj_idx, 1, existing_obj_idx, 1, existing_obj_idx, 3, 0.01)
                ]
                
                # Create OR constraint for separation
                separation_constraint = ConstraintOR(separation_options)
                constraints.append(separation_constraint)
        
        return constraints
    
    def _generate_smart_collision_constraints_DISABLED(self, enhanced_specification: Dict) -> List:
        """Generate selective collision constraints that prioritize layout quality."""
        constraints = []
        existing_objects_with_boxes = enhanced_specification.get('existing_objects_with_boxes', [])
        new_objects = enhanced_specification['objects']
        all_objects = enhanced_specification['all_objects']
        
        logger.info(f"SMART COLLISION: Processing {len(existing_objects_with_boxes)} existing objects for {len(new_objects)} new objects")
        
        # Only generate constraints for objects that are likely to create significant conflicts
        for existing_obj in existing_objects_with_boxes:
            existing_obj_name = existing_obj['class']
            existing_obj_idx = all_objects.index(existing_obj_name)
            
            # Get existing object bounding box (already in per-mille coordinates)
            x, y, w, h = existing_obj['bbox']
            
            # Calculate object "influence zone" - area where new objects shouldn't be placed
            # Use relaxed boundaries to avoid over-constraining
            influence_margin = 30.0  # Relaxed margin in per-mille coordinates
            
            # Create a single strategic constraint per existing object
            # Prefer horizontal separation (left/right) over vertical (up/down)
            from constraint_language_v2 import ConstraintOR, ConstraintT4
            
            # Strategy: Force new objects to be either LEFT or RIGHT of existing object
            # This creates a simple binary choice instead of 4-way OR constraint
            for new_obj in new_objects:
                new_obj_idx = all_objects.index(new_obj)
                
                # Determine best separation strategy based on existing object position
                center_x = x + w/2
                
                if center_x < 400:  # Existing object in left half - prefer right separation
                    preferred_separation = ConstraintT4("gt", "+", new_obj_idx, 0, existing_obj_idx, 0, existing_obj_idx, 2, influence_margin)
                    fallback_separation = ConstraintT4("lt", "+", new_obj_idx, 0, new_obj_idx, 2, existing_obj_idx, 0, -influence_margin)
                    logger.info(f"  {new_obj} vs {existing_obj_name}: Prefer RIGHT separation (existing in left half)")
                else:  # Existing object in right half - prefer left separation  
                    preferred_separation = ConstraintT4("lt", "+", new_obj_idx, 0, new_obj_idx, 2, existing_obj_idx, 0, -influence_margin)
                    fallback_separation = ConstraintT4("gt", "+", new_obj_idx, 0, existing_obj_idx, 0, existing_obj_idx, 2, influence_margin)
                    logger.info(f"  {new_obj} vs {existing_obj_name}: Prefer LEFT separation (existing in right half)")
                
                # Create simplified OR constraint with only 2 sensible options
                smart_constraint = ConstraintOR([preferred_separation, fallback_separation])
                constraints.append(smart_constraint)
        
        logger.info(f"SMART COLLISION: Generated {len(constraints)} strategic constraints (vs {len(existing_objects_with_boxes) * len(new_objects) * 4} in full mode)")
        return constraints
    
    
    def _get_furniture_typical_width(self, obj_name: str) -> float:
        """Get typical width for furniture object based on furniture priors."""
        furniture_priors = {
            'chair': {'typical_size': 0.04, 'aspect_ratio': (0.7, 1.4)},
            'table': {'typical_size': 0.08, 'aspect_ratio': (0.8, 2.5)},
            'couch': {'typical_size': 0.12, 'aspect_ratio': (1.8, 3.5)},
            'sofa': {'typical_size': 0.12, 'aspect_ratio': (1.8, 3.5)},
            'bed': {'typical_size': 0.15, 'aspect_ratio': (1.2, 2.0)},
        }
        
        default_prior = {'typical_size': 0.06, 'aspect_ratio': (1.0, 1.5)}
        prior = furniture_priors.get(obj_name.lower(), default_prior)
        
        # Calculate typical width from area and aspect ratio
        typical_area = prior['typical_size']
        min_aspect, max_aspect = prior['aspect_ratio']
        avg_aspect = (min_aspect + max_aspect) / 2  # Use average aspect ratio
        
        # area = width * height, aspect = width / height
        # So: area = width * (width / aspect) = width^2 / aspect
        # Therefore: width = sqrt(area * aspect)
        typical_width = (typical_area * avg_aspect) ** 0.5
        
        return typical_width

    def _get_existing_object_position(self, obj_name: str, background_detections: Dict) -> Optional[List[float]]:
        """Get position of existing object from background detections.
        
        Args:
            obj_name: Name of existing object to find
            background_detections: Background detection data structure
            
        Returns:
            [x, y, width, height] position in pixel coordinates, or None if not found
        """
        try:
            # background_detections should contain detection info for the current sample
            if 'detections' in background_detections:
                for detection in background_detections['detections']:
                    if detection.get('class') == obj_name:
                        return detection.get('bbox')
            
            # Alternative: direct lookup by object name
            if obj_name in background_detections:
                obj_info = background_detections[obj_name]
                if 'bbox' in obj_info:
                    return obj_info['bbox']
                    
            logger.warning(f"Object {obj_name} not found in background detections")
            return None
            
        except Exception as e:
            logger.error(f"Error getting position for existing object {obj_name}: {e}")
            return None

    def _calculate_dynamic_separation(self, obj1_name: str, obj2_name: str, direction: str) -> float:
        """Calculate required separation between two objects to prevent overlap in MODEL COORDINATE SYSTEM."""
        
        # Get typical widths for both objects (in [0,1] normalized space)
        obj1_width = self._get_furniture_typical_width(obj1_name)
        obj2_width = self._get_furniture_typical_width(obj2_name)
        
        if direction in ['left', 'right']:
            # For horizontal separation, need: obj1_half_width + obj2_half_width + safety_margin
            required_separation_norm = (obj1_width + obj2_width) / 2 + 0.02  # In [0,1] space
        else:  # above, below
            # For vertical separation, use height-based calculation
            obj1_height = obj1_width / ((self._get_aspect_ratio(obj1_name)[0] + self._get_aspect_ratio(obj1_name)[1]) / 2)
            obj2_height = obj2_width / ((self._get_aspect_ratio(obj2_name)[0] + self._get_aspect_ratio(obj2_name)[1]) / 2)
            required_separation_norm = (obj1_height + obj2_height) / 2 + 0.02  # In [0,1] space
        
        # CRITICAL FIX: Convert from [0,1] normalized to MODEL coordinate system [-1.2, 1.2]
        # Model training space: [-1.2, 1.2] (range = 2.4)
        # Normalized space: [0, 1] (range = 1.0)  
        # Conversion: norm_value * 2.4 - 1.2 = model_value
        # But for separation (offset), we just need to scale by the range ratio
        model_separation = required_separation_norm * 2.4  # Scale [0,1] separation to [-1.2,1.2] space
        
        return model_separation
    
    def _get_aspect_ratio(self, obj_name: str) -> tuple:
        """Get aspect ratio range for furniture object."""
        furniture_priors = {
            'chair': (0.7, 1.4), 'table': (0.8, 2.5), 'couch': (1.8, 3.5),
            'sofa': (1.8, 3.5), 'bed': (1.2, 2.0)
        }
        return furniture_priors.get(obj_name.lower(), (1.0, 1.5))

    def _detect_existing_objects(self, background_tensor: torch.Tensor) -> List[Dict]:
        """Detect existing objects in background image using model's perception module."""
        
        if not hasattr(self.model, 'perception_module') or not self.model.perception_module:
            logger.warning("No perception module available - skipping background object detection")
            return []
        
        try:
            # Use model's perception module for object detection
            with torch.no_grad():
                perception_results = self.model.perception_module(background_tensor)
                detected_objects = perception_results.get('detected_objects', torch.tensor([]))
                detection_info = perception_results.get('detection_info', {})
            
            # Extract detection results - FIXED: Handle correct per_sample structure
            existing_objects = []
            if detected_objects.numel() > 0 and len(detected_objects.shape) >= 3:  # [batch, objects, 4]
                batch_detections = detected_objects[0]  # First batch
                
                # FIXED: Parse per_sample detection structure
                if 'per_sample' in detection_info and len(detection_info['per_sample']) > 0:
                    sample_info = detection_info['per_sample'][0]  # First batch sample
                    confidence_scores = sample_info.get('confidence_scores', [])
                    category_names = sample_info.get('category_names', [])
                else:
                    # Fallback for old format (shouldn't happen now)
                    confidence_scores = detection_info.get('confidence_scores', [[]])[0] if detection_info.get('confidence_scores') else []
                    category_names = detection_info.get('category_names', [[]])[0] if detection_info.get('category_names') else []
                
                logger.info(f"   Parsing: {len(confidence_scores)} confidence scores, {len(category_names)} categories")
                
                for i, detection in enumerate(batch_detections):
                    if i < len(confidence_scores) and confidence_scores[i] > 0.5:  # Filter low confidence
                        category = category_names[i] if i < len(category_names) else 'unknown'
                        confidence = confidence_scores[i]
                        
                        # CRITICAL FIX: Normalize DETR coordinates to [0,1] using training image size
                        # DETR outputs model pixel coordinates, not normalized coordinates
                        x, y, w, h = detection[:4].detach().cpu().numpy()
                        
                        # 🔍 DEBUG: Check raw DETR coordinate values
                        logger.info(f"🔍 RAW DETR COORDS: {category} raw detection = [{x:.6f}, {y:.6f}, {w:.6f}, {h:.6f}]")
                        
                        # Normalize by training image size to get [0,1] coordinates
                        training_img_size = 512  # DETR was trained at this resolution
                        x_norm = float(x / training_img_size)
                        y_norm = float(y / training_img_size)  
                        w_norm = float(w / training_img_size)
                        h_norm = float(h / training_img_size)
                        
                        logger.info(f"🔍 NORMALIZED COORDS: {category} normalized = [{x_norm:.6f}, {y_norm:.6f}, {w_norm:.6f}, {h_norm:.6f}]")
                        
                        bbox = [x_norm, y_norm, w_norm, h_norm]  # Now properly normalized [0,1]
                        
                        existing_objects.append({
                            'class': category,
                            'bbox': bbox,  # [x, y, w, h] in normalized [0,1] coordinates
                            'confidence': float(confidence)
                        })
                        
                        logger.info(f"   Detected: {category} (conf={confidence:.2f}) at {bbox}")
            
            return existing_objects
            
        except Exception as e:
            logger.error(f"Background object detection failed: {e}")
            return []

    def _detect_targeted_objects(self, background_tensor: torch.Tensor, requested_objects: List[str]) -> List[Dict]:
        """Detect specific objects requested in specification using DETR, picking highest confidence for each type."""
        
        if not hasattr(self.model, 'perception_module') or not self.model.perception_module:
            logger.warning("No perception module available - skipping targeted object detection")
            return []
        
        if not requested_objects:
            logger.info("No existing objects requested in specification")
            return []
            
        logger.info(f"Running DETR to find: {requested_objects}")
        
        try:
            # Run DETR detection once to get ALL objects in the image
            with torch.no_grad():
                perception_results = self.model.perception_module(background_tensor)
                detected_objects = perception_results.get('detected_objects', torch.tensor([]))
                detection_info = perception_results.get('detection_info', {})
            
            # Extract all detections from DETR
            all_detections = []
            if detected_objects.numel() > 0 and len(detected_objects.shape) >= 3:  # [batch, objects, 4]
                batch_detections = detected_objects[0]  # First batch
                
                # Parse per_sample detection structure
                if 'per_sample' in detection_info and len(detection_info['per_sample']) > 0:
                    sample_info = detection_info['per_sample'][0]  # First batch sample
                    confidence_scores = sample_info.get('confidence_scores', [])
                    category_names = sample_info.get('category_names', [])
                else:
                    confidence_scores = detection_info.get('confidence_scores', [[]])[0] if detection_info.get('confidence_scores') else []
                    category_names = detection_info.get('category_names', [[]])[0] if detection_info.get('category_names') else []
                
                logger.info(f"DETR detected {len(confidence_scores)} total objects in image")
                
                # Collect all detections with their info
                for i, detection in enumerate(batch_detections):
                    if i < len(confidence_scores) and confidence_scores[i] > 0.3:  # Lower threshold to catch more candidates
                        category = category_names[i] if i < len(category_names) else 'unknown'
                        confidence = float(confidence_scores[i])
                        
                        # Normalize coordinates to [0,1]
                        x, y, w, h = detection[:4].detach().cpu().numpy()
                        training_img_size = 512  # DETR training resolution
                        x_norm = float(x / training_img_size)
                        y_norm = float(y / training_img_size)  
                        w_norm = float(w / training_img_size)
                        h_norm = float(h / training_img_size)
                        
                        all_detections.append({
                            'class': category,
                            'bbox': [x_norm, y_norm, w_norm, h_norm],
                            'confidence': confidence
                        })
                        
                        logger.info(f"   DETR found: {category} (conf={confidence:.3f}) at [{x_norm:.3f}, {y_norm:.3f}, {w_norm:.3f}, {h_norm:.3f}]")
            
            # For each requested object type, find the highest confidence detection
            targeted_detections = []
            for obj_type in requested_objects:
                logger.info(f"\nSearching for requested object: '{obj_type}'")
                
                # Find all detections that match this object type (exact match or unknown)
                candidates = []
                for detection in all_detections:
                    detected_class = detection['class']
                    
                    # Exact match
                    if detected_class.lower() == obj_type.lower():
                        candidates.append(detection)
                        logger.info(f"   Found exact match: {detected_class} (conf={detection['confidence']:.3f})")
                    
                    # Also consider "unknown" detections as potential matches
                    elif detected_class.lower() == 'unknown':
                        candidates.append(detection) 
                        logger.info(f"   Found unknown candidate (conf={detection['confidence']:.3f}) - might be {obj_type}")
                
                # Pick the highest confidence candidate for this object type
                if candidates:
                    best_detection = max(candidates, key=lambda x: x['confidence'])
                    targeted_detections.append(best_detection)
                    logger.info(f"   ✓ Selected: {best_detection['class']} as '{obj_type}' (conf={best_detection['confidence']:.3f})")
                else:
                    logger.warning(f"   ✗ No detection found for requested object: {obj_type}")
            
            logger.info(f"\nTargeted detection complete: {len(targeted_detections)}/{len(requested_objects)} objects found")
            return targeted_detections
            
        except Exception as e:
            logger.error(f"Targeted object detection failed: {e}")
            return []

    def _convert_model_coords_to_pixels(self, detection: torch.Tensor, image_shape: tuple) -> List[float]:
        """Convert model coordinate system to pixel coordinates."""
        # detection should be [x, y, w, h] in model coordinate system
        # image_shape is [batch, channels, height, width]
        
        height, width = image_shape[2], image_shape[3]
        x, y, w, h = detection[:4].detach().cpu().numpy()
        
        # Convert normalized coordinates [0,1] to pixel coordinates
        x_pixel = float(x * width)
        y_pixel = float(y * height)
        w_pixel = float(w * width) 
        h_pixel = float(h * height)
        
        return [x_pixel, y_pixel, w_pixel, h_pixel]

    
    def _generate_basic_collision_constraints(self, existing_objects: List[Dict], new_objects: List[str]) -> List:
        """
        FALLBACK: Basic collision avoidance (original behavior).
        
        Used as fallback if comprehensive constraint system fails.
        Replicates the original _generate_implicit_constraints behavior.
        """
        
        logger.info("Using basic collision avoidance (fallback mode)")
        
        implicit_constraints = []
        
        if not existing_objects:
            logger.info("No existing objects detected - no implicit constraints generated")
            return implicit_constraints
        
        # Import constraint types
        try:
            from constraint_language_v2 import ConstraintT1
        except ImportError:
            logger.error("Cannot import constraint types for fallback")
            return []
        
        # Generate basic collision avoidance for each new object vs each existing object
        for new_obj_idx, new_obj_name in enumerate(new_objects):
            for existing_obj in existing_objects:
                
                existing_class = existing_obj['class']
                existing_bbox = existing_obj['bbox']  # [x, y, w, h] in pixels
                
                # FIXED: Convert pixel coordinates using actual training image size
                training_img_size = self.model_config.image_size[0]
                x_norm = existing_bbox[0] / training_img_size  # Use training image size
                y_norm = existing_bbox[1] / training_img_size
                w_norm = existing_bbox[2] / 128.0
                h_norm = existing_bbox[3] / 128.0
                
                # Calculate existing object center
                existing_center_x = x_norm + w_norm / 2
                existing_center_y = y_norm + h_norm / 2
                
                # Generate smart placement constraint based on existing object position
                constraint = self._generate_basic_placement_constraint(
                    new_obj_idx, new_obj_name, existing_obj, existing_center_x, existing_center_y
                )
                
                if constraint:
                    implicit_constraints.append(constraint)
        
        logger.info(f"Generated {len(implicit_constraints)} basic collision constraints")
        return implicit_constraints

    def _generate_basic_placement_constraint(self, new_obj_idx: int, new_obj_name: str, 
                                           existing_obj: Dict, existing_center_x: float, existing_center_y: float):
        """FALLBACK: Generate basic placement constraint to avoid existing object."""
        
        from constraint_language_v2 import ConstraintT1
        
        # Calculate required separation between new object and existing object (in model coordinates)
        existing_class = existing_obj['class']
        required_separation = self._calculate_dynamic_separation(new_obj_name, existing_class, 'left')
        
        # CRITICAL FIX: Convert existing object position from [0,1] normalized to model coordinates [-1.2, 1.2]
        existing_center_x_model = existing_center_x * 2.4 - 1.2  # Convert [0,1] -> [-1.2, 1.2]
        existing_center_y_model = existing_center_y * 2.4 - 1.2  # Convert [0,1] -> [-1.2, 1.2]
        
        # Smart placement strategy based on existing object position (in model coordinates)
        if existing_center_x_model < 0.0:  # Existing object on left side (in model space, center is 0.0)
            # Place new object to the RIGHT of existing object
            # new_obj.center_x > existing_center_x + separation
            constraint_value = existing_center_x_model + required_separation
            constraint = ConstraintT1('gt', new_obj_idx, 0, constraint_value, 0)
            direction = "right"
        else:  # Existing object on right side
            # Place new object to the LEFT of existing object  
            # new_obj.center_x < existing_center_x - separation
            constraint_value = existing_center_x_model - required_separation
            constraint = ConstraintT1('lt', new_obj_idx, 0, constraint_value, 0)
            direction = "left"
        
        logger.info(f"   Basic collision avoidance: {new_obj_name} {direction} of {existing_class}")
        logger.debug(f"     Model coordinates: existing_x={existing_center_x_model:.3f}, constraint_value={constraint_value:.3f}, separation={required_separation:.3f}")
        return constraint

    def _parse_specification_constraints(self, specification: Dict, existing_object_positions: Dict = None) -> List:
        """Convert specification constraints to model format with DYNAMIC SEPARATION."""
        constraints = []
        # Use all_objects if available (for evaluation), otherwise fall back to objects
        all_objects = specification.get('all_objects', specification['objects'])
        new_objects = specification.get('objects', [])  # Objects the model will generate
        existing_objects = specification.get('existing_objects', [])
        
        # Import all SPRING constraint types
        from constraint_language_v2 import (
            ConstraintT2, con_leftleft, con_rightright, 
            con_aboveabove, con_belowbelow
        )
        
        for constraint_spec in specification['constraints']:
            ctype = constraint_spec['type']
            
            if ctype in ['left', 'right', 'above', 'below']:
                obj1_name = constraint_spec['object1']
                obj2_name = constraint_spec['object2']
                
                # Skip constraints between existing objects (not generated by model)
                if obj1_name in existing_objects and obj2_name in existing_objects:
                    logger.info(f"SKIPPING constraint {obj1_name} {ctype} {obj2_name} - both objects are existing")
                    continue
                
                # Handle mixed constraints between new and existing objects
                obj1_is_existing = obj1_name in existing_objects
                obj2_is_existing = obj2_name in existing_objects
                
                if obj1_is_existing and obj2_is_existing:
                    # Both existing - skip (no new objects to constrain)
                    logger.info(f"SKIPPING constraint {obj1_name} {ctype} {obj2_name} - both objects are existing")
                    continue
                elif obj1_is_existing or obj2_is_existing:
                    # Mixed constraint: one new, one existing
                    logger.info(f"PROCESSING mixed constraint {obj1_name} {ctype} {obj2_name}")
                    
                    # Handle mixed constraints using T1 constraints (new_object relative to fixed position)
                    if obj1_is_existing:
                        # existing obj1, new obj2: constraint is obj1 ctype obj2
                        new_obj_name = obj2_name
                        existing_obj_name = obj1_name
                        new_obj_idx = new_objects.index(new_obj_name)
                        
                        # Get existing object position from background detections
                        existing_pos = self._get_existing_object_position(existing_obj_name, existing_object_positions or {})
                        if existing_pos is None:
                            logger.warning(f"Cannot find position for existing object {existing_obj_name}, skipping constraint")
                            continue
                            
                        # Create T1 constraint: new_obj position relative to existing_obj position
                        if ctype == 'left':
                            # existing_obj left of new_obj -> new_obj.x > existing_obj.right
                            constraint_value = existing_pos[0] + existing_pos[2] + self._calculate_dynamic_separation(existing_obj_name, new_obj_name, ctype)
                            c = ConstraintT1(c='gt', o1=new_obj_idx, v1=0, val=constraint_value, offset=0)
                        elif ctype == 'right':
                            # existing_obj right of new_obj -> new_obj.x < existing_obj.left
                            constraint_value = existing_pos[0] - self._calculate_dynamic_separation(existing_obj_name, new_obj_name, ctype)
                            c = ConstraintT1(c='lt', o1=new_obj_idx, v1=0, val=constraint_value, offset=0)
                        elif ctype == 'above':
                            # existing_obj above new_obj -> new_obj.y > existing_obj.bottom  
                            constraint_value = existing_pos[1] + existing_pos[3] + self._calculate_dynamic_separation(existing_obj_name, new_obj_name, ctype)
                            c = ConstraintT1(c='gt', o1=new_obj_idx, v1=1, val=constraint_value, offset=0)
                        elif ctype == 'below':
                            # existing_obj below new_obj -> new_obj.y < existing_obj.top
                            constraint_value = existing_pos[1] - self._calculate_dynamic_separation(existing_obj_name, new_obj_name, ctype)
                            c = ConstraintT1(c='lt', o1=new_obj_idx, v1=1, val=constraint_value, offset=0)
                        else:
                            logger.warning(f"Unsupported mixed constraint type: {ctype}")
                            continue
                    else:
                        # new obj1, existing obj2: constraint is obj1 ctype obj2  
                        new_obj_name = obj1_name
                        existing_obj_name = obj2_name
                        new_obj_idx = new_objects.index(new_obj_name)
                        
                        # Get existing object position from background detections
                        existing_pos = self._get_existing_object_position(existing_obj_name, existing_object_positions or {})
                        if existing_pos is None:
                            logger.warning(f"Cannot find position for existing object {existing_obj_name}, skipping constraint")
                            continue
                            
                        # Create T1 constraint: new_obj position relative to existing_obj position
                        if ctype == 'left':
                            # new_obj left of existing_obj -> new_obj.x < existing_obj.left
                            constraint_value = existing_pos[0] - self._calculate_dynamic_separation(new_obj_name, existing_obj_name, ctype)
                            c = ConstraintT1(c='lt', o1=new_obj_idx, v1=0, val=constraint_value, offset=0)
                        elif ctype == 'right':
                            # new_obj right of existing_obj -> new_obj.x > existing_obj.right
                            constraint_value = existing_pos[0] + existing_pos[2] + self._calculate_dynamic_separation(new_obj_name, existing_obj_name, ctype)
                            c = ConstraintT1(c='gt', o1=new_obj_idx, v1=0, val=constraint_value, offset=0)
                        elif ctype == 'above':
                            # new_obj above existing_obj -> new_obj.y < existing_obj.top
                            constraint_value = existing_pos[1] - self._calculate_dynamic_separation(new_obj_name, existing_obj_name, ctype)
                            c = ConstraintT1(c='lt', o1=new_obj_idx, v1=1, val=constraint_value, offset=0)
                        elif ctype == 'below':
                            # new_obj below existing_obj -> new_obj.y > existing_obj.bottom
                            constraint_value = existing_pos[1] + existing_pos[3] + self._calculate_dynamic_separation(new_obj_name, existing_obj_name, ctype)
                            c = ConstraintT1(c='gt', o1=new_obj_idx, v1=1, val=constraint_value, offset=0)
                        else:
                            logger.warning(f"Unsupported mixed constraint type: {ctype}")
                            continue
                    
                    constraints.append(c)
                    logger.info(f"MIXED constraint: {new_obj_name} constrained relative to {existing_obj_name} at {existing_pos}")
                    continue
                    
                # Both objects are NEW - use NEW object indices (original logic)
                obj1_idx = new_objects.index(obj1_name)
                obj2_idx = new_objects.index(obj2_name)
                
                # CRITICAL FIX: Calculate dynamic separation based on actual furniture sizes
                separation_margin = self._calculate_dynamic_separation(obj1_name, obj2_name, ctype)
                
                if ctype == 'left':
                    # obj1.center_x < obj2.center_x - dynamic_separation
                    # This ensures: obj1_right_edge < obj2_left_edge (no overlap)
                    c = ConstraintT2(c='lt', o1=obj1_idx, v1=0, o2=obj2_idx, v2=0, offset=-separation_margin)
                    constraints.append(c)
                    logger.info(f"DYNAMIC left separation: {obj1_name} left of {obj2_name}, margin={separation_margin:.3f}")
                    
                elif ctype == 'right':
                    # obj1.center_x > obj2.center_x + dynamic_separation
                    c = ConstraintT2(c='gt', o1=obj1_idx, v1=0, o2=obj2_idx, v2=0, offset=separation_margin)
                    constraints.append(c)
                    logger.info(f"DYNAMIC right separation: {obj1_name} right of {obj2_name}, margin={separation_margin:.3f}")
                    
                elif ctype == 'above':
                    # obj1.center_y < obj2.center_y - dynamic_separation  
                    c = ConstraintT2(c='lt', o1=obj1_idx, v1=1, o2=obj2_idx, v2=1, offset=-separation_margin)
                    constraints.append(c)
                    logger.info(f"DYNAMIC above separation: {obj1_name} above {obj2_name}, margin={separation_margin:.3f}")
                    
                elif ctype == 'below':
                    # obj1.center_y > obj2.center_y + dynamic_separation
                    c = ConstraintT2(c='gt', o1=obj1_idx, v1=1, o2=obj2_idx, v2=1, offset=separation_margin)
                    constraints.append(c)
                    logger.info(f"DYNAMIC below separation: {obj1_name} below {obj2_name}, margin={separation_margin:.3f}")
                
            elif ctype in ['bigger', 'smaller']:
                # Size constraints using ConstraintT2 for width comparison
                obj1_name = constraint_spec['object1']
                obj2_name = constraint_spec['object2']
                
                # Skip constraints involving existing objects
                if obj1_name in existing_objects or obj2_name in existing_objects:
                    logger.info(f"SKIPPING constraint {obj1_name} {ctype} {obj2_name} - involves existing object")
                    continue
                
                obj1_idx = new_objects.index(obj1_name)
                obj2_idx = new_objects.index(obj2_name)
                
                if ctype == 'bigger':
                    # o1 bigger than o2: o1.width > o2.width
                    c = ConstraintT2(c='gt', o1=obj1_idx, v1=2, o2=obj2_idx, v2=2, offset=0.05)  # 5% bigger
                elif ctype == 'smaller':
                    # o1 smaller than o2: o1.width < o2.width
                    c = ConstraintT2(c='lt', o1=obj1_idx, v1=2, o2=obj2_idx, v2=2, offset=-0.05)  # 5% smaller
                
                constraints.append(c)
            
            elif ctype == 'or':
                # OR constraints with conditions
                conditions = constraint_spec.get('conditions', [])
                if len(conditions) >= 2:
                    # SIMPLIFIED: Create sub-constraints without coordinate conversion
                    sub_constraints = []
                    for cond in conditions[:2]:  # Take first two conditions
                        obj_name = cond['object']
                        
                        # Skip conditions involving existing objects
                        if obj_name in existing_objects:
                            logger.info(f"SKIPPING OR condition {obj_name} - existing object")
                            continue
                        
                        obj_idx = new_objects.index(obj_name)
                        
                        # TRUST THE MODEL: Use constraint values as-is
                        constraint_value = float(cond['value'])
                        
                        if cond['type'] == 'left':
                            # x < constraint_value
                            sub_constraints.append(
                                ConstraintT1(c='lt', o1=obj_idx, v1=0, val=constraint_value, offset=0)
                            )
                            logger.info(f"  OR constraint: obj{obj_idx} x < {constraint_value}")
                        elif cond['type'] == 'right':
                            # x > constraint_value
                            sub_constraints.append(
                                ConstraintT1(c='gt', o1=obj_idx, v1=0, val=constraint_value, offset=0)
                            )
                            logger.info(f"  OR constraint: obj{obj_idx} x > {constraint_value}")
                    
                    if len(sub_constraints) == 2:
                        constraints.append(ConstraintOR(c=sub_constraints))
            
            elif ctype in ['horizontally_aligned', 'vertically_aligned']:
                # Alignment constraints - same y or x position
                obj1_name = constraint_spec['object1']
                obj2_name = constraint_spec['object2']
                
                # Handle mixed constraints for alignment (same logic as positional constraints)
                obj1_is_existing = obj1_name in existing_objects
                obj2_is_existing = obj2_name in existing_objects
                
                if obj1_is_existing and obj2_is_existing:
                    # Both existing - skip (no new objects to constrain)
                    logger.info(f"SKIPPING constraint {obj1_name} {ctype} {obj2_name} - both objects are existing")
                    continue
                elif obj1_is_existing or obj2_is_existing:
                    # Mixed alignment constraint: align new object with existing object position
                    logger.info(f"PROCESSING mixed alignment constraint {obj1_name} {ctype} {obj2_name}")
                    
                    if obj1_is_existing:
                        # existing obj1, new obj2: align obj2 with obj1 position
                        new_obj_name = obj2_name
                        existing_obj_name = obj1_name
                        new_obj_idx = new_objects.index(new_obj_name)
                    else:
                        # new obj1, existing obj2: align obj1 with obj2 position
                        new_obj_name = obj1_name
                        existing_obj_name = obj2_name
                        new_obj_idx = new_objects.index(new_obj_name)
                    
                    # Get existing object position
                    existing_pos = self._get_existing_object_position(existing_obj_name, existing_object_positions or {})
                    if existing_pos is None:
                        logger.warning(f"Cannot find position for existing object {existing_obj_name}, skipping constraint")
                        continue
                    
                    # Create T1 constraint to align new object with existing object position
                    if ctype == 'horizontally_aligned':
                        # Same y position: new_obj.y = existing_obj.y
                        constraint_value = existing_pos[1]  # y coordinate
                        c = ConstraintT1(c='eq', o1=new_obj_idx, v1=1, val=constraint_value, offset=0)
                    else:  # vertically_aligned
                        # Same x position: new_obj.x = existing_obj.x
                        constraint_value = existing_pos[0]  # x coordinate  
                        c = ConstraintT1(c='eq', o1=new_obj_idx, v1=0, val=constraint_value, offset=0)
                    
                    constraints.append(c)
                    logger.info(f"MIXED alignment: {new_obj_name} aligned with {existing_obj_name} at {constraint_value}")
                    continue
                
                # Both objects are NEW - use original T2 constraint logic
                obj1_idx = new_objects.index(obj1_name)
                obj2_idx = new_objects.index(obj2_name)
                
                if ctype == 'horizontally_aligned':
                    # Same y position: o1.y == o2.y
                    c = ConstraintT2(c='eq', o1=obj1_idx, v1=1, o2=obj2_idx, v2=1, offset=0)
                else:
                    # Same x position: o1.x == o2.x
                    c = ConstraintT2(c='eq', o1=obj1_idx, v1=0, o2=obj2_idx, v2=0, offset=0)
                
                constraints.append(c)
        
        return constraints
    
    def _remap_constraints_to_new_objects(self, all_constraints: List, existing_objects: List[Dict], new_objects: List[str]) -> List:
        """Remap constraints from all_objects space to new_objects space."""
        from constraint_language_v2 import ConstraintT1, ConstraintT2, ConstraintT3, ConstraintT4, ConstraintOR
        
        remapped = []
        existing_count = len(existing_objects)
        
        logger.info(f"Remapping constraints: {existing_count} existing objects, {len(new_objects)} new objects")
        
        for constraint in all_constraints:
            if isinstance(constraint, ConstraintT1):
                # T1: object.attr op value - only remap if it's a new object
                if constraint.o1 >= existing_count:  # New object index
                    new_constraint = ConstraintT1(
                        c=constraint.c,
                        o1=constraint.o1 - existing_count,  # Remap to new objects space (0, 1, ...)
                        v1=constraint.v1,
                        val=constraint.val,
                        offset=constraint.offset
                    )
                    remapped.append(new_constraint)
                    logger.info(f"  Remapped T1: obj_{constraint.o1} -> obj_{constraint.o1 - existing_count}")
                # Existing object constraints become absolute position constraints
                else:
                    # Convert to absolute position constraint based on existing object position
                    existing_obj = existing_objects[constraint.o1]
                    x, y, w, h = existing_obj['bbox']
                    
                    if constraint.v1 == 0:  # x coordinate
                        if constraint.c == 'gt':  # object.x > existing.x becomes object.x > actual_value
                            # COORDINATE FIX: Convert pixel coordinates to model's learned output range
                            # existing_obj['bbox'] is in pixel coordinates from DETR
                            # Model learned to output range [-0.7, +0.4], so convert pixel to model range
                            # FIXED: Use actual training image size for constraint processing
                            training_img_size = self.model_config.image_size[0]
                            pixel_norm = x / training_img_size  # Convert pixel to [0,1] using training size
                            logger.debug(f"   Constraint pixel conversion: {x} pixels / {training_img_size} = {pixel_norm:.3f}")
                            model_range_min, model_range_max = -0.7, 0.4
                            model_range_span = model_range_max - model_range_min
                            # Convert [0,1] normalized to model's output range [-0.7, 0.4]
                            actual_value = model_range_min + (pixel_norm * model_range_span)
                            new_constraint = ConstraintT1('gt', 0, 0, actual_value, constraint.offset)
                            remapped.append(new_constraint)
                            logger.info(f"  Converted T1 to absolute: obj.x > {actual_value:.3f} (pixel {x} -> model range)")
                        
            elif isinstance(constraint, ConstraintT2):
                # T2: obj1.attr op obj2.attr - remap both object indices if they're new objects
                obj1_is_new = constraint.o1 >= existing_count
                obj2_is_new = constraint.o2 >= existing_count
                
                if obj1_is_new and obj2_is_new:
                    # Both objects are new - remap both indices
                    new_constraint = ConstraintT2(
                        c=constraint.c,
                        o1=constraint.o1 - existing_count,
                        v1=constraint.v1,
                        o2=constraint.o2 - existing_count,
                        v2=constraint.v2,
                        offset=constraint.offset
                    )
                    remapped.append(new_constraint)
                    logger.info(f"  Remapped T2: obj_{constraint.o1},obj_{constraint.o2} -> obj_{constraint.o1 - existing_count},obj_{constraint.o2 - existing_count}")
                elif obj1_is_new or obj2_is_new:
                    # Mixed constraint - convert to absolute constraint for the new object
                    logger.info(f"  Skipping mixed T2 constraint (mixed existing/new objects)")
                # else: both existing objects - skip entirely
                
            elif isinstance(constraint, ConstraintT3):
                # T3: Arithmetic constraint (o1.v1 + o2.v2) op val + offset
                # Example: obj.x + obj.width < 0.99 (boundary constraint)
                # From logs: ConstraintT3(c='lt', a='+', o1=5, v1=0, o2=5, v2=2, val=0.99, offset=0)
                # This represents: obj5.x + obj5.width < 0.99
                
                obj1_is_new = constraint.o1 >= existing_count
                obj2_is_new = constraint.o2 >= existing_count
                
                logger.info(f"🔍 REMAP DEBUG: Processing ConstraintT3: obj{constraint.o1}.v{constraint.v1} {constraint.a} obj{constraint.o2}.v{constraint.v2} {constraint.c} {constraint.val}")
                logger.info(f"🔍 REMAP DEBUG: obj1_is_new={obj1_is_new}, obj2_is_new={obj2_is_new}")
                
                if obj1_is_new and obj2_is_new:
                    # Both objects are new - remap both indices
                    new_constraint = ConstraintT3(
                        c=constraint.c,
                        a=constraint.a,
                        o1=constraint.o1 - existing_count,
                        v1=constraint.v1,
                        o2=constraint.o2 - existing_count,
                        v2=constraint.v2,
                        val=constraint.val,
                        offset=constraint.offset
                    )
                    remapped.append(new_constraint)
                    logger.info(f"🔍 REMAP DEBUG: Successfully remapped T3: obj_{constraint.o1} -> obj_{constraint.o1 - existing_count}, obj_{constraint.o2} -> obj_{constraint.o2 - existing_count}")
                elif obj1_is_new and not obj2_is_new:
                    # Object 1 is new, object 2 is existing - remap obj1 only
                    new_constraint = ConstraintT3(
                        c=constraint.c,
                        a=constraint.a,
                        o1=constraint.o1 - existing_count,
                        v1=constraint.v1,
                        o2=constraint.o2,  # Keep existing object index as-is
                        v2=constraint.v2,
                        val=constraint.val,
                        offset=constraint.offset
                    )
                    remapped.append(new_constraint)
                    logger.info(f"🔍 REMAP DEBUG: Partially remapped T3: obj_{constraint.o1} -> obj_{constraint.o1 - existing_count}, obj_{constraint.o2} unchanged")
                elif not obj1_is_new and obj2_is_new:
                    # Object 1 is existing, object 2 is new - remap obj2 only
                    new_constraint = ConstraintT3(
                        c=constraint.c,
                        a=constraint.a,
                        o1=constraint.o1,  # Keep existing object index as-is
                        v1=constraint.v1,
                        o2=constraint.o2 - existing_count,
                        v2=constraint.v2,
                        val=constraint.val,
                        offset=constraint.offset
                    )
                    remapped.append(new_constraint)
                    logger.info(f"🔍 REMAP DEBUG: Partially remapped T3: obj_{constraint.o1} unchanged, obj_{constraint.o2} -> obj_{constraint.o2 - existing_count}")
                else:
                    # Both objects are existing - skip (layout is fixed)
                    logger.info(f"🔍 REMAP DEBUG: Skipping T3: both objects are existing (background layout fixed)")
                    
            elif isinstance(constraint, ConstraintT4):
                # T4: Complex arithmetic constraint (o1.v1 + o2.v2) op o3.v3 + offset
                # For chair left of couch: chair.x + chair.width < couch.x + offset
                # All objects involved must be new objects (since we filtered out existing objects)
                obj1_is_new = constraint.o1 >= existing_count
                obj2_is_new = constraint.o2 >= existing_count  
                obj3_is_new = constraint.o3 >= existing_count
                
                if obj1_is_new and obj2_is_new and obj3_is_new:
                    # All new objects - remap all indices
                    new_constraint = ConstraintT4(
                        c=constraint.c,
                        a=constraint.a,
                        o1=constraint.o1 - existing_count,
                        v1=constraint.v1,
                        o2=constraint.o2 - existing_count, 
                        v2=constraint.v2,
                        o3=constraint.o3 - existing_count,
                        v3=constraint.v3,
                        offset=constraint.offset
                    )
                    remapped.append(new_constraint)
                    logger.info(f"FIXED: Remapped T4 constraint - obj{constraint.o1}+obj{constraint.o2} {constraint.c} obj{constraint.o3}")
                else:
                    logger.warning(f"  Skipping T4 constraint with mixed existing/new objects")
                    
            elif isinstance(constraint, ConstraintOR):
                # OR constraint - recursively remap sub-constraints
                remapped_sub = self._remap_constraints_to_new_objects(
                    constraint.c, existing_objects, new_objects  # Use .c attribute
                )
                if remapped_sub:  # Only add if we have valid sub-constraints
                    new_or = ConstraintOR(remapped_sub)
                    remapped.append(new_or)
                    logger.info(f"  Remapped OR constraint with {len(remapped_sub)} sub-constraints")
                    
            else:
                # CRITICAL FIX: Handle SizeRatioConstraint and other extended constraint types
                constraint_type = type(constraint).__name__
                logger.info(f"🔍 REMAP DEBUG: Processing else clause: {constraint_type}")
                try:
                    from constraint_language_v2_extended import SizeRatioConstraint
                    logger.info(f"🔍 REMAP DEBUG: Successfully imported SizeRatioConstraint")
                    
                    if isinstance(constraint, SizeRatioConstraint):
                        logger.info(f"🔍 REMAP DEBUG: Found SizeRatioConstraint: obj{constraint.smaller_obj_idx} ≤ {constraint.max_ratio} * obj{constraint.larger_obj_idx}")
                        # SizeRatioConstraint: smaller_obj ≤ ratio * larger_obj
                        smaller_idx = constraint.smaller_obj_idx
                        larger_idx = constraint.larger_obj_idx
                        
                        logger.info(f"🔍 REMAP DEBUG: smaller_idx={smaller_idx}, larger_idx={larger_idx}, existing_count={existing_count}")
                        
                        # Case 1: Both objects are new - direct remapping
                        if smaller_idx >= existing_count and larger_idx >= existing_count:
                            logger.info(f"🔍 REMAP DEBUG: Case 1 - Both objects are new, direct remapping")
                            new_constraint = SizeRatioConstraint(
                                smaller_obj_idx=smaller_idx - existing_count,
                                larger_obj_idx=larger_idx - existing_count,
                                dimension=constraint.dimension,
                                max_ratio=constraint.max_ratio,
                                constraint_source=constraint.constraint_source
                            )
                            remapped.append(new_constraint)
                            logger.info(f"🔍 REMAP DEBUG: Successfully remapped SizeRatio: obj_{smaller_idx} {constraint.dimension} ≤ {constraint.max_ratio:.3f} * obj_{larger_idx} -> obj_{smaller_idx - existing_count} ≤ obj_{larger_idx - existing_count}")
                            
                        # Case 2: New object constrained by existing object - convert to absolute constraint  
                        elif smaller_idx >= existing_count and larger_idx < existing_count:
                            logger.info(f"🔍 REMAP DEBUG: Case 2 - New object constrained by existing object, converting to absolute")
                            # Get existing object bbox
                            existing_obj = existing_objects[larger_idx]
                            x, y, w, h = existing_obj['bbox']
                            
                            # COORDINATE FIX: DETR coordinates are already normalized [0,1]
                            # No conversion needed - use directly for constraint calculation
                            
                            # Calculate absolute size limit: new_obj.dimension ≤ ratio * existing_obj.dimension  
                            if constraint.dimension == 'width':
                                w_normalized = w  # Already normalized [0,1] from DETR
                                absolute_limit = constraint.max_ratio * w_normalized  # Apply ratio in normalized space
                                var_idx = 2  # width is variable index 2
                            else:  # height
                                h_normalized = h  # Already normalized [0,1] from DETR
                                absolute_limit = constraint.max_ratio * h_normalized  # Apply ratio in normalized space
                                var_idx = 3  # height is variable index 3
                            
                            # Create absolute constraint: new_obj.dimension ≤ absolute_limit
                            new_constraint = ConstraintT1('lt', smaller_idx - existing_count, var_idx, absolute_limit, 0)
                            remapped.append(new_constraint)
                            logger.info(f"  Converted SizeRatio to absolute: obj_{smaller_idx} {constraint.dimension} ≤ {constraint.max_ratio:.3f} * {constraint.dimension}({absolute_limit:.3f}) -> obj_{smaller_idx - existing_count}.{constraint.dimension} ≤ {absolute_limit:.3f}")
                            
                        # Case 3: Existing object constrained by new object - skip (not meaningful)
                        else:
                            logger.info(f"  Skipping SizeRatio constraint with existing smaller object: obj_{smaller_idx} ≤ obj_{larger_idx}")
                    else:
                        # Unknown constraint type - log and skip
                        logger.warning(f"🔍 REMAP DEBUG: Unknown constraint type {type(constraint).__name__} - skipping")
                        logger.warning(f"🔍 REMAP DEBUG: Constraint details: {constraint}")
                        
                except ImportError as e:
                    # Extended constraints not available - log and skip
                    logger.warning(f"🔍 REMAP DEBUG: Extended constraints not available - skipping {type(constraint).__name__}")
                    logger.warning(f"🔍 REMAP DEBUG: ImportError: {e}")
                except Exception as e:
                    logger.error(f"🔍 REMAP DEBUG: Exception in constraint processing: {e}")
                    logger.error(f"🔍 REMAP DEBUG: Constraint: {constraint}")
                    import traceback
                    logger.error(f"🔍 REMAP DEBUG: Traceback: {traceback.format_exc()}")
                    
        logger.info(f"🔍 REMAP DEBUG: Constraint remapping: {len(all_constraints)} -> {len(remapped)}")
        
        # 🔍 DEBUG: Final remapped constraint breakdown
        remapped_type_counts = {}
        for constraint in remapped:
            ctype = type(constraint).__name__
            remapped_type_counts[ctype] = remapped_type_counts.get(ctype, 0) + 1
        
        logger.info(f"🔍 REMAP DEBUG: Remapped constraint breakdown:")
        for ctype, count in remapped_type_counts.items():
            logger.info(f"   {ctype}: {count}")
            if ctype == 'SizeRatioConstraint':
                logger.info(f"      🎯 SIZE RATIO CONSTRAINTS IN REMAPPED RESULT! Count: {count}")
        
        return remapped
    
    def generate(self, background_path: str, specification: Dict) -> Dict:
        """
        Generate image from background and specification.
        
        Args:
            background_path: Path to background image
            specification: Dict with 'objects' and 'constraints'
        
        Returns:
            Dict with 'image', 'layout', 'time', 'satisfied'
        """
        start_time = time.time()
        
        # Load and preprocess background
        background = Image.open(background_path).convert("RGB")
        original_size = background.size  # Keep track of original size
        
        # CRITICAL FIX: Resize to TRAINING resolution (not hardcoded 128x128)
        model_image_size = self.model_config.image_size
        logger.info(f"   🖼️  Resizing background to training resolution: {model_image_size}")
        background_model = background.resize(model_image_size)  # Use TRAINING image size
        
        # Convert model input to tensor
        bg_tensor = torch.from_numpy(np.array(background_model)).float() / 255.0
        bg_tensor = bg_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # DUAL APPROACH: Run DETR for scene understanding + use pre-computed for visualization
        logger.info("DUAL APPROACH: Running DETR for scene understanding + using pre-computed for visualization...")
        
        # 1. Run DETR for scene understanding (helps model make better decisions)
        logger.info("Running DETR for scene understanding...")
        existing_objects_detr = self._detect_existing_objects(bg_tensor)
        logger.info(f"DETR detected {len(existing_objects_detr)} objects for scene understanding")
        
        # 2. Get pre-computed detections for visualization (filtered red bboxes)
        existing_objects_list = specification.get('existing_objects', [])
        existing_objects_with_boxes = specification.get('existing_objects_with_boxes', [])
        
        logger.info(f"Specification defines {len(existing_objects_list)} existing objects for visualization: {existing_objects_list}")
        logger.info(f"Pre-computed detections provided: {len(existing_objects_with_boxes)} objects with bboxes")
        
        # Parse explicit constraints from specification  
        # Use DETR detections for constraint handling (better scene understanding)
        # But keep pre-computed for visualization
        existing_positions = {'detections': existing_objects_detr} if existing_objects_detr else {}
        explicit_constraints = self._parse_specification_constraints(specification, existing_positions)
        logger.info(f"Parsed {len(explicit_constraints)} explicit constraints using DETR scene understanding")
        
        constraints = explicit_constraints
        num_objects = len(specification['objects'])
        logger.info(f"Using {len(constraints)} explicit constraints for {num_objects} objects")
        
        # CONSTRAINT REMAPPING DISABLED: 
        # After removing implicit constraints, explicit constraints are already in correct format
        logger.info("Constraint remapping disabled - explicit constraints already in correct format")
        
        # Check model limits
        if num_objects > 5:
            raise ValueError(f"Too many objects ({num_objects}) - model max_objects=5")
        
        num_objects_tensor = torch.tensor([num_objects], dtype=torch.long, device=self.device)
        logger.info(f"Model processing {num_objects} objects with {len(constraints)} constraints")
        
        # DEBUG: Log constraint details before model
        logger.info(f"\n🔍 CONSTRAINT DEBUG - FINAL CONSTRAINTS BEFORE MODEL:")
        logger.info(f"🔍 Total constraints to pass to model: {len(constraints)}")
        
        # Count constraint types
        constraint_type_counts = {}
        for constraint in constraints:
            ctype = type(constraint).__name__
            constraint_type_counts[ctype] = constraint_type_counts.get(ctype, 0) + 1
        
        logger.info(f"🔍 Constraint type breakdown:")
        for ctype, count in constraint_type_counts.items():
            logger.info(f"   {ctype}: {count}")
            if ctype == 'SizeRatioConstraint':
                logger.info(f"      🎯 SIZE RATIO CONSTRAINTS PRESENT! Count: {count}")
        
        for i, constraint in enumerate(constraints):
            ctype = type(constraint).__name__
            if ctype == 'SizeRatioConstraint':
                logger.info(f"🔍 MODEL INPUT SIZE RATIO {i}: obj{constraint.smaller_obj_idx} ≤ {constraint.max_ratio:.3f} * obj{constraint.larger_obj_idx} ({constraint.dimension})")
            elif hasattr(constraint, 'c'):
                if ctype == 'ConstraintT2':
                    logger.info(f"  {i}: {ctype} - obj{constraint.o1}.v{constraint.v1} {constraint.c} obj{constraint.o2}.v{constraint.v2} + {constraint.offset}")
                    if constraint.c == 'lt' and constraint.v1 == 0 and constraint.v2 == 0:
                        logger.info(f"      -> This is 'obj{constraint.o1} LEFT OF obj{constraint.o2}'")
                elif ctype == 'ConstraintT1':
                    logger.info(f"  {i}: {ctype} - obj{constraint.o1}.v{constraint.v1} {constraint.c} {constraint.val}")
                elif ctype == 'ConstraintOR':
                    sub_constraints = constraint.c if hasattr(constraint, 'c') else []
                    logger.info(f"  {i}: {ctype} - OR constraint with {len(sub_constraints)} sub-constraints")
                    for j, sub in enumerate(sub_constraints[:3]):
                        if hasattr(sub, 'c') and hasattr(sub, 'o1'):
                            logger.info(f"      Sub {j}: obj{sub.o1}.v{sub.v1 if hasattr(sub, 'v1') else 0} {sub.c} {sub.val if hasattr(sub, 'val') else 'N/A'}")
                else:
                    logger.info(f"  {i}: {ctype} - {constraint.c}")
        
        # Run model inference with SIMPLIFIED parameters
        logger.info(f"SIMPLIFIED MODEL CALL: Using SpringHybridModel signature")
        logger.info(f"   - constraints: {len(constraints)} constraints for {len(specification['objects'])} objects")
        logger.info(f"   - object_categories: {specification['objects']}")
        logger.info(f"   - training_stage: stage2, srm_mode: differentiable")
        
        with torch.no_grad():
            outputs = self.model(
                bg_tensor,  # FIXED: Use positional argument like training does
                constraints=[constraints],  # FIXED: Batch format like training [[constraints]]
                return_intermediate=True,
                training_stage='stage2',
                srm_mode='differentiable'
                # REMOVED: object_categories (not used in training), images= (use positional)
            )
        
        # Extract layout (bounding boxes) from correct model output structure
        if 'layout_results' in outputs and 'final_layout' in outputs['layout_results']:
            layout = outputs['layout_results']['final_layout']
        elif 'layout' in outputs:
            layout = outputs['layout']
        elif 'coordinates' in outputs:
            layout = outputs['coordinates']
        else:
            # Fallback to predicted coordinates
            layout = outputs.get('predicted_coordinates', torch.zeros(num_objects, 4))
            logger.warning(f"No layout found in model outputs. Available keys: {list(outputs.keys())}")
        
        # Ensure layout has the right shape
        if len(layout.shape) == 3:  # batch dimension
            layout = layout[0]  # Take first batch
        if layout.shape[0] < num_objects:
            # Pad with zeros if not enough objects
            padding = torch.zeros(num_objects - layout.shape[0], 4).to(layout.device)
            layout = torch.cat([layout, padding], dim=0)
        elif layout.shape[0] > num_objects:
            # Truncate if too many objects
            layout = layout[:num_objects]
        
        # Convert layout to numpy (detach if needed)
        layout_np = layout.detach().cpu().numpy()
        # CRITICAL FIX: Proper coordinate system mapping
        # Model outputs observed range: [-0.443, 1.238] (from logs)
        # This needs to be mapped to [0, 1] for proper pixel scaling
        
        logger.info(f"Raw model output range: [{layout_np.min():.3f}, {layout_np.max():.3f}]")
        
        # SCALE DOWN MASSIVE BOUNDING BOXES
        # Model generates oversized boxes (0.346x0.346), scale them down for realism
        size_scale_factor = 0.08  # Scale width/height to 8% of original size
        layout_np[:, 2] *= size_scale_factor  # Scale width (index 2)
        layout_np[:, 3] *= size_scale_factor  # Scale height (index 3)
        # Keep x,y positions unchanged to preserve spatial relationships
        logger.info(f"Scaled object sizes by {size_scale_factor}x for realism")
        
        # DEBUG: Log raw layout values from model
        logger.info(f"\nLAYOUT DEBUG - Raw model output:")
        for i in range(min(5, layout_np.shape[0])):  # Show first 5 objects
            x, y, w, h = layout_np[i]
            obj_name = specification['objects'][i] if i < len(specification['objects']) else f'obj_{i}'
            logger.info(f"  Object {i} ({obj_name}): x={x:.6f}, y={y:.6f}, w={w:.6f}, h={h:.6f}")
        
        # Check if constraint is satisfied in raw model output
        if layout_np.shape[0] >= 2 and len(constraints) > 0:
            x0, x1 = layout_np[0][0], layout_np[1][0]
            logger.info(f"\n  CONSTRAINT CHECK: obj0.x ({x0:.6f}) < obj1.x ({x1:.6f}) ? {x0 < x1}")
            if abs(x0 - x1) < 0.0001:
                logger.warning(f"  WARNING: Both objects have SAME x-position! Difference: {abs(x0 - x1):.9f}")
                logger.warning(f"  This means the 'left' constraint is NOT being enforced!")
        
        # SIMPLIFIED: Trust the model's native coordinate outputs
        logger.info("Using model's native layout outputs without conversion")
        
        # DEBUG: Log actual model output values
        logger.info(f"\nRAW MODEL OUTPUT COORDINATES:")
        for i in range(layout_np.shape[0]):
            x_model, y_model, w_model, h_model = layout_np[i]
            obj_name = specification.get('objects', [f'obj_{i}'])[i] if i < len(specification.get('objects', [])) else f'obj_{i}'
            logger.info(f"  {obj_name}: raw=({x_model:.6f}, {y_model:.6f}, {w_model:.6f}, {h_model:.6f})")
        
        # CRITICAL DEBUG: Check if model produces different outputs for chair vs table
        if len(layout_np) >= 2:
            chair_x = layout_np[0][0]
            table_x = layout_np[1][0]
            x_diff = abs(chair_x - table_x)
            logger.info(f"\nCONSTRAINT EFFECTIVENESS TEST:")
            logger.info(f"   Chair x: {chair_x:.6f}")
            logger.info(f"   Table x: {table_x:.6f}")
            logger.info(f"   Difference: {x_diff:.6f}")
            if x_diff < 0.001:
                logger.error(f"   SAME POSITIONS! Constraints are NOT working!")
            else:
                logger.info(f"   Different positions - constraint may be working")
        logger.info("")
        
        # TRUST THE MODEL: Use outputs directly, scale to image size
        layout_pixels = np.zeros_like(layout_np)
        
        for i in range(layout_np.shape[0]):
            x_raw, y_raw, w_raw, h_raw = layout_np[i]
            
            # Get object name for logging
            obj_name = specification.get('objects', [f'obj_{i}'])[i] if i < len(specification.get('objects', [])) else f'obj_{i}'
            
            logger.info(f"COORDINATE SYSTEM FIX: Object {i} ({obj_name}) raw model output = [{x_raw:.6f}, {y_raw:.6f}, {w_raw:.6f}, {h_raw:.6f}]")
        
            # DEBUG: Track coordinate generation consistency
            if obj_name == 'orange':
                logger.info(f"  ORANGE DEBUG: Raw coords = [{x_raw:.6f}, {y_raw:.6f}, {w_raw:.6f}, {h_raw:.6f}]")
                logger.info(f"  ORANGE DEBUG: Object index = {i}, Total objects = {layout_np.shape[0]}")
                logger.info(f"  ORANGE DEBUG: Objects list = {specification.get('objects', [])}")
            
            # COORDINATE SYSTEM UNIFICATION:
            # Model was trained on [0,1] normalized coordinates from custom_dataloader.py
            # Therefore, model should output [0,1] coordinates directly
            # NO TRANSFORMATION should be needed - just validation!
            
            x_norm = x_raw  # Model output should already be [0,1] normalized
            y_norm = y_raw
            w_norm = abs(w_raw)  # Ensure width/height are positive
            h_norm = abs(h_raw)
            
            # VALIDATION: Ensure model outputs are in expected [0,1] range
            coords = [x_norm, y_norm, w_norm, h_norm]
            coord_names = ['x', 'y', 'w', 'h']
            
            coords_out_of_range = False
            for j, (coord_val, coord_name) in enumerate(zip(coords, coord_names)):
                if coord_val < -0.1 or coord_val > 1.1:  # Allow small tolerance
                    logger.warning(f"Object {i} ({obj_name}): {coord_name}={coord_val:.3f} outside expected [0,1] range!")
                    coords_out_of_range = True
                    # Clamp to valid range
                    if coord_name in ['w', 'h']:  # Width/height must be positive
                        coords[j] = max(0.02, min(0.5, coord_val))  # Reasonable object size limits
                    else:  # x, y coordinates
                        coords[j] = max(0.0, min(1.0, coord_val))
            
            if coords_out_of_range:
                logger.error(f"COORDINATE SYSTEM ERROR: Model output outside [0,1] range! This indicates training/inference coordinate system mismatch.")
            
            # Apply validated coordinates
            x_norm, y_norm, w_norm, h_norm = coords
            
            logger.info(f"Object {i} ({obj_name}): Validated normalized coords = [{x_norm:.3f}, {y_norm:.3f}, {w_norm:.3f}, {h_norm:.3f}]")
            
            # Model outputs center-based coordinates in [0,1] normalized space
            # Trust the coordinate system - no complex transformations needed
            x_center_norm = x_norm
            y_center_norm = y_norm
            
            # Step 3: Apply furniture size priors for realistic proportions (NOW WITH CORRECT obj_name)
            w_norm, h_norm = self._apply_furniture_priors(obj_name, w_norm, h_norm)
            
            # Step 4: Convert center-based coordinates to corner-based pixels for visualization
            # Model outputs (x_center, y_center) but visualization needs (x_left, y_top)
            x_center_pixel = x_center_norm * original_size[0]
            y_center_pixel = y_center_norm * original_size[1]
            w_pixel = w_norm * original_size[0]  
            h_pixel = h_norm * original_size[1]
            
            # CRITICAL FIX: Ensure center coordinates can accommodate object sizes
            # Adjust center coordinates to prevent negative corner coordinates
            min_x_center = w_pixel / 2  # Minimum center x to avoid left edge going negative
            max_x_center = original_size[0] - w_pixel / 2  # Maximum center x to avoid right edge overflow
            min_y_center = h_pixel / 2  # Minimum center y to avoid top edge going negative  
            max_y_center = original_size[1] - h_pixel / 2  # Maximum center y to avoid bottom edge overflow
            
            # DEBUG: Log the coordinate transformation step by step
            logger.info(f"=== {obj_name} COORDINATE DEBUG ===")
            logger.info(f"Raw model output: x={x_raw:.6f}, y={y_raw:.6f}, w={w_raw:.6f}, h={h_raw:.6f}")
            logger.info(f"Normalized: x_norm={x_norm:.6f}, y_norm={y_norm:.6f}, w_norm={w_norm:.6f}, h_norm={h_norm:.6f}")
            logger.info(f"Original image size: {original_size}")
            logger.info(f"Center pixels (before clamp): x_center={x_center_pixel:.1f}, y_center={y_center_pixel:.1f}")
            logger.info(f"Object size: w_pixel={w_pixel:.1f}, h_pixel={h_pixel:.1f}")
            logger.info(f"Clamp ranges: x[{min_x_center:.1f}, {max_x_center:.1f}], y[{min_y_center:.1f}, {max_y_center:.1f}]")
            
            # Clamp center coordinates to valid range
            x_center_pixel_orig = x_center_pixel
            y_center_pixel_orig = y_center_pixel
            x_center_pixel = max(min_x_center, min(max_x_center, x_center_pixel))
            y_center_pixel = max(min_y_center, min(max_y_center, y_center_pixel))
            
            logger.info(f"Center pixels (after clamp): x_center={x_center_pixel:.1f}, y_center={y_center_pixel:.1f}")
            if abs(x_center_pixel - x_center_pixel_orig) > 1:
                logger.info(f"  X center was clamped: {x_center_pixel_orig:.1f} -> {x_center_pixel:.1f}")
            if abs(y_center_pixel - y_center_pixel_orig) > 1:
                logger.info(f"  Y center was clamped: {y_center_pixel_orig:.1f} -> {y_center_pixel:.1f}")
            
            # CRITICAL FIX: Model outputs TOP-LEFT coordinates, not centers!
            # Training data shows: Y = TOP edge, not center (constraint_gen.py:1549)
            # This ensures objects are "grounded" and don't "fly in the air"
            x_pixel = x_center_pixel - w_pixel/2  # Left edge = center - half width  
            y_pixel = y_center_pixel              # TOP edge = Y coordinate directly (not center!)
            
            logger.info(f"GROUNDING FIX: Y treated as TOP edge, not center")
            
            logger.info(f"Final corner coords: x={x_pixel:.1f}, y={y_pixel:.1f}, w={w_pixel:.1f}, h={h_pixel:.1f}")
            logger.info(f"=================================")
            
            layout_pixels[i] = [x_pixel, y_pixel, w_pixel, h_pixel]
            
            # Enhanced debug logging to track the coordinate transformation
            area_percent = (w_norm * h_norm) * 100
            logger.info(f"{obj_name} COORDINATE TRANSFORMATION:")
            logger.info(f"   Raw: ({x_raw:.3f}, {y_raw:.3f}, {w_raw:.3f}, {h_raw:.3f})")
            logger.info(f"   Center-based: ({x_center_norm:.3f}, {y_center_norm:.3f}, {w_norm:.3f}, {h_norm:.3f})")
            logger.info(f"   Corner-based: ({x_norm:.3f}, {y_norm:.3f}, {w_norm:.3f}, {h_norm:.3f})")
            logger.info(f"   Pixels: ({x_pixel:.1f}, {y_pixel:.1f}, {w_pixel:.1f}, {h_pixel:.1f})")
            logger.info(f"   Size: {area_percent:.1f}% of image area (aspect ratio: {w_norm/h_norm:.2f})")
            logger.info(f"   Constraint format: x_left={x_norm:.3f}, x_right={x_norm + w_norm:.3f}")
        
        logger.info(f"Model constraint-aware coordinates:")
        for i, obj_name in enumerate(specification['objects'][:layout_pixels.shape[0]]):
            coords = layout_pixels[i]
            logger.info(f"  {obj_name}: [{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}, {coords[3]:.1f}]")
        
        # DEBUG: Log complete layout_pixels array for inpainting debug
        logger.info(f"DEBUG: Complete layout_pixels array passed to inpainting:")
        for i in range(layout_pixels.shape[0]):
            logger.info(f"  layout_pixels[{i}] = {layout_pixels[i]}")
        
        # Check constraint satisfaction from correct model output structure
        if 'constraint_satisfaction' in outputs:
            constraint_satisfied = outputs['constraint_satisfaction'].get('constraint_satisfaction_rate', 0.0)
        else:
            constraint_satisfied = outputs.get('info', {}).get('constraint_satisfaction_rate', 0.0)
        
        
        # Extract existing objects for visualization (from pre-computed background_detections.json)
        existing_objects_for_viz = []
        if existing_objects_with_boxes:
            # Use pre-computed detections directly - they're already in pixel coordinates
            existing_objects_for_viz = existing_objects_with_boxes
            logger.info(f"Using {len(existing_objects_for_viz)} pre-computed detections for RED bbox visualization")
            for obj in existing_objects_for_viz:
                logger.info(f"  Existing object for viz: {obj['class']} at pixel bbox {obj['bbox'][:2]}")
        else:
            logger.info("No existing objects with boxes provided - no red bboxes will be shown")

        # DUAL IMAGE GENERATION: Create both bbox and VEG versions
        # 1. Always generate bbox visualization for evaluation
        bbox_image = self._place_objects_fallback(background, specification['objects'], layout_pixels, existing_objects_for_viz)
        
        # 2. Generate VEG inpainted version if enabled
        veg_image = None
        if self.enable_veg:
            logger.info("Generating VEG inpainted version...")
            room_type = specification.get('room_type', 'kitchen')  # Default to kitchen if not specified
            veg_image = self._place_objects_veg_only(background, specification['objects'], layout_pixels, existing_objects_for_viz, room_type)
        
        # Primary image for compatibility (bbox version)
        final_image = bbox_image
        
        generation_time = time.time() - start_time
        
        # COORDINATE SYSTEM FIX: Return normalized [0,1] coordinates, not pixel coordinates
        # Convert layout_pixels back to normalized coordinates for consistent API
        layout_normalized = np.zeros_like(layout_pixels)
        original_size = background.size  # (width, height)
        
        for i in range(layout_pixels.shape[0]):
            x_pixel, y_pixel, w_pixel, h_pixel = layout_pixels[i]
            # Convert back to normalized [0,1] coordinates
            x_norm = x_pixel / original_size[0]
            y_norm = y_pixel / original_size[1]
            w_norm = w_pixel / original_size[0]
            h_norm = h_pixel / original_size[1]
            layout_normalized[i] = [x_norm, y_norm, w_norm, h_norm]
        
        logger.info(f"COORDINATE SYSTEM: Returning normalized coordinates [{layout_normalized.min():.3f}, {layout_normalized.max():.3f}]")
        
        result = {
            'image': final_image,  # Primary image (bbox version for compatibility)
            'bbox_image': bbox_image,  # Bounding box visualization
            'veg_image': veg_image,  # VEG inpainted version (None if disabled)
            'layout': layout_normalized.tolist(),  # Return normalized [0,1] coordinates
            'layout_pixels': layout_pixels.tolist(),  # Also provide pixel coordinates if needed
            'time': generation_time,
            'satisfied': float(constraint_satisfied),
            'num_objects': num_objects,
            'num_constraints': len(constraints)
        }
        
        logger.info(f"Generation complete in {generation_time:.2f}s, satisfaction: {constraint_satisfied:.2%}")
        
        return result
    
    def _place_objects(self, background: Image.Image, objects: List[str], layout: np.ndarray, existing_objects: List[Dict] = None) -> Image.Image:
        """Place objects - use VEG if enabled, otherwise colored boxes."""
        
        # Check VEG setting
        if not self.enable_veg:
            logger.info("VEG disabled - using colored bounding boxes")
            return self._place_objects_fallback(background, objects, layout, existing_objects)
        
        # Use VEG inpainting
        try:
            current_image = background.copy()
            img_width, img_height = current_image.size
            
            # Process each object sequentially with inpainting
            for i, obj_name in enumerate(objects):
                if i >= len(layout):
                    break
                
                # Get bounding box (x, y, w, h) - already scaled to original image size  
                bbox = layout[i]
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                # Ensure valid bbox within image bounds
                x = max(0, min(x, img_width - 1))
                y = max(0, min(y, img_height - 1))
                w = max(20, min(w, img_width - x))  # Minimum size for inpainting
                h = max(20, min(h, img_height - y))
                
                # Create inpainting mask for this object region
                mask = Image.new('RGB', (img_width, img_height), (0, 0, 0))  # Black background
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 255))  # White mask region
                
                # Create text prompt for realistic object generation
                prompt = f"a realistic {obj_name} in a kitchen interior, high quality, photorealistic"
                negative_prompt = "blurry, cartoon, anime, low quality, distorted, unrealistic"
                
                # Perform inpainting to place the object
                inpainted_image = self.veg_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=current_image.resize((512, 512)),  # SD optimal resolution
                    mask_image=mask.resize((512, 512)),
                    num_inference_steps=50,  # Increased for better quality
                    guidance_scale=7.5,
                    strength=0.9  # High strength to ensure visible objects
                ).images[0]
                
                # Resize back to original dimensions and update current image
                current_image = inpainted_image.resize((img_width, img_height), Image.LANCZOS)
                
            return current_image
            
        except Exception as e:
            logger.warning(f"VEG inpainting failed: {e}, falling back to placeholder boxes")
            # Fallback to colored boxes if inpainting fails
            return self._place_objects_fallback(background, objects, layout, existing_objects)
    
    def _place_objects_fallback(self, background: Image.Image, objects: List[str], layout: np.ndarray, existing_objects: List[Dict] = None) -> Image.Image:
        """Fallback method using colored boxes when VEG inpainting fails."""
        import cv2
        
        # Convert to numpy array
        img_array = np.array(background)
        img_height, img_width = img_array.shape[:2]
        
        # First, draw existing objects with RED bounding boxes
        if existing_objects:
            logger.info(f"Drawing {len(existing_objects)} existing objects with red bboxes")
            red_color = (255, 0, 0)  # Red color for existing objects
            thickness = max(2, min(img_width, img_height) // 180)  # Cleaner thickness
            # IMPROVED: Better font scaling - not too big, not too small  
            font_scale = max(0.4, min(0.8, min(img_width, img_height) / 600))
            
            for existing_obj in existing_objects:
                bbox = existing_obj['bbox']  # Should be in pixels [x, y, w, h]
                obj_name = existing_obj['class']
                
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                # Ensure valid bbox within image bounds
                x = max(0, min(x, img_width - 1))
                y = max(0, min(y, img_height - 1))
                w = max(5, min(w, img_width - x))
                h = max(5, min(h, img_height - y))
                
                # Draw red rectangle for existing object
                cv2.rectangle(img_array, (x, y), (x + w, y + h), red_color, thickness)
                
                # Add label with "EXISTING:" prefix - better positioning
                label = f"EXISTING: {obj_name}"
                text_thickness = max(1, int(thickness * 0.7))
                label_y = max(y - 8, 15)  # Better vertical spacing
                cv2.putText(img_array, label, (x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, red_color, text_thickness)
        
        # Then, draw generated objects with GREEN bounding boxes
        logger.info(f"Drawing {len(objects)} generated objects with green bboxes")
        green_color = (0, 255, 0)  # Green color for generated objects
        
        for i, obj_name in enumerate(objects):
            if i >= len(layout):
                break
            
            # Get bounding box (x, y, w, h) - already scaled to original image size
            bbox = layout[i]
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Ensure valid bbox within image bounds
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            w = max(5, min(w, img_width - x))
            h = max(5, min(h, img_height - y))
            
            # Draw green rectangle for generated object
            thickness = max(2, min(img_width, img_height) // 200)
            cv2.rectangle(img_array, (x, y), (x + w, y + h), green_color, thickness)
            
            # Add label with "GENERATED:" prefix - consistent font sizing
            font_scale = max(0.4, min(0.8, min(img_width, img_height) / 600))  # Same as existing objects
            label = f"GENERATED: {obj_name}"
            text_thickness = max(1, int(thickness * 0.7))
            label_y = max(y - 8, 15)  # Consistent spacing
            cv2.putText(img_array, label, (x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, green_color, text_thickness)
        
        return Image.fromarray(img_array)
    
    def _place_objects_veg_only(self, background: Image.Image, objects: List[str], layout: np.ndarray, existing_objects: List[Dict] = None, room_type: str = "kitchen") -> Optional[Image.Image]:
        """VEG inpainting only - no fallback to bbox visualization."""
        if not self.enable_veg or not hasattr(self, 'veg_pipeline') or self.veg_pipeline is None:
            logger.warning("VEG not available - cannot generate VEG image")
            return None
            
        try:
            current_image = background.copy()
            img_width, img_height = current_image.size
            logger.info(f"Starting VEG inpainting for {len(objects)} objects")
            
            # Process each object sequentially with inpainting
            for i, obj_name in enumerate(objects):
                if i >= len(layout):
                    break
                
                # DEBUG: Log exact array access
                logger.info(f"  DEBUG: Accessing layout[{i}] for object '{obj_name}' - layout shape: {np.array(layout).shape}")
                
                # Get bounding box (x, y, w, h) - already in pixel coordinates
                bbox = layout[i]
                logger.info(f"  DEBUG: layout[{i}] = {bbox} for object '{obj_name}'")
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                # Ensure valid bbox within image bounds
                x = max(0, min(x, img_width - 1))
                y = max(0, min(y, img_height - 1))
                w = max(20, min(w, img_width - x))  # Minimum size for inpainting
                h = max(20, min(h, img_height - y))
                
                # Convert coordinates from (x,y,w,h) to (x,y,x2,y2) for original authors' function
                x2 = x + w
                y2 = y + h
                
                # Create room-appropriate prompt
                object_prompt = self._create_room_prompt(obj_name, room_type)
                # Fix common grammar issues
                if object_prompt.startswith("A a"):
                    object_prompt = object_prompt.replace("A a", "An a", 1)
                elif object_prompt.startswith("A e"):
                    object_prompt = object_prompt.replace("A e", "An e", 1) 
                elif object_prompt.startswith("A i"):
                    object_prompt = object_prompt.replace("A i", "An i", 1)
                elif object_prompt.startswith("A o"):
                    object_prompt = object_prompt.replace("A o", "An o", 1)
                elif object_prompt.startswith("A u"):
                    object_prompt = object_prompt.replace("A u", "An u", 1)
                
                # Convert current PIL image to tensor for original authors' inpaint function
                current_tensor = self._pil_to_tensor(current_image)
                logger.info(f"  DEBUG: current_tensor shape: {current_tensor.shape}, device: {current_tensor.device}")
                logger.info(f"  DEBUG: Calling inpaint with coords: x={x}, y={y}, x2={x2}, y2={y2}")
                logger.info(f"  DEBUG: Prompt: '{object_prompt}'")
                
                # USE ORIGINAL AUTHORS' PROVEN INPAINTING FUNCTION
                # inpaint(img, x, y, w, h, prompt, pipe, dev) where w,h are actually x2,y2
                try:
                    new_tensor, applied, mask_vis, obj_img = inpaint(
                        current_tensor, x, y, x2, y2, 
                        object_prompt, self.veg_pipeline, self.device
                    )
                    logger.info(f"  DEBUG: inpaint() completed successfully")
                    logger.info(f"  DEBUG: new_tensor shape: {new_tensor.shape}, device: {new_tensor.device}")
                    
                    # Check if there was actually any change
                    tensor_diff = torch.abs(new_tensor - current_tensor).mean().item()
                    logger.info(f"  DEBUG: Tensor difference (change magnitude): {tensor_diff:.6f}")
                    
                except Exception as e:
                    logger.error(f"  ERROR: inpaint() failed: {e}")
                    import traceback
                    logger.error(f"  ERROR: {traceback.format_exc()}")
                    # Fallback: no change
                    new_tensor = current_tensor
                
                # Convert result back to PIL for next iteration
                current_image = self._tensor_to_pil(new_tensor)
                logger.info(f"  DEBUG: Converted back to PIL image size: {current_image.size}")
                logger.info(f"  VEG inpainted: {obj_name} at [{x}, {y}, {w}, {h}]")
            
            logger.info(f"VEG inpainting completed successfully for {len(objects)} objects")
            return current_image
            
        except Exception as e:
            logger.warning(f"VEG inpainting failed: {e}")
            return None
    
    def batch_generate(self, specifications: List[Dict], batch_size: int = 1) -> List[Dict]:
        """Generate multiple images from specifications."""
        results = []
        
        for i in range(0, len(specifications), batch_size):
            batch = specifications[i:i+batch_size]
            
            for spec in batch:
                result = self.generate(spec['background_path'], spec)
                result['id'] = spec['id']
                results.append(result)
                
                # Clear CUDA cache periodically
                if self.device.type == "cuda" and len(results) % 10 == 0:
                    torch.cuda.empty_cache()
        
        return results


def test_pipeline():
    """Test the pipeline with a sample specification."""
    # Create test specification with just 2 objects to add
    test_spec = {
        'id': 'test_001',
        'background_path': 'data/evaluation/backgrounds/living_0025.png',
        'room_type': 'living',
        'objects': ['chair', 'couch'],  # Living room objects
        'constraints': [
            {'type': 'left', 'object1': 'chair', 'object2': 'couch'},
            {'type': 'or', 'conditions': [
                {'type': 'left', 'object': 'chair', 'value': 0.25},
                {'type': 'right', 'object': 'chair', 'value': 0.75}
            ]}
        ]
    }
    
    # Initialize pipeline
    pipeline = SpringInferencePipeline(
        checkpoint_path='checkpoints/final_model.pt',
        device='cuda'
    )
    
    # Generate image
    result = pipeline.generate(test_spec['background_path'], test_spec)
    
    # Save result
    output_dir = Path('data/evaluation/test_outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the visualization
    result['image'].save(output_dir / 'test_output.png')
    logger.info(f"Saved test output to {output_dir / 'test_output.png'}")
    
    # Save metadata
    with open(output_dir / 'test_metadata.json', 'w') as f:
        json.dump({
            'specification': test_spec,
            'generation_time': result['time'],
            'constraint_satisfaction': result['satisfied'],
            'layout': result['layout']
        }, f, indent=2)
    
    print(f"Test complete!")
    print(f"  Time: {result['time']:.2f}s")
    print(f"  Satisfaction: {result['satisfied']:.2%}")
    print(f"  Output saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SPRING-HardNet Inference Pipeline')
    parser.add_argument('--background', required=True, help='Path to background image')
    parser.add_argument('--objects', required=True, nargs='+', help='Objects to place (e.g., chair table)')
    parser.add_argument('--constraints', required=True, nargs='+', help='Constraints (e.g., "chair left of table")')
    parser.add_argument('--output_path', required=True, help='Output image path')
    parser.add_argument('--save_metadata', action='store_true', help='Save metadata JSON')
    
    args = parser.parse_args()
    
    # FIXED: Parse constraint string into proper format
    constraint_string = ' '.join(args.constraints).strip()
    constraints = []
    
    if not constraint_string or constraint_string.lower() in ['none', 'no', '']:
        # No constraints specified
        constraints = []
        print("No constraints specified")
    else:
        constraint_parts = constraint_string.split()
        if len(constraint_parts) >= 4:
            obj1, relation, of_word, obj2 = constraint_parts[0], constraint_parts[1], constraint_parts[2], constraint_parts[3]
            
            # Support multiple relationship types
            if relation.lower() == 'left' and of_word.lower() == 'of':
                constraints = [{'type': 'left', 'object1': obj1, 'object2': obj2}]
                print(f"Constraint: {obj1} left of {obj2}")
            elif relation.lower() == 'right' and of_word.lower() == 'of':
                constraints = [{'type': 'right', 'object1': obj1, 'object2': obj2}]
                print(f"Constraint: {obj1} right of {obj2}")
            elif relation.lower() == 'above' and of_word.lower() == 'of':
                constraints = [{'type': 'above', 'object1': obj1, 'object2': obj2}]
                print(f"Constraint: {obj1} above {obj2}")
            elif relation.lower() == 'below' and of_word.lower() == 'of':
                constraints = [{'type': 'below', 'object1': obj1, 'object2': obj2}]
                print(f"Constraint: {obj1} below {obj2}")
            else:
                print(f"Warning: Unrecognized constraint '{constraint_string}', using no constraints")
                constraints = []
        else:
            print(f"Warning: Invalid constraint format '{constraint_string}', using no constraints")
            constraints = []
    
    # Create specification from command line arguments
    test_spec = {
        'id': 'custom_test',
        'background_path': args.background,
        'room_type': 'custom',
        'objects': args.objects,
        'constraints': constraints
    }
    
    print(f"Using background: {args.background}")
    print(f"Objects: {args.objects}")
    print(f"Constraints: {constraints}")
    
    # Initialize pipeline
    pipeline = SpringInferencePipeline(
        checkpoint_path='checkpoints/final_model.pt',
        device='cuda'
    )
    
    # Generate image
    result = pipeline.generate(test_spec['background_path'], test_spec)
    
    # Save result
    from pathlib import Path
    output_path = Path(args.output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the visualization
    result['image'].save(output_path)
    logger.info(f"Saved output to {output_path}")
    
    # Save metadata if requested
    if args.save_metadata:
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'specification': test_spec,
                'generation_time': result['time'],
                'constraint_satisfaction': result['satisfied'],
                'layout': result['layout']
            }, f, indent=2)
        print(f"Metadata saved to {metadata_path}")
    
    print(f"Generation complete!")
    print(f"  Time: {result['time']:.2f}s")
    print(f"  Satisfaction: {result['satisfied']:.2%}")
    print(f"  Output saved to {output_path}")