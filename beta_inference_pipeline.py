"""
Beta SPRING Inference Pipeline
Loads trained Beta SPRING model and generates images from specifications using
probabilistic spatial reasoning with Beta distributions.
NO ERROR HANDLING - Let it crash to identify issues.
"""

# print("SYSTEM DEBUG: beta_inference_pipeline.py loaded with Beta SPRING")

import torch
from PIL import Image, ImageDraw
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
import logging
import traceback

# Import Beta SPRING components
from beta_spring_model import BetaSpringModel, BetaSpringConfig, DeploymentMode, SpatialReasoningMode
from constraint_language_v2 import (
    ConstraintT1, ConstraintT2, ConstraintT3, ConstraintOR
)
from diffusers import StableDiffusionInpaintPipeline

# Beta SPRING uses its own configuration system - no need for training config imports

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BetaSpringInferencePipeline:
    """Main inference pipeline for Beta SPRING system using probabilistic spatial reasoning."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """Initialize the inference pipeline with trained model."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        
        logger.info(f"INITIALIZING: Beta SPRING inference pipeline on {self.device}")
        
        # Load the model
        self._load_model()
        
        # VEG Control: Set to False for colored box testing, True for realistic images
        self.enable_veg = True
          # TOGGLE THIS: True=realistic images, False=colored boxes
        
        if self.enable_veg:
            self._initialize_veg()
            logger.info("VEG ENABLED - realistic image generation")
        else:
            logger.info("VEG DISABLED - colored bounding boxes")
        
        # Initialize constraint generator
        self._initialize_constraint_generator()
        
        logger.info("PIPELINE READY")
    
    def _load_model(self):
        """Load the trained Beta SPRING model from checkpoint."""
        logger.info(f"Loading Beta SPRING model checkpoint from: {self.checkpoint_path}")
        
        # Verify checkpoint exists
        import os
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Load checkpoint - NO ERROR HANDLING
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # BETA SPRING MODEL CONFIGURATION
        # Extract configuration from checkpoint or use sensible defaults
        checkpoint_config = checkpoint.get('config', None)
        
        # Default Beta SPRING configuration for inference
        image_size = (512, 512)
        max_objects = 5
        enable_perception = True
        
        if checkpoint_config is not None:
            logger.info("✓ Found checkpoint configuration (extracting Beta SPRING settings)")
            logger.info(f"   Config type: {type(checkpoint_config).__name__}")
            
            try:
                # Extract image size if available
                if hasattr(checkpoint_config, 'image_size'):
                    image_size = checkpoint_config.image_size
                elif hasattr(checkpoint_config, 'dataset_image_size'):
                    training_image_size = checkpoint_config.dataset_image_size
                    if isinstance(training_image_size, (tuple, list)):
                        image_size = tuple(training_image_size)
                    else:
                        image_size = (training_image_size, training_image_size)
                
                # Extract max objects if available
                if hasattr(checkpoint_config, 'max_objects'):
                    max_objects = checkpoint_config.max_objects
                elif hasattr(checkpoint_config, 'max_objects_per_scene'):
                    max_objects = checkpoint_config.max_objects_per_scene
                
                # Extract perception setting if available
                if hasattr(checkpoint_config, 'enable_perception_module'):
                    enable_perception = checkpoint_config.enable_perception_module
                
                logger.info(f"   ✓ Extracted from checkpoint config:")
                logger.info(f"     • image_size: {image_size}")
                logger.info(f"     • max_objects: {max_objects}")
                logger.info(f"     • enable_perception: {enable_perception}")
                
            except Exception as e:
                logger.warning(f"   Failed to extract config details: {e}")
                logger.info("   Using Beta SPRING defaults")
        else:
            logger.info("No checkpoint config found - using Beta SPRING defaults")
        
        # Create Beta SPRING model configuration
        self.model_config = BetaSpringConfig(
            deployment_mode=DeploymentMode.INFERENCE,
            device=self.device,
            image_size=image_size,
            max_objects=max_objects,
            mixed_precision=False,  # Disable for inference stability
            gradient_checkpointing=False,  # Not needed for inference
            srm_mode=SpatialReasoningMode.DIFFERENTIABLE,  # Beta SPRING is inherently differentiable
            enable_perception_module=enable_perception,
            enable_veg_module=False  # VEG handled separately in pipeline
        )
        
        # VERIFICATION: Log final model configuration
        logger.info(f"BUILD: CREATING Beta SPRING MODEL with configuration:")
        logger.info(f"   SIZE: Image size: {self.model_config.image_size}")
        logger.info(f"   COUNT: Max objects: {self.model_config.max_objects}")
        logger.info(f"   VISION: Perception module: {self.model_config.enable_perception_module}")
        logger.info(f"   BRAIN: SRM mode: {self.model_config.srm_mode}")
        
        # Create Beta SPRING model
        self.model = BetaSpringModel(self.model_config)
        
        # Load checkpoint weights into Beta SPRING model
        logger.info("Loading checkpoint state dict into Beta SPRING model...")
        
        if 'model_state_dict' in checkpoint:
            checkpoint_state = checkpoint['model_state_dict'].copy()
            
            # Remove perception module keys to preserve HuggingFace DETR weights
            detr_keys = [k for k in checkpoint_state.keys() if 'perception_module' in k]
            for key in detr_keys:
                del checkpoint_state[key]
            
            logger.info(f"Excluded {len(detr_keys)} perception keys to preserve pre-trained DETR")
            
            # Load state dict (non-strict to handle architectural differences)
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_state, strict=False)
            
            if missing_keys:
                logger.info(f"Missing keys (expected for new Beta architecture): {len(missing_keys)}")
                logger.debug(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                logger.info(f"Unexpected keys (from different architecture): {len(unexpected_keys)}")
                logger.debug(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
                
            logger.info("Beta SPRING model weights loaded successfully")
        else:
            logger.warning("No model_state_dict found in checkpoint - using untrained model")
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Log checkpoint metadata
        if 'training_metrics' in checkpoint:
            metrics = checkpoint['training_metrics']
            logger.info(f"Checkpoint trained for {checkpoint.get('epoch', 'unknown')} epochs")
            if isinstance(metrics, dict):
                constraint_sat = metrics.get('constraint_satisfaction', 'unknown')
                logger.info(f"Final constraint satisfaction: {constraint_sat}")
        
        logger.info("Beta SPRING model loaded and ready for inference")
    
    def _initialize_veg(self):
        """Initialize Stable Diffusion inpainting pipeline for object placement."""
        logger.info("Initializing VEG (Stable Diffusion Inpainting)...")
        
        # Use SD 1.5 inpainting model (better than 2.0 for inpainting per user expert advice)
        self.veg_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",  # SD 1.5 inpainting
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
        self.veg_pipeline = self.veg_pipeline.to(self.device)
        
        # Disable safety checker for faster inference
        self.veg_pipeline.safety_checker = None
        self.veg_pipeline.requires_safety_checker = False
        
        logger.info("VEG READY")
    def _test_veg_pipeline(self):
        """Test if VEG pipeline actually works."""
        import numpy as np
        from PIL import Image, ImageDraw
        
        logger.info("Testing VEG pipeline...")
        
        # Create simple test case
        test_img = Image.new('RGB', (512, 512), 'white')
        test_mask = Image.new('L', (512, 512), 'black')
        
        # Draw a rectangle mask in the center
        draw = ImageDraw.Draw(test_mask)
        draw.rectangle([150, 150, 350, 350], fill='white')
        
        try:
            result = self.veg_pipeline(
                prompt="a red apple",
                image=test_img,
                mask_image=test_mask,
                height=512,
                width=512,
                num_inference_steps=25,
                guidance_scale=7.5
            ).images[0]
            
            # Check if anything changed
            diff = np.sum(np.array(test_img) != np.array(result))
            change_pct = (diff / (512*512*3)) * 100
            
            if change_pct < 1:
                logger.error(f"VEG TEST FAILED: Only {change_pct:.3f}% pixels changed")
                logger.error("VEG pipeline is not working! Disabling VEG.")
                self.enable_veg = False  # Disable VEG if it's not working
            else:
                logger.info(f"VEG TEST PASSED: {change_pct:.1f}% pixels changed")
                result.save("/tmp/veg_test_success.png")
                
        except Exception as e:
            logger.error(f"VEG TEST EXCEPTION: {e}")
            logger.error("Disabling VEG due to test failure")
            self.enable_veg = False
    
    def _initialize_constraint_generator(self):
        """Initialize constraint processing for Beta SPRING."""
        # Beta SPRING uses existing constraint conversion system
        # No additional constraint generator needed - constraints come from evaluation pipeline
        logger.info("Beta SPRING constraint processing initialized")
    
    def _generate_smart_prompt(self, obj_name, current_image):
        """
        AUTOMATED SMART PROMPT GENERATION SYSTEM
        
        Intelligently generates ultra-detailed Stable Diffusion prompts based on:
        - Object category and properties
        - Material characteristics  
        - Kitchen context
        - Professional photography standards
        """
        
        # STEP 1: Object Categorization System
        OBJECT_CATEGORIES = {
            'utensils': ['fork', 'spoon', 'knife', 'spatula', 'ladle', 'whisk'],
            'food_items': ['cake', 'pizza', 'sandwich', 'apple', 'orange', 'banana', 'carrot', 'donut'],
            'containers': ['bottle', 'cup', 'bowl', 'vase', 'jar', 'glass'],
            'appliances': ['microwave', 'oven', 'toaster', 'blender', 'refrigerator'],
            'fixtures': ['sink', 'clock'],
            'plants': ['potted plant']
        }
        
        # STEP 2: Material Knowledge Base
        OBJECT_MATERIALS = {
            'fork': 'silver stainless steel', 'spoon': 'polished stainless steel', 'knife': 'sharp stainless steel',
            'bottle': 'clear glass', 'cup': 'white ceramic', 'bowl': 'ceramic', 'vase': 'ceramic',
            'cake': 'layered chocolate', 'pizza': 'cheesy', 'apple': 'fresh red', 'orange': 'fresh ripe',
            'banana': 'ripe yellow', 'sandwich': 'fresh deli', 'carrot': 'fresh orange', 'donut': 'glazed',
            'microwave': 'white metal', 'oven': 'stainless steel', 'refrigerator': 'white metal',
            'sink': 'stainless steel', 'potted plant': 'ceramic pot with green', 'clock': 'wall-mounted'
        }
        
        # STEP 3: Detailed Visual Characteristics
        OBJECT_DETAILS = {
            'fork': 'four sharp tines with reflective surface, oriented horizontally on counter',
            'spoon': 'oval bowl shape with straight handle, reflective metallic surface',
            'knife': 'sharp blade with black handle on wooden cutting board, blade pointing safely away',
            'bottle': 'filled with clear water showing refraction, black cap, product label visible',
            'cup': 'cylindrical shape with comfortable handle, empty interior visible',
            'bowl': 'simple round shape, empty and ready for use',
            'vase': 'elegant curved shape with narrow neck and wider base',
            'cake': 'rich frosting with smooth swirls, layers visible, small slice cut revealing interior',
            'pizza': 'melted mozzarella with golden spots, red sauce, thin crust, appetizing steam',
            'apple': 'bright skin with natural highlights, small brown stem at top, spherical shape',
            'orange': 'bright peel with natural texture and pores, small green stem',
            'banana': 'curved shape with natural brown ripeness spots, stem end visible',
            'sandwich': 'layers of meat, cheese, lettuce, tomato, cut diagonally showing interior',
            'carrot': 'long orange vegetable with natural texture and leafy top',
            'donut': 'ring shape with glaze coating and colorful sprinkles',
            'microwave': 'rectangular shape with glass door and control panel',
            'oven': 'large appliance with metal door and control knobs',
            'refrigerator': 'tall appliance with double doors and handles',
            'sink': 'rectangular basin with faucet and drain',
            'potted plant': 'green leaves in decorative ceramic pot',
            'clock': 'round face with numbers and hands showing time'
        }
        
        # STEP 4: Photography Style Templates by Category  
        CATEGORY_TEMPLATES = {
            'utensils': "A photograph of a {material} {obj_name} {placement} on a kitchen counter, the {obj_name} has {details}, bright kitchen lighting creates realistic shadows and reflections on the metallic surface, professional product photography style, highly detailed close-up shot",
            
            'food_items': "A photograph of a {material} {obj_name} {placement} on a kitchen counter, the {obj_name} {details}, kitchen lighting creates appetizing shadows and highlights, professional food photography style, sharp focus, makes the food look fresh and appealing",
            
            'containers': "A photograph of a {material} {obj_name} sitting upright on a kitchen counter, the {obj_name} has {details}, kitchen lighting creates realistic shadows on the counter surface, professional kitchenware photography with realistic lighting and subtle reflections",
            
            'appliances': "A photograph of a {material} {obj_name} positioned on a kitchen counter, the {obj_name} {details}, kitchen lighting reflects off the surface creating realistic highlights, professional appliance photography, clean and modern appearance",
            
            'fixtures': "A photograph of a {material} {obj_name} in a kitchen setting, the {obj_name} {details}, natural kitchen lighting creates realistic shadows and highlights, professional interior photography style",
            
            'plants': "A photograph of a {material} {obj_name} sitting on a kitchen counter, the {obj_name} has {details}, soft kitchen lighting creates natural shadows, professional home decor photography"
        }
        
        # STEP 5: Smart Prompt Assembly
        def categorize_object(obj_name):
            for category, objects in OBJECT_CATEGORIES.items():
                if obj_name in objects:
                    return category
            return 'containers'  # Default fallback
        
        def get_placement_context(obj_name):
            placement_map = {
                'knife': 'lying flat',
                'fork': 'lying flat', 
                'spoon': 'lying flat',
                'pizza': 'on a white plate',
                'sandwich': 'on a white plate',
                'cake': 'on a white ceramic plate'
            }
            return placement_map.get(obj_name, '')
        
        # Generate the smart prompt
        category = categorize_object(obj_name)
        material = OBJECT_MATERIALS.get(obj_name, f'realistic {obj_name}')
        details = OBJECT_DETAILS.get(obj_name, f'natural {obj_name} characteristics')
        placement = get_placement_context(obj_name)
        template = CATEGORY_TEMPLATES.get(category, CATEGORY_TEMPLATES['containers'])
        
        smart_prompt = template.format(
            material=material,
            obj_name=obj_name, 
            placement=placement,
            details=details
        )
        
        logger.debug(f"SMART PROMPT for {obj_name}: {smart_prompt[:100]}...")
        return smart_prompt
    
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
            influence_margin = 0.03  # Relaxed margin in normalized [0,1] coordinates (3% of canvas)
            
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
        
        # COORDINATE SYSTEM FIX: Remove incorrect 2.4x scaling
        # Analysis shows model operates in coordinate range similar to [0,1], not [-1.2, 1.2]
        # Training evidence: Target range [-1.0, 1.9], Predicted range [-0.6, 0.4]
        # The 2.4x scaling created impossible constraints by making separations too large
        # Keep separations in the model's natural coordinate space
        
        return required_separation_norm  # No scaling needed - use natural separation
    
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
                        
                        # DEBUG: DEBUG: Check raw DETR coordinate values
                        logger.info(f"DEBUG: RAW DETR COORDS: {category} raw detection = [{x:.6f}, {y:.6f}, {w:.6f}, {h:.6f}]")
                        
                        # Normalize by training image size to get [0,1] coordinates
                        training_img_size = self.model_config.image_size[0]  # Use actual training image size
                        x_norm = float(x / training_img_size)
                        y_norm = float(y / training_img_size)  
                        w_norm = float(w / training_img_size)
                        h_norm = float(h / training_img_size)
                        
                        logger.info(f"DEBUG: NORMALIZED COORDS: {category} normalized = [{x_norm:.6f}, {y_norm:.6f}, {w_norm:.6f}, {h_norm:.6f}]")
                        
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
        
        # COORDINATE SYSTEM FIX: Remove incorrect coordinate conversion
        # Existing objects are detected in [0,1] normalized space
        # Model operates in similar coordinate range, no conversion needed
        existing_center_x_model = existing_center_x  # Keep in natural coordinate space
        existing_center_y_model = existing_center_y  # Keep in natural coordinate space
        
        # Smart placement strategy based on existing object position (in model coordinates)
        if existing_center_x_model < 0.5:  # Existing object on left side (center is 0.5 in [0,1] space)
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
        """Convert specification constraints using UNIFIED CONVERTER to eliminate T1/T2 mismatch."""
        # PHASE 4A: UNIFIED CONSTRAINT CONVERSION
        # Replace 311 lines of duplicate constraint logic with calls to unified converter
        # This eliminates the T1 absolute vs T2 relative constraint mismatch
        
        from evaluation_constraint_converter import EvaluationConstraintConverter
        
        # Extract objects and positions
        new_objects = specification.get('objects', [])  # Objects the model will generate
        existing_objects = specification.get('existing_objects', [])
        constraints_json = specification.get('constraints', [])
        
        print(f"UNIFIED INFERENCE: Processing {len(constraints_json)} constraints for {len(existing_objects)} existing + {len(new_objects)} new objects")
        
        # Prepare existing object positions for unified converter
        existing_positions_dict = {}
        if existing_object_positions and 'detections' in existing_object_positions:
            for detection in existing_object_positions['detections']:
                obj_name = detection.get('class', '')
                bbox = detection.get('bbox', [])
                if obj_name and len(bbox) >= 4:
                    existing_positions_dict[obj_name] = bbox  # [x, y, w, h] format
                    print(f"  Existing object: {obj_name} at {bbox}")
        
        # Create unified converter and process constraints
        converter = EvaluationConstraintConverter()
        
        # Use unified constraint conversion with mixed scenario support
        unified_constraints = converter.convert_constraints_unified(
            eval_constraints=constraints_json,
            existing_objects=existing_objects,
            existing_positions=existing_positions_dict,
            new_objects=new_objects
        )
        
        print(f"UNIFIED INFERENCE: Generated {len(unified_constraints)} internal constraints")
        print(f"  ✓ ELIMINATED T1/T2 MISMATCH: All constraints now use T2 relative semantics with enhanced offsets")
        
        return unified_constraints

    def _deduplicate_equivalent_constraints(self, constraints: List) -> List:
        """Remove mathematically equivalent T2 constraints to prevent SVD rank deficiency."""
        from constraint_language_v2 import ConstraintT2
        
        unique_constraints = []
        removed_count = 0
        
        for i, constraint in enumerate(constraints):
            is_duplicate = False
            
            # Only check T2 constraints for mathematical equivalence
            if isinstance(constraint, ConstraintT2):
                for j, existing in enumerate(unique_constraints):
                    if isinstance(existing, ConstraintT2) and self._are_t2_constraints_equivalent(constraint, existing):
                        is_duplicate = True
                        removed_count += 1
                        break
            
            if not is_duplicate:
                unique_constraints.append(constraint)
        
        return unique_constraints

    def _are_t2_constraints_equivalent(self, c1, c2) -> bool:
        """Check if two T2 constraints are mathematically equivalent."""
        # Both must be T2 constraints on same variable type
        if c1.v1 != c2.v1 or c1.v2 != c2.v2:
            return False
        
        # Check for direct equivalence: same objects, same operation, same offset
        if (c1.o1 == c2.o1 and c1.o2 == c2.o2 and c1.c == c2.c and 
            abs(c1.offset - c2.offset) < 1e-10):
            return True
        
        # Check for inverse equivalence: "A > B + offset" ≡ "B < A - offset"
        if (c1.o1 == c2.o2 and c1.o2 == c2.o1 and 
            abs(c1.offset + c2.offset) < 1e-10):
            
            # gt/lt are inverse operations
            if (c1.c == 'gt' and c2.c == 'lt') or (c1.c == 'lt' and c2.c == 'gt'):
                return True
            
            # eq/eq with flipped objects and negated offsets
            if c1.c == 'eq' and c2.c == 'eq':
                return True
        
        return False

    def _parse_single_condition(self, condition: Dict, all_objects: List[str], new_objects: List[str], 
                               existing_objects: List[str], existing_object_positions: Dict = None):
        """Parse a single constraint condition - LEGACY METHOD (now handled by unified converter)."""
        # PHASE 4B: This method is now obsolete due to unified constraint conversion
        # All constraint parsing is handled by evaluation_constraint_converter.py
        # This method is kept for compatibility but should not be called
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("LEGACY: _parse_single_condition called - should use unified converter instead")
        return None

    def _remap_constraints_to_new_objects(self, all_constraints: List, existing_objects: List[Dict], new_objects: List[str]) -> List:
        """LEGACY: Remap constraints from all_objects space to new_objects space.
        
        This method is now largely obsolete due to unified constraint conversion.
        The unified converter handles object indexing directly.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("LEGACY: _remap_constraints_to_new_objects called - constraint remapping handled by unified converter")
        
        # PHASE 4B: Most constraint remapping is now handled by the unified converter
        # This method is kept for compatibility but returns constraints as-is
        return all_constraints

    def generate(self, background_path: str, specification: Dict) -> Dict:
        """
        Generate image from background and specification.
        
        Args:
            background_path: Path to background image
            specification: Layout specification with objects and constraints
            
        Returns:
            Dict with generated image and metadata
        """
        start_time = time.time()
        
        # Load and preprocess background
        background = Image.open(background_path).convert("RGB")
        original_size = background.size  # Keep track of original size
        
        # CRITICAL FIX: Resize to TRAINING resolution (not hardcoded 128x128)
        model_image_size = self.model_config.image_size
        logger.info(f"   IMAGE:  Resizing background to training resolution: {model_image_size}")
        background_model = background.resize(model_image_size)  # Use TRAINING image size
        
        # Convert model input to tensor
        bg_tensor = torch.from_numpy(np.array(background_model)).float() / 255.0
        bg_tensor = bg_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # CRITICAL FIX: Trust specification's existing_objects count instead of perception detection
        expected_existing_count = len(specification.get('existing_objects', []))
        
        if expected_existing_count > 0:
            logger.info("DETECTING existing objects in background using perception module...")
            # Use model's perception module to detect existing objects
            all_detected_objects = self._detect_existing_objects(bg_tensor)
            
            if len(all_detected_objects) > expected_existing_count:
                # Keep only the most confident detections
                all_detected_objects.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                existing_objects = all_detected_objects[:expected_existing_count]
                logger.info(f"Limited to {len(existing_objects)} most confident detections (expected: {expected_existing_count})")
            else:
                existing_objects = all_detected_objects
        else:
            # If specification expects 0 existing objects, don't run perception detection
            existing_objects = []
            logger.info("Specification expects 0 existing objects - skipping background detection")
            
        logger.info(f"Using {len(existing_objects)} existing objects in background")
        
        # PHASE 4 UNIFIED CONSTRAINT CONVERSION
        # Pass existing object positions for mixed constraint handling
        existing_positions = {'detections': existing_objects}
        explicit_constraints = self._parse_specification_constraints(specification, existing_positions)
        logger.info(f"UNIFIED: Parsed {len(explicit_constraints)} constraints using unified converter")
        
        # CRITICAL FIX 3.1: Add collision avoidance constraints to prevent bbox overlap
        collision_constraints = self._generate_basic_collision_constraints(existing_objects, specification['objects'])
        logger.info(f"Generated {len(collision_constraints)} collision avoidance constraints")
        
        constraints = explicit_constraints + collision_constraints
        
        # Count total objects for validation - use specification count, not detection count
        # This prevents over-detection from causing validation failures
        if 'all_objects' in specification:
            num_objects = len(specification['all_objects'])
        else:
            num_objects = len(specification.get('existing_objects', [])) + len(specification['objects'])
        
        detected_count = len(existing_objects) + len(specification['objects'])
        logger.info(f"Using {len(constraints)} constraints for {num_objects} objects (spec) vs {detected_count} detected")
        
        # Check model limits (use actual model config, not hardcoded value)
        if num_objects > self.model_config.max_objects:
            raise ValueError(f"Too many objects ({num_objects}) - model max_objects={self.model_config.max_objects}")
        
        # Process with spatial reasoning module
        num_objects_tensor = torch.tensor([num_objects], dtype=torch.long, device=self.device)
        logger.info(f"Model processing {num_objects} objects with {len(constraints)} constraints")
        
        # Generate layout using model forward pass
        with torch.no_grad():
            outputs = self.model(
                bg_tensor,  # images
                constraints=[constraints],  # batch format
                return_intermediate=True
            )
            
            # Extract layout from model outputs
            if 'layout_results' in outputs and 'final_layout' in outputs['layout_results']:
                layout = outputs['layout_results']['final_layout']
            elif 'coordinates' in outputs:
                layout = outputs['coordinates']
            else:
                layout = outputs.get('predicted_coordinates', torch.zeros(num_objects, 4))
                
            # Extract constraint satisfaction
            constraint_sat = outputs.get('constraint_satisfaction', 0.0)
            if isinstance(constraint_sat, dict):
                constraint_satisfied = constraint_sat.get('average', 0.0)
            elif isinstance(constraint_sat, (list, tuple)) and len(constraint_sat) > 0:
                constraint_satisfied = float(constraint_sat[0])
            else:
                constraint_satisfied = float(constraint_sat)
            
        # Convert layout to numpy for processing
        if isinstance(layout, torch.Tensor):
            layout_np = layout.detach().cpu().numpy()
        else:
            layout_np = layout
            
        # COORDINATE DEBUGGING: Investigate bottom-right clustering issue
        logger.info(f"COORDINATE DEBUG: Raw model output analysis")
        logger.info(f"  Model output range: [{layout_np.min():.6f}, {layout_np.max():.6f}]")
        logger.info(f"  Model output shape: {layout_np.shape}")
        
        # Log each object's coordinates to identify clustering pattern
        # Handle different shape formats: (batch, objects, 4) or (objects, 4)
        if len(layout_np.shape) == 3 and layout_np.shape[0] == 1:
            # Batch dimension present, squeeze it
            coords_to_log = layout_np[0]  # Take first (and only) batch
            logger.info(f"  Found batch dimension, using shape: {coords_to_log.shape}")
        elif len(layout_np.shape) == 2:
            coords_to_log = layout_np
            logger.info(f"  Using 2D shape: {coords_to_log.shape}")
        else:
            logger.info(f"  Unexpected shape: {layout_np.shape}")
            coords_to_log = layout_np
            
        # Log individual object coordinates
        if len(coords_to_log.shape) == 2 and coords_to_log.shape[1] >= 4:
            for i in range(min(coords_to_log.shape[0], 5)):  # Log first 5 objects
                x, y, w, h = float(coords_to_log[i][0]), float(coords_to_log[i][1]), float(coords_to_log[i][2]), float(coords_to_log[i][3])
                logger.info(f"  Object {i}: x={x:.6f}, y={y:.6f}, w={w:.6f}, h={h:.6f}")
                coverage = w * h * 100
                position_desc = f"{'bottom' if y > 0.5 else 'top'}, {'right' if x > 0.5 else 'left'}"
                logger.info(f"    → Coverage: {coverage:.1f}%, Position: ({position_desc})")
                
                # Flag problematic coordinates
                if coverage > 30:
                    logger.warning(f"    ❌ MASSIVE OBJECT: {coverage:.1f}% coverage!")
                if x > 0.5 and y > 0.5:
                    logger.warning(f"    ⚠️  BOTTOM-RIGHT CLUSTERING detected")
        
        # Check if outputs contain constraint processing info
        logger.info(f"Available output keys: {list(outputs.keys()) if isinstance(outputs, dict) else 'Not dict'}")
        if isinstance(outputs, dict) and 'constraint_satisfaction' in outputs:
            logger.info(f"Constraint satisfaction in outputs: {outputs['constraint_satisfaction']}")
        
        layout_normalized = layout_np.copy()
        if layout_normalized.ndim == 3:
            layout_normalized = layout_normalized.squeeze(0)  # Remove batch dimension if present
            
        logger.info(f"Generated layout shape: {layout_normalized.shape}")
        
        # Place objects using VEG if enabled
        try:
            enhanced_image = self._place_objects(background, specification['objects'], layout_normalized[len(existing_objects):])
        except Exception as e:
            logger.warning(f"Object placement failed: {e}")
            enhanced_image = background
            
        # Calculate generation time and constraint satisfaction
        generation_time = time.time() - start_time
        
        # Create result dictionary
        result = {
            'image': enhanced_image,
            'layout': layout_normalized.tolist(),  # Convert to list for JSON serialization
            'time': generation_time,
            'satisfied': float(constraint_satisfied),
            'num_objects': len(layout_normalized),
            'num_constraints': len(constraints)
        }
        
        logger.info(f"GENERATION COMPLETE: {generation_time:.2f}s, constraint satisfaction: {constraint_satisfied:.2%}")
        
        return result
    
    def _place_objects(self, background: Image.Image, objects: List[str], layout: np.ndarray) -> Image.Image:
        """Place objects - use VEG if enabled, otherwise colored boxes."""
        
        # logger.info(f"DEBUG: _place_objects called with {len(objects)} objects, layout shape: {layout.shape if hasattr(layout, 'shape') else len(layout)}")
        # logger.info(f"DEBUG: Objects to place: {objects}")
        logger.info(f"VEG STATUS: {self.enable_veg}, objects: {objects}")
        
        # Check VEG setting
        if not self.enable_veg:
            logger.info("VEG DISABLED - using colored bounding boxes")
            return self._place_objects_fallback(background, objects, layout)
        
        # Use VEG inpainting
        try:
            logger.info(f"VEG STARTED: Processing {len(objects)} objects")
            current_image = background.copy()
            img_width, img_height = current_image.size
            
            # Process each object sequentially with inpainting
            logger.info(f"VEG LOOP: Starting inpainting for {objects}")
            for i, obj_name in enumerate(objects):
                if i >= len(layout):
                    logger.warning(f"DEBUG: Breaking at object {i} - layout only has {len(layout)} entries")
                    break
                
                logger.info(f"VEG: Processing {obj_name}")
                # Get bounding box (x, y, w, h) - COORDINATES ARE NOW NORMALIZED [0,1] from BetaSpringModel
                bbox = layout[i]
                # CRITICAL FIX: Convert normalized [0,1] coordinates to pixel coordinates
                x = int(bbox[0] * img_width)   # Convert normalized x to pixels
                y = int(bbox[1] * img_height)  # Convert normalized y to pixels
                w = int(bbox[2] * img_width)   # Convert normalized w to pixels
                h = int(bbox[3] * img_height)  # Convert normalized h to pixels
                
                # ASPECT RATIO FIX: Correct unrealistic object proportions
                aspect_ratio = w / h if h > 0 else 1.0
                if obj_name in ['bottle', 'cup', 'vase'] and aspect_ratio > 2.0:
                    # Tall objects should be taller than wide - fix if width > 2*height
                    logger.warning(f"VEG: {obj_name} has bad aspect ratio {aspect_ratio:.1f} (too wide), correcting")
                    new_h = int(w * 1.5)  # Make height 1.5x width (tall object)
                    new_y = max(0, y - (new_h - h) // 2)  # Center the taller box
                    if new_y + new_h > img_height:
                        new_y = img_height - new_h
                    h, y = new_h, new_y
                    logger.info(f"VEG: {obj_name} corrected to {w}x{h}px at ({x},{y})")
                elif obj_name in ['fork', 'knife'] and aspect_ratio < 0.5:
                    # Elongated objects should be longer than tall
                    logger.warning(f"VEG: {obj_name} has bad aspect ratio {aspect_ratio:.1f} (too tall), correcting")
                    new_w = int(h * 2.0)  # Make width 2x height
                    new_x = max(0, x - (new_w - w) // 2)  # Center the wider box
                    if new_x + new_w > img_width:
                        new_x = img_width - new_w
                    w, x = new_w, new_x
                    logger.info(f"VEG: {obj_name} corrected to {w}x{h}px at ({x},{y})")
                
                # Ensure valid bbox within image bounds
                x = max(0, min(x, img_width - 1))
                y = max(0, min(y, img_height - 1))
                w = max(20, min(w, img_width - x))  # Minimum size for inpainting
                h = max(20, min(h, img_height - y))
                
                # Create inpainting mask for this object region
                mask = Image.new('L', (img_width, img_height), 0)  # Black background (grayscale)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle([x, y, x + w, y + h], fill=255)  # White mask region
                
                # CRITICAL DEBUG: Check mask creation
                logger.info(f"VEG: {obj_name} bbox [{x},{y},{w},{h}] = {w}x{h}px")
                logger.info(f"VEG: Image size: {img_width}x{img_height}")
                logger.info(f"VEG: Mask rectangle: [{x}, {y}, {x + w}, {y + h}]")
                
                # Validate mask has white pixels
                import numpy as np
                mask_array = np.array(mask)
                white_pixels_before_resize = np.sum(mask_array > 200)
                total_pixels = mask_array.size
                logger.info(f"VEG: Mask white pixels: {white_pixels_before_resize}/{total_pixels} ({100*white_pixels_before_resize/total_pixels:.1f}%)")
                
                # Check if bounding box is valid for inpainting
                if w < 10 or h < 10:
                    logger.warning(f"VEG: {obj_name} bbox {w}x{h}px may be too small for inpainting")
                if x < 0 or y < 0 or x+w > img_width or y+h > img_height:
                    logger.warning(f"VEG: {obj_name} bbox outside image bounds, skipping")
                
                # Enhanced object-specific prompts (from beta_spring_evaluation_pipeline.py)
                object_prompts = {
                    'chair': 'A modern comfortable chair',
                    'couch': 'A comfortable living room couch', 
                    'potted plant': 'A green potted plant',
                    'dining table': 'A wooden dining table',
                    'television': 'A flat screen television',
                    'microwave': 'A kitchen microwave oven',
                    'oven': 'A kitchen oven',
                    'toaster': 'A kitchen toaster',
                    'refrigerator': 'A white kitchen refrigerator',
                    'bed': 'A comfortable modern bed',
                    'mirror': 'A wall-mounted mirror',
                    'window': 'A clear glass window',
                    'desk': 'A wooden office desk',
                    'toilet': 'A white ceramic toilet',
                    'door': 'A wooden interior door',
                    'sink': 'A kitchen sink',
                    'blender': 'A modern kitchen blender',
                    # CRITICAL FIX: Add missing kitchen items with detailed, contextual prompts
                    'fork': 'A shiny metal fork lying on a kitchen counter',
                    'bottle': 'A clear glass bottle standing upright on a kitchen counter',
                    'cake': 'A chocolate cake sitting on a plate on the kitchen counter',
                    'orange': 'A bright orange fruit placed on the kitchen counter',
                    'vase': 'A ceramic vase with flowers on the kitchen counter',
                    'banana': 'A yellow banana lying on the kitchen counter',
                    'cup': 'A ceramic coffee mug sitting on the kitchen counter',
                    'pizza': 'A slice of pizza on a plate on the kitchen counter'
                }
                
                # AUTOMATED SMART PROMPT GENERATION SYSTEM
                prompt = self._generate_smart_prompt(obj_name, current_image)
                negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy, abstract, logo, symbol, text, cartoon, unrealistic, floating objects"
                
                # Prepare inputs for Stable Diffusion inpainting
                sd_input = current_image.resize((512, 512))
                sd_mask = mask.resize((512, 512))
                
                # DEBUG: Save input and mask for inspection
                sd_input.save(f"/tmp/sd_input_{obj_name}_{i}.png")
                sd_mask.save(f"/tmp/sd_mask_{obj_name}_{i}.png")
                logger.debug(f"VEG: Saved input and mask for {obj_name} debugging")
                
                # Validate mask has inpainting regions
                import numpy as np
                mask_array = np.array(sd_mask)
                white_pixels = np.sum(mask_array > 200)
                if white_pixels < 100:  # Skip if mask too small
                    logger.warning(f"VEG: {obj_name} mask too small ({white_pixels} pixels), skipping")
                    continue
                
                # Perform inpainting to place the object
                logger.info(f"VEG: Inpainting {obj_name} with prompt: '{prompt}'")
                inpainted_image = self.veg_pipeline(
                    prompt=prompt,  # Use enhanced object-specific prompt
                    negative_prompt=negative_prompt,
                    image=sd_input,
                    mask_image=sd_mask,
                    height=512,  # Add explicit dimensions
                    width=512,
                    num_inference_steps=50,  # At least 50 steps per user advice
                    guidance_scale=7.5,      # Lower guidance - high guidance can cause artifacts  
                    strength=1.0,            # Full denoising strength per user expert advice
                ).images[0]
                
                # DEBUG: Save SD output 
                sd_debug_path = f"/tmp/sd_output_{obj_name}_{i}.png"
                inpainted_image.save(sd_debug_path)
                logger.info(f"SD COMPLETE: {obj_name} - SAVED TO {sd_debug_path}")
                
                # Resize back to original dimensions and update current image
                new_image = inpainted_image.resize((img_width, img_height), Image.LANCZOS)
                new_image.save(f"/tmp/sd_resized_{obj_name}_{i}.png")
                
                # CRITICAL: Verify image actually changed (detect silent failures)
                import numpy as np
                old_array = np.array(current_image)
                new_array = np.array(new_image)
                diff_pixels = np.sum(old_array != new_array)
                pixel_change_percent = (diff_pixels / old_array.size) * 100
                
                # DEBUG: Check changes in masked region only
                mask_array = np.array(mask)
                mask_region = mask_array > 127  # White areas of mask
                if np.any(mask_region):
                    masked_old = old_array[mask_region]
                    masked_new = new_array[mask_region]
                    masked_diff = np.sum(masked_old != masked_new)
                    masked_change_percent = (masked_diff / masked_old.size) * 100 if masked_old.size > 0 else 0
                    logger.info(f"VEG MASK ANALYSIS: {masked_change_percent:.1f}% change in masked region ({masked_old.size} pixels)")
                else:
                    logger.error(f"VEG ERROR: No white pixels found in mask!")
                
                if pixel_change_percent < 0.1:
                    logger.warning(f"VEG LOW CHANGE: {pixel_change_percent:.3f}% pixels changed for {obj_name}")
                else:
                    logger.info(f"VEG SUCCESS: {pixel_change_percent:.1f}% pixels changed for {obj_name}")
                
                current_image = new_image
                current_debug_path = f"/tmp/current_image_after_{obj_name}_{i}.png"
                current_image.save(current_debug_path)
                logger.info(f"VEG DONE: {obj_name} - CURRENT IMAGE SAVED TO {current_debug_path}")
                
            logger.info(f"VEG COMPLETE: Returning modified image")
            return current_image
            
        except Exception as e:
            logger.error(f"VEG EXCEPTION: {type(e).__name__}: {e}")
            logger.error(f"VEG FALLBACK: Using colored boxes")
            return self._place_objects_fallback(background, objects, layout)
    
    def _place_objects_fallback(self, background: Image.Image, objects: List[str], layout: np.ndarray) -> Image.Image:
        """Fallback method using colored boxes when VEG inpainting fails."""
        logger.info(f"FALLBACK: Drawing colored boxes for {len(objects)} objects: {objects}")
        import cv2
        
        # Convert to numpy array
        img_array = np.array(background)
        img_height, img_width = img_array.shape[:2]
        
        # Colors for different objects
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
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
            
            # Draw rectangle with thickness proportional to image size
            thickness = max(2, min(img_width, img_height) // 200)
            color = colors[i % len(colors)]
            cv2.rectangle(img_array, (x, y), (x + w, y + h), color, thickness)
            
            # Add label with size proportional to image
            font_scale = max(0.5, min(img_width, img_height) / 500)
            cv2.putText(img_array, obj_name, (x, max(y - 10, 20)), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, max(1, thickness // 2))
        
        return Image.fromarray(img_array)
    
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
    pipeline = BetaSpringInferencePipeline(
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
    pipeline = BetaSpringInferencePipeline(
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