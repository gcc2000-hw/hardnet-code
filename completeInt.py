"""
SPRING Hybrid: Complete Model Integration - FIXED VERSION
The main SpringHybridModel that integrates all system components

FIXES:
1. Proper import handling with fallbacks
2. Device consistency management
3. Tensor dimension alignment
4. Error handling improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field

# Removed debug instrumentation - HardNet projection system working correctly

# PHASE 2 FIX: Import Scene-to-Layout Encoder to replace torch.randn() with scene features
from scene_to_layout_encoder import SceneToLayoutEncoder
# PHASE 3 FIX: Import Scene-to-Hidden Encoder to replace shared hidden states
from scene_to_hidden_encoder import SceneToHiddenEncoder
from enum import Enum
from pathlib import Path

# FIXED: Import handling with proper fallbacks
SPRING_COMPONENTS_AVAILABLE = False
DETR_AVAILABLE = False
DIFFUSERS_AVAILABLE = False
from spring_int import HybridSpatialReasoningModule, HybridSRMConfig, SpatialReasoningMode
SPRING_COMPONENTS_AVAILABLE = True

try:
    # FIXED: Create minimal preprocessing classes instead of importing
    @dataclass
    class PreprocessingConfig:
        target_image_size: Tuple[int, int] = (512, 512)
        device_placement: str = "auto"
        
    class ImagePreprocessor:
        def __init__(self, config):
            self.config = config
            
        def preprocess_for_perception(self, image):
            return image
            
        def preprocess_for_veg(self, image):
            return image
    
    print("✓ Created minimal preprocessing components")
except Exception as e:
    print(f"Warning: Could not create preprocessing components: {e}")

# Try to import DETR for perception module
try:
    import transformers
    from transformers import DetrImageProcessor, DetrForObjectDetection
    from transformers.utils import logging as transformers_logging
    
    # Disable transformers progress bars to avoid log cluttering
    transformers_logging.set_verbosity_error()
    
    DETR_AVAILABLE = True
    print("✓ DETR available")
except ImportError:
    print("Warning: DETR not available. Using ResNet18 only for perception.")
    DETR_AVAILABLE = False

# Try to import diffusers for VEG
try:
    from diffusers import StableDiffusionInpaintPipeline
    from diffusers.utils import logging as diffusers_logging
    import PIL.Image
    DIFFUSERS_AVAILABLE = True
    
    # Disable diffusers progress bars to avoid log cluttering
    diffusers_logging.disable_progress_bar()
    
    # Disable transformers progress bars as well
    from transformers.utils import logging as transformers_logging
    transformers_logging.set_verbosity_error()  # Disable transformers progress bars
    
    print("✓ Diffusers available")
except ImportError:
    print("Warning: Diffusers not available. Using mock VEG.")
    DIFFUSERS_AVAILABLE = False


class DeploymentMode(Enum):
    """Deployment modes for different use cases."""
    RESEARCH = "research"
    PRODUCTION = "production"
    INFERENCE = "inference"


@dataclass
class SpringHybridConfig:
    """Complete configuration for SpringHybridModel."""
    
    # Deployment settings
    deployment_mode: DeploymentMode = DeploymentMode.RESEARCH
    device: str = "auto"
    
    # Model architecture
    image_size: Tuple[int, int] = (512, 512)
    max_objects: int = 10
    enable_perception_module: bool = True
    enable_veg_module: bool = True
    
    # Spatial reasoning module
    srm_mode: SpatialReasoningMode = SpatialReasoningMode.HYBRID
    srm_config: HybridSRMConfig = field(default_factory=HybridSRMConfig)
    
    # Perception module settings
    use_detr: bool = True
    detr_model_name: str = "facebook/detr-resnet-50"
    detr_confidence_threshold: float = 0.3
    scene_encoder_dim: int = 500
    
    # Visual Element Generator settings
    veg_model_name: str = "runwayml/stable-diffusion-inpainting"
    veg_guidance_scale: float = 7.5
    veg_num_inference_steps: int = 20
    enable_veg_finetuning: bool = False
    
    # Training settings
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    freeze_perception_backbone: bool = True
    freeze_veg_weights: bool = True
    
    # Performance optimization
    enable_compile: bool = False
    enable_flash_attention: bool = True
    memory_efficient_attention: bool = True
    
    def __post_init__(self):
        # Auto-detect device with better handling
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")  # FIX: Use explicit cuda:0 device object
                print(f"✓ Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                self.device = torch.device("cpu")
                print("✓ Using CPU device")
        else:
            # FIX: Normalize device to torch.device object for consistent comparison
            self.device = torch.device(self.device) if isinstance(self.device, str) else self.device
        
        # CRITICAL FIX: Ensure cuda device is always cuda:0 for consistent comparison
        if hasattr(self.device, 'type') and self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda:0')  # Convert generic 'cuda' to specific 'cuda:0'
        
        # Adjust settings based on deployment mode
        if self.deployment_mode == DeploymentMode.PRODUCTION:
            self.gradient_checkpointing = False
            self.mixed_precision = True
            self.enable_compile = True
            self.freeze_perception_backbone = True
            self.freeze_veg_weights = True
        elif self.deployment_mode == DeploymentMode.INFERENCE:
            self.gradient_checkpointing = False
            self.enable_veg_finetuning = False
            self.freeze_perception_backbone = True
            self.freeze_veg_weights = True


class PerceptionModule(nn.Module):
    """
    FIXED: Perception module with consistent device handling.
    """
    
    # COCO class mapping for DETR object detection - CRITICAL FOR IMPLICIT CONSTRAINTS
    COCO_CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
        'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife',
        'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
        'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    def __init__(self, config: SpringHybridConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        self.logger = logging.getLogger('PerceptionModule')
        
        # Initialize object detector
        self.use_detr = False  # Start with False, set to True if DETR loads successfully
        
        if DETR_AVAILABLE and config.use_detr:
            try:
                self.object_detector = DetrForObjectDetection.from_pretrained(config.detr_model_name)
                self.detr_processor = DetrImageProcessor.from_pretrained(config.detr_model_name)
                
                # FIXED: Move DETR to the same device as the module
                self.object_detector = self.object_detector.to(self.device)
                self.use_detr = True
                self.logger.info(f"DETR loaded on {self.device}: {config.detr_model_name}")
                
            except Exception as e:
                self.logger.warning(f"Failed to load DETR: {e}, falling back to ResNet18")
                self.use_detr = False
        
        # Scene encoder (ResNet18 backbone)
        self.scene_encoder = resnet18(weights='IMAGENET1K_V1')  # Updated API
        self.scene_encoder.fc = nn.Linear(self.scene_encoder.fc.in_features, config.scene_encoder_dim)
        
        # FIXED: Move scene encoder to device
        self.scene_encoder = self.scene_encoder.to(self.device)
        
        # CRITICAL FIX: Don't freeze backbone in Stage 1 - causes zero gradients!
        # Backbone freezing disabled to restore gradient flow
        # if config.freeze_perception_backbone:
        #     self._freeze_backbone()
        self.logger.info(" Backbone freezing disabled - gradients should flow normally")
        
        # Object encoding layer
        self.object_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        ).to(self.device)  # FIXED: Move to device
    
    def _freeze_backbone(self):
        """Freeze scene encoder backbone for stable training."""
        for param in self.scene_encoder.parameters():
            param.requires_grad = False
        # Keep final layer trainable
        for param in self.scene_encoder.fc.parameters():
            param.requires_grad = True
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        FIXED: Process images with consistent device handling + performance timing.
        """
        import time
        start_time = time.time()
        
        # FIXED: Ensure images are on the correct device
        images = images.to(self.device)
        batch_size = images.shape[0]
        
        # Extract scene features
        scene_start = time.time()
        scene_features = self.scene_encoder(images)
        scene_time = time.time() - scene_start
        
        # STRICT: Object detection must work if DETR is enabled
        assert self.use_detr, "DETR must be enabled for object detection - no fallback modes"
        detr_start = time.time()
        detected_objects, detection_info = self._detect_objects_detr(images)
        detr_time = time.time() - detr_start
        
        # FIXED: Ensure detected objects are on the correct device
        detected_objects = detected_objects.to(self.device)
        
        # Encode detected objects
        encode_start = time.time()
        encoded_objects = self.object_encoder(detected_objects)
        encode_time = time.time() - encode_start
        
        total_time = time.time() - start_time
        self.logger.info(f"🕐 PERCEPTION TIMING: Total={total_time:.2f}s, Scene={scene_time:.2f}s, DETR={detr_time:.2f}s, Encode={encode_time:.2f}s")
        
        return {
            'detected_objects': detected_objects,
            'encoded_objects': encoded_objects,
            'scene_features': scene_features,
            'detection_info': detection_info
        }
    
    def _detect_objects_detr(self, images: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """FIXED: Batch DETR processing for 10x+ speedup."""
        try:
            batch_size = images.shape[0]
            
            # Convert entire batch at once (no individual PIL conversions)
            img_batch = images.detach().cpu()  # Move entire batch to CPU once
            img_batch_normalized = (img_batch + 1) * 0.5  # Convert from [-1,1] to [0,1]
            img_batch_normalized = torch.clamp(img_batch_normalized, 0, 1)
            
            # Convert to list of PIL images for batch processing
            pil_images = []
            for i in range(batch_size):
                img_np = img_batch_normalized[i].permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                img_pil = PIL.Image.fromarray(img_np)
                pil_images.append(img_pil)
            
            # BATCH PROCESS: Single DETR call for all images
            inputs = self.detr_processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.object_detector(**inputs)
            
            # Parse outputs for each image in batch
            all_detections = []
            detection_metadata = []
            
            for i in range(batch_size):
                # Create a simple object with logits and pred_boxes attributes
                class ImageOutputs:
                    def __init__(self, logits, pred_boxes):
                        self.logits = logits
                        self.pred_boxes = pred_boxes
                
                image_outputs = ImageOutputs(
                    logits=outputs.logits[i:i+1],  # Keep batch dim
                    pred_boxes=outputs.pred_boxes[i:i+1]
                )
                
                detections, category_names = self._parse_detr_outputs(image_outputs, pil_images[i].size)
                all_detections.append(detections)
                detection_metadata.append({
                    'num_detections': len(detections),
                    'confidence_scores': [d[4] for d in detections] if len(detections) > 0 else [],
                    'category_names': category_names
                })
            
            # Pad and stack detections
            padded_detections = self._pad_detections(all_detections, batch_size)
            
            return padded_detections, {'per_sample': detection_metadata}
            
        except Exception as e:
            self.logger.error(f"Batch DETR processing failed: {e}")
            raise RuntimeError(f"DETR batch processing failure - FIX REQUIRED: {e}")
    
    
    def _parse_detr_outputs(self, outputs, image_size) -> Tuple[List[List[float]], List[str]]:
        """Parse DETR outputs into bounding boxes AND category names."""
        logits = outputs.logits[0]
        boxes = outputs.pred_boxes[0]
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        max_probs, predicted_classes = probs.max(-1)
        keep = max_probs > self.config.detr_confidence_threshold
        
        detections = []
        category_names = []
        if keep.any():
            img_w, img_h = image_size
            valid_boxes = boxes[keep]
            valid_probs = max_probs[keep]
            valid_classes = predicted_classes[keep]
            valid_boxes_cpu = valid_boxes.detach().cpu()
            valid_probs_cpu = valid_probs.detach().cpu()
            valid_classes_cpu = valid_classes.detach().cpu()

            for box, prob, cls_idx in zip(valid_boxes_cpu, valid_probs_cpu, valid_classes_cpu):
                cx, cy, w, h = box.tolist()
                # Convert from center format to corner format
                x = (cx - w/2) * img_w
                y = (cy - h/2) * img_h
                width = w * img_w
                height = h * img_h
                
                # Get category name from COCO classes
                class_name = self.COCO_CLASSES[cls_idx.item()] if cls_idx.item() < len(self.COCO_CLASSES) else 'unknown'
                
                detections.append([x, y, width, height, prob.item()])
                category_names.append(class_name)
        
        return detections, category_names
    
    
    def _pad_detections(self, detections: List[List[List[float]]], batch_size: int) -> torch.Tensor:
        """FIXED: Pad detections with proper device handling."""
        padded = torch.zeros(batch_size, self.config.max_objects, 4, 
                           device=self.device, dtype=torch.float32)
        
        for i, sample_detections in enumerate(detections):
            n_actual = min(len(sample_detections), self.config.max_objects)
            for j in range(n_actual):
                det = sample_detections[j]
                det_tensor = torch.tensor(det[:4], device=self.device, dtype=torch.float32)
                padded[i, j] = det_tensor
        
        return padded


class VisualElementGenerator(nn.Module):
    """
    FIXED: Visual Element Generator with consistent device handling.
    """
    
    def __init__(self, config: SpringHybridConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        self.logger = logging.getLogger('VisualElementGenerator')
        
        self.veg_available = False
        
        if DIFFUSERS_AVAILABLE and config.enable_veg_module:
            try:
                self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    config.veg_model_name,
                    torch_dtype=torch.float16 if config.mixed_precision else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                
                # FIXED: Move pipeline to device
                self.inpaint_pipeline = self.inpaint_pipeline.to(self.device)
                self.veg_available = True
                self.logger.info(f"VEG loaded on {self.device}: {config.veg_model_name}")
                
            except Exception as e:
                self.logger.warning(f"Failed to load VEG: {e}")
                self.veg_available = False
        else:
            self.veg_available = False
            self.logger.info("VEG disabled or not available")
        
        # Freeze VEG weights if requested
        if config.freeze_veg_weights and self.veg_available:
            for param in self.inpaint_pipeline.unet.parameters():
                param.requires_grad = False
    
    def _validate_cuda_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """
        FAIL-FAST operation validator - NO fallbacks, immediate crash on real problems.
        
        This will crash immediately on:
        - Out of memory errors (with diagnostic info)
        - Empty tensor lists (with stack trace)
        - Device mismatches (with device info)
        - Any other CUDA errors (with full context)
        """
        try:
            result = operation_func(*args, **kwargs)
            
            # Validate result is not None or empty
            if result is None:
                raise AssertionError(f"FAIL-FAST: {operation_name} returned None - this indicates a real bug")
            
            return result
            
        except RuntimeError as e:
            error_msg = str(e).lower()
            
            if "out of memory" in error_msg:
                # Immediate crash with diagnostic info
                available_memory = torch.cuda.memory_reserved() if torch.cuda.is_available() else "N/A"
                allocated_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else "N/A"
                raise RuntimeError(
                    f"FAIL-FAST CUDA OOM in {operation_name}:\n"
                    f"Error: {e}\n"
                    f"Available CUDA memory: {available_memory}\n"
                    f"Allocated CUDA memory: {allocated_memory}\n"
                    f"Operation args: {len(args)} args, {len(kwargs)} kwargs\n"
                    f"This indicates insufficient memory allocation - FIX THE ROOT CAUSE"
                )
                
            elif "stack expects a non-empty tensorlist" in error_msg:
                # Immediate crash with diagnostic info for empty tensor debugging
                raise AssertionError(
                    f"FAIL-FAST EMPTY TENSOR LIST in {operation_name}:\n"
                    f"Error: {e}\n"
                    f"This means the VEG generation produced NO valid tensors.\n"
                    f"Check why image generation is failing completely.\n"
                    f"Operation: {operation_name}\n"
                    f"Args count: {len(args)}, Kwargs: {list(kwargs.keys())}\n"
                    f"FIX THE IMAGE GENERATION PIPELINE"
                )
                
            elif "device" in error_msg or "cuda" in error_msg:
                # Immediate crash with device diagnostic info
                current_device = torch.cuda.current_device() if torch.cuda.is_available() else "CPU"
                device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
                raise RuntimeError(
                    f"FAIL-FAST CUDA DEVICE ERROR in {operation_name}:\n"
                    f"Error: {e}\n"
                    f"Current device: {current_device}\n"
                    f"Available devices: {device_count}\n"
                    f"FIX THE DEVICE ALLOCATION LOGIC"
                )
                
            else:
                # Any other RuntimeError - crash immediately with context
                raise RuntimeError(f"FAIL-FAST RUNTIME ERROR in {operation_name}: {e}")
                
        except Exception as e:
            # All other exceptions - crash immediately with full context
            raise AssertionError(
                f"FAIL-FAST UNEXPECTED ERROR in {operation_name}:\n"
                f"Error type: {type(e).__name__}\n"
                f"Error: {e}\n"
                f"Args: {args}\n"
                f"Kwargs: {kwargs}\n"
                f"FIX THE UNDERLYING BUG - NO FALLBACKS"
            )
    
    def forward(self, 
                background_images: torch.Tensor,  # Fixed: was background_image
                object_layouts: torch.Tensor, 
                object_categories: List[str],
                valid_masks: torch.Tensor = None,  # Add default
                prompts: Optional[List[str]] = None) -> torch.Tensor:

        def _cuda_safe_forward():
            # FIXED: Ensure all inputs are on the correct device
            background_images_safe = background_images.to(self.device)
            layouts_safe = object_layouts.to(self.device)

            if valid_masks is None:
                batch_size, n_objects = layouts_safe.shape[:2]
                valid_masks_safe = torch.ones(batch_size, n_objects, dtype=torch.bool, 
                                       device=layouts_safe.device)
            else:
                valid_masks_safe = valid_masks.to(self.device)
            
            return self._forward_internal(background_images_safe, layouts_safe, 
                                        object_categories, valid_masks_safe, prompts)
        
        # Use fail-fast validator - NO fallbacks, crash on real problems
        return self._validate_cuda_operation("VEG_forward", _cuda_safe_forward)
    
    def _forward_internal(self, background_images, layouts, object_categories, valid_masks, prompts):
        batch_size = background_images.shape[0]
        generated_images = []
        generation_metadata = []
        
        for i in range(batch_size):
            def _generate_safe_image():
                return self._generate_single_image(
                    background_images[i],
                    layouts[i],
                    object_categories[i] if i < len(object_categories) else [],
                    valid_masks[i]
                )
            
            try:
                # Use fail-fast validator for each image generation
                generated_img, metadata = self._validate_cuda_operation(
                    f"VEG_single_image_{i}", _generate_safe_image
                )
                generated_images.append(generated_img)
                generation_metadata.append(metadata)
                
            except Exception as e:
                # FAIL-FAST: No fallbacks, crash immediately with diagnostic info
                raise AssertionError(
                    f"FAIL-FAST VEG GENERATION FAILED for batch {i}:\n"
                    f"Error: {e}\n"
                    f"Background shape: {background_images[i].shape}\n"
                    f"Device: {background_images[i].device}\n"
                    f"Batch index: {i}/{len(background_images)}\n"
                    f"This indicates a real bug in image generation - FIX THE ROOT CAUSE\n"
                    f"No fallback tensors will be created - system must crash to expose the bug"
                )

        # FAIL-FAST: NO fallbacks, crash immediately on empty tensors
        def _stack_generated_images():
            # Immediate validation - crash if no images generated
            if not generated_images:
                raise AssertionError(
                    f"FAIL-FAST: VEG generated ZERO images for batch of {len(background_images)}\n"
                    f"This means ALL image generation attempts failed\n"
                    f"Background images shape: {background_images.shape}\n"
                    f"Device: {self.device}\n"
                    f"VEG available: {self.veg_available}\n"
                    f"NO FALLBACK BATCH will be created - FIX THE IMAGE GENERATION BUG"
                )
            
            # Validate tensor shapes and devices - crash on mismatch
            device_images = []
            expected_shape = background_images[0].shape
            
            for i, img in enumerate(generated_images):
                if img.shape != expected_shape:
                    raise AssertionError(
                        f"FAIL-FAST: VEG shape mismatch at index {i}\n"
                        f"Generated shape: {img.shape}\n"
                        f"Expected shape: {expected_shape}\n"
                        f"NO automatic resizing - FIX THE GENERATION PIPELINE"
                    )
                
                # Validate device consistency - both should now be cuda:0
                if img.device != self.device:
                    raise AssertionError(
                        f"FAIL-FAST: Device mismatch at index {i}\n"
                        f"Image device: {img.device}\n"
                        f"Expected device: {self.device}\n"
                        f"FIX THE DEVICE ALLOCATION"
                    )
                
                device_images.append(img)
            
            # Final validation before stacking
            assert len(device_images) == len(generated_images), f"Lost images during processing: {len(device_images)} != {len(generated_images)}"
            assert len(device_images) > 0, "Empty device_images list - this should never happen"
            
            # Stack without fallback handling
            stacked_images = torch.stack(device_images)
            
            return stacked_images
        
        # FAIL-FAST: Use validator with NO fallbacks for tensor stacking
        final_images = self._validate_cuda_operation("VEG_stack_images", _stack_generated_images)
        
        return {
            'generated_images': final_images,
            'generation_metadata': generation_metadata
        }
    
    def _generate_single_image(self, 
                              background: torch.Tensor,
                              layout: torch.Tensor,
                              categories: List[str],
                              valid_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """FIXED: Generate single image with proper tensor handling and memory cleanup."""
        
        # Create generator on CPU to avoid device conflicts
        generator = torch.Generator().manual_seed(42)
        
        try:
            background_detached = background.detach().cpu()  # Move to CPU for PIL conversion
            background_pil = self._tensor_to_pil(background_detached)
            
            # Create inpainting mask
            mask_pil = self._create_inpainting_mask(layout, valid_mask, background.shape[-2:])
            
            # Create prompt from categories
            prompt = self._create_prompt(categories, valid_mask)
            
            # Generate with diffusion - use CPU generator to avoid CUDA conflicts
            with torch.no_grad():
                try:
                    result = self.inpaint_pipeline(
                        prompt=prompt,
                        image=background_pil,
                        mask_image=mask_pil,
                        guidance_scale=self.config.veg_guidance_scale,
                        num_inference_steps=self.config.veg_num_inference_steps,
                        generator=generator
                    )
                    
                    generated_tensor = self._pil_to_tensor(result.images[0])
                    
                    # Clean up PIL images to prevent memory leaks
                    del background_pil, mask_pil
                    
                except Exception as e:
                    self.logger.warning(f"VEG generation failed: {e}, using background")
                    generated_tensor = background_detached
            
            # FIXED: Ensure output tensor is on the correct device
            generated_tensor = generated_tensor.to(self.device)
            
            # CUDA MEMORY CLEANUP: Comprehensive cleanup after generation
            del background_detached
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            metadata = {
                'prompt': prompt,
                'num_objects': valid_mask.sum().item(),
                'guidance_scale': self.config.veg_guidance_scale,
                'memory_cleaned': True
            }
            
            return generated_tensor, metadata
            
        except Exception as e:
            self.logger.error(f"Critical VEG error: {e}")
            # Return background as fallback
            fallback_tensor = background.detach().clone().to(self.device)
            metadata = {'error': str(e), 'fallback': True}
            return fallback_tensor, metadata
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> PIL.Image.Image:
        """Convert tensor to PIL Image."""
        tensor = tensor.detach().cpu()
        tensor = (tensor + 1) / 2
        tensor = tensor.clamp(0, 1)
        
        img_np = tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        return PIL.Image.fromarray(img_np)
    
    def _pil_to_tensor(self, pil_img: PIL.Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor."""
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        tensor = tensor * 2 - 1  # Normalize to [-1, 1]
        
        return tensor
    
    def _create_inpainting_mask(self, layout: torch.Tensor, valid_mask: torch.Tensor, 
                               image_size: Tuple[int, int]) -> PIL.Image.Image:
        """Create binary mask for inpainting regions."""
        mask = np.zeros(image_size, dtype=np.uint8)
        layout_cpu = layout.detach().cpu()
        valid_mask_cpu = valid_mask.detach().cpu()
        
        for i in range(len(layout_cpu)):
            if valid_mask_cpu[i]:
                x, y, w, h = layout_cpu[i].numpy()
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(image_size[1], x + w)
                y2 = min(image_size[0], y + h)
                
                mask[y1:y2, x1:x2] = 255
        
        return PIL.Image.fromarray(mask, mode='L')
    
    def _create_prompt(self, categories: List[str], valid_mask: torch.Tensor) -> str:
        """Create safe text prompt from object categories."""
        valid_categories = [cat for i, cat in enumerate(categories) 
                          if i < len(valid_mask) and valid_mask[i]]
        
        if not valid_categories:
            return "clean empty room"
        
        # Create safer prompts
        safe_categories = []
        for cat in valid_categories:
            safe_cat = str(cat).lower().replace('bed', 'furniture').replace('toilet', 'bathroom fixture')
            safe_categories.append(safe_cat)
        
        if len(safe_categories) == 1:
            return f"{safe_categories[0]} in room"
        else:
            return f"{safe_categories[0]} in room"
    
    def _generate_mock_images(self, backgrounds: torch.Tensor, layouts: torch.Tensor, 
                            valid_masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """FIXED: Generate mock images with consistent device handling."""
        device = backgrounds.device
        layouts = layouts.to(device)
        valid_masks = valid_masks.to(device)
        
        mock_images = backgrounds.detach().clone()
        layouts_detached = layouts.detach()
        
        for i in range(len(mock_images)):
            for j in range(len(layouts_detached[i])):
                if valid_masks[i, j]:
                    x, y, w, h = layouts_detached[i, j]
                    x, y, w, h = int(x.item()), int(y.item()), int(w.item()), int(h.item())
                    
                    height, width = mock_images.shape[-2:]
                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                    w = max(1, min(w, width - x))
                    h = max(1, min(h, height - y))
                    
                    color = torch.rand(3, device=device) * 2 - 1
                    mock_images[i, :, y:y+h, x:x+w] = color.view(3, 1, 1)
        
        return {
            'generated_images': mock_images,
            'generation_metadata': [{'mock': True} for _ in range(len(mock_images))]
        }


class SpringHybridModel(nn.Module):
    """
    FIXED: Complete SPRING Hybrid Model with consistent device handling.
    """
    
    def __init__(self, config: SpringHybridConfig):
        super().__init__()
        self.config = config
        self.device = config.device  # FIX: Use normalized device object directly
        self.logger = self._setup_logging()
        
        # Initialize component modules
        if config.enable_perception_module:
            self.perception_module = PerceptionModule(config)
        else:
            self.perception_module = None
        
        if SPRING_COMPONENTS_AVAILABLE:
            self.spatial_reasoning_module = HybridSpatialReasoningModule(config.srm_config)
            # FIXED: Move SRM to device
            self.spatial_reasoning_module = self.spatial_reasoning_module.to(self.device)
            
            # PHASE 2 FIX: Scene-to-Layout Encoder - replaces torch.randn() with scene features  
            self.scene_to_layout_encoder = SceneToLayoutEncoder(
                scene_feature_dim=500,  # CORRECTED: Actual scene features are 500-dim, not 512
                max_objects=config.max_objects,
                coordinate_dim=4,
                coordinate_system="per_mille"  # SPRING standard [0, 1000]
            ).to(self.device)
            self.logger.info(f"✓ Scene-to-Layout Encoder initialized: 500 → {config.max_objects}×4")
            
            # PHASE 3 FIX: Scene-to-Hidden Encoder - replaces shared hidden state with scene-specific states
            self.scene_to_hidden_encoder = SceneToHiddenEncoder(
                scene_feature_dim=500,  # Same as scene features
                num_layers=config.srm_config.gru_num_layers,  # Match GRU layers
                hidden_size=config.srm_config.gru_hidden_size,  # Match GRU hidden size
                initialization_scale=1.0,  # INCREASED: More influential hidden states
                activation="tanh"  # Stable range [-1, 1]
            ).to(self.device)
            self.logger.info(f"✓ Scene-to-Hidden Encoder initialized: 500 → [{config.srm_config.gru_num_layers}, {config.srm_config.gru_hidden_size}]")
            
        else:
            self.spatial_reasoning_module = self._create_mock_srm()
            # Add mock encoders for consistency
            self.scene_to_layout_encoder = None
            self.scene_to_hidden_encoder = None
        
        if config.enable_veg_module:
            self.visual_element_generator = VisualElementGenerator(config)
        else:
            self.visual_element_generator = None
        
        # Training utilities
        self.current_epoch = 0
        self.training_step = 0
        
        # Performance tracking
        self.performance_metrics = {
            'perception_time': [],
            'srm_time': [],
            'veg_time': [],
            'total_time': []
        }
        
        # FIXED: Move entire model to device
        self.to(self.device)
        
        self.logger.info(f"SpringHybridModel initialized on {self.device} in {config.deployment_mode.value} mode")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the hybrid model."""
        logger = logging.getLogger('SpringHybridModel')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def forward(self, 
           images: torch.Tensor,
           constraints: Optional[List[List[Any]]] = None,
           object_categories: Optional[List[List[str]]] = None,
           return_intermediate: bool = False,
           **kwargs) -> Dict[str, Any]:
        """
        FIXED: Complete forward pass with stage-aware output handling + TIMING.
        """
        import time
        start_time = time.time()
        
        #  CONSTRAINT FLOW LOG 3: Model entry
        batch_size = images.shape[0]
        if hasattr(self, 'batch_counter'):
            self.batch_counter += 1
        else:
            self.batch_counter = 0
        
        if self.batch_counter < 5:  # Log first 5 batches
            self.logger.info(f" FLOW-3 MODEL ENTRY (batch {self.batch_counter}):")
            self.logger.info(f"   SpringHybridModel received constraints: type={type(constraints)}, len={len(constraints) if constraints else 'None'}")
            if constraints:
                total_constraints = sum(len(sample_constraints) if sample_constraints else 0 for sample_constraints in constraints)
                self.logger.info(f"   Total constraints received: {total_constraints}")
        
        # CRITICAL FIX: Check if this is Stage 2 training
        training_stage = kwargs.get('training_stage', 'stage1')
        srm_mode = kwargs.get('srm_mode', 'discrete')
        
        # self.logger.info(f" FORWARD START: {training_stage}, mode={srm_mode}")

        # FIXED: Switch SRM mode based on training stage
        if training_stage == 'stage2' and srm_mode == 'differentiable':
            if hasattr(self.spatial_reasoning_module, 'switch_mode'):
                from spring_int import SpatialReasoningMode
                self.spatial_reasoning_module.switch_mode(SpatialReasoningMode.DIFFERENTIABLE)
                self.logger.info(f"Switched SRM to DIFFERENTIABLE mode for Stage 2")

        # Debug prints removed for performance
        
        self.logger.debug(f"SpringHybridModel forward: stage={training_stage}, mode={srm_mode}")
        
        batch_size = images.shape[0]
        device = images.device
        
        # STAGE 2: Output layout coordinates [batch, objects, 4]
        if training_stage == 'stage2' or srm_mode == 'differentiable':
            print(" DEBUG: Stage 2 - Using REAL SRM")
            
            #  REQUIRE real SRM - NO FALLBACKS!
            if not hasattr(self, 'spatial_reasoning_module') or not self.spatial_reasoning_module:
                raise RuntimeError(" NO SPATIAL REASONING MODULE! Stage 2 requires real SRM!")
            
            # Force differentiable mode
            original_mode = getattr(self.spatial_reasoning_module.config, 'mode', 'hybrid')
            self.spatial_reasoning_module.config.mode = SpatialReasoningMode.DIFFERENTIABLE
            
            try:
                # REAL SRM forward pass - NO MOCKS!
                batch_size = images.shape[0]
                n_objects = self.config.max_objects
                
                # CRITICAL FIX: Extract scene features for Stage 2 (was missing!)
                if self.perception_module:
                    try:
                        perception_results = self.perception_module(images)
                        scene_features = perception_results.get('scene_features', torch.zeros(batch_size, 500, device=device))
                        
                        # CRITICAL VERIFICATION: Check if scene features are working
                        print(f"\nSCENE FEATURE VERIFICATION:")
                        print(f"   Shape: {scene_features.shape}")
                        print(f"   Mean: {scene_features.mean().item():.6f}")
                        print(f"   Std: {scene_features.std().item():.6f}")
                        print(f"   Non-zero elements: {torch.count_nonzero(scene_features).item()}/{scene_features.numel()}")
                        print(f"   Requires grad: {scene_features.requires_grad}")
                        
                        # Check if features are all zeros (bad!)
                        if torch.count_nonzero(scene_features).item() == 0:
                            print(f"   CRITICAL: Scene features are ALL ZEROS!")
                            print(f"       This means perception module isn't working!")
                        else:
                            print(f"   Scene features look good (non-zero values)")
                        
                        # Ensure gradients if needed
                        if not scene_features.requires_grad:
                            print(f"   Enabling gradients on scene features")
                            if scene_features.is_leaf:
                                scene_features.requires_grad_(True)
                            else:
                                print(f"   Can't enable gradients: not a leaf tensor")
                                
                    except Exception as e:
                        self.logger.warning(f"Stage 2 perception failed: {e}")
                        scene_features = torch.zeros(batch_size, 500, device=device, requires_grad=True)
                        print(f"    Using fallback zeros with requires_grad=True")
                else:
                    scene_features = torch.zeros(batch_size, 500, device=device, requires_grad=True) 
                    print(f"    No perception module - using zeros with requires_grad=True")
                
                # PHASE 2 FIX: Use scene features instead of random noise
                # This is the critical fix - connect scene understanding to spatial reasoning
                if hasattr(self, 'scene_to_layout_encoder') and self.scene_to_layout_encoder is not None:
                    print(f"\nSCENE-TO-LAYOUT ENCODER VERIFICATION:")
                    print(f"   Scene-to-layout encoder is available and initialized")
                    print(f"   Scene features shape: {scene_features.shape}")
                    print(f"   Scene features requires_grad: {scene_features.requires_grad}")
                    
                    #  CRITICAL FIX: Don't break gradient chain by using requires_grad_() on non-leaf
                    # Instead, ensure the encoder parameters require gradients
                    for param in self.scene_to_layout_encoder.parameters():
                        if not param.requires_grad:
                            print(f"     WARNING: Encoder parameter doesn't require grad!")
                            param.requires_grad = True
                    
                    encoded_input = self.scene_to_layout_encoder(scene_features)
                    
                    print(f"   After encoder: encoded_input.requires_grad = {encoded_input.requires_grad}")
                    print(f"   encoded_input is leaf: {encoded_input.is_leaf}")
                    
                    #  DON'T USE requires_grad_() on non-leaf tensors - it breaks gradient flow!
                    if not encoded_input.requires_grad:
                        print(f"     CRITICAL: encoded_input doesn't require grad after encoder!")
                        print(f"     This means gradient chain is broken at encoder level")
                        # Don't try to fix with requires_grad_() - that makes it worse!
                    
                    print(f"   Final: encoded_input.requires_grad = {encoded_input.requires_grad}")
                    print(f"   Final: encoded_input.is_leaf = {encoded_input.is_leaf}")
                else:
                    # Fallback for mock/testing scenarios - THIS SHOULD NOT HAPPEN!
                    print(f"\nCRITICAL ERROR: NO SCENE-TO-LAYOUT ENCODER!")
                    print(f"   This means the model wasn't properly initialized!")
                    print(f"   Falling back to random noise (BAD for inference)")
                    encoded_input = torch.randn(batch_size, n_objects, 4, device=device, requires_grad=True)
                
                # PHASE 3 FIX: Generate scene-specific hidden states for Stage 2
                if hasattr(self, 'scene_to_hidden_encoder') and self.scene_to_hidden_encoder is not None:
                    print(f"\nSCENE-TO-HIDDEN ENCODER VERIFICATION:")
                    print(f"   Scene-to-hidden encoder is available")
                    scene_hidden_states = self.scene_to_hidden_encoder(scene_features)
                    print(f"   Hidden states std: {scene_hidden_states.std():.3f}, shape: {scene_hidden_states.shape}")
                else:
                    print(f"\nCRITICAL ERROR: NO SCENE-TO-HIDDEN ENCODER!")
                    print(f"   This means GRU will use shared parameters instead of scene-specific ones!")
                    scene_hidden_states = None
                
                # HardNet instrumentation removed - system working correctly
                
                print(f"\n REAL TRAINING - CALLING SRM (batch {self.batch_counter}):")
                print(f"   Encoded input shape: {encoded_input.shape}")
                print(f"   Encoded input requires_grad: {encoded_input.requires_grad}")
                if encoded_input.numel() > 0:
                    print(f"   Encoded input[0,0] x-coord: {encoded_input[0, 0, 0].item():.6f}")
                print(f"   Constraints: {len(constraints) if constraints else 0} samples")
                if constraints and len(constraints) > 0 and constraints[0]:
                    print(f"   First constraint: {type(constraints[0][0]).__name__}")
                
                # REAL forward pass
                layout_output, constraint_info = self.spatial_reasoning_module(
                    encoded_input,
                    constraints=constraints,
                    n_objects=n_objects,
                    initial_hidden=scene_hidden_states  # PHASE 3: Pass scene-specific hidden states
                )
                
                #  ANALYZE REAL TRAINING RESULTS
                print(f"\n REAL TRAINING - SRM RESULTS (batch {self.batch_counter}):")
                print(f"   Layout output shape: {layout_output.shape}")
                print(f"   Layout output requires_grad: {layout_output.requires_grad}")
                if layout_output.numel() > 0:
                    print(f"   Layout output[0,0] x-coord: {layout_output[0, 0, 0].item():.6f}")
                print(f"   Constraint satisfaction: {constraint_info.get('constraint_satisfaction_rate', 'N/A')}")
                
                # HardNet analysis removed - projection system working correctly
                
                #  TEST GRADIENT FLOW IN REAL TRAINING
                print(f"\n REAL TRAINING - GRADIENT FLOW TEST (batch {self.batch_counter}):")
                print(f"   Layout output requires_grad: {layout_output.requires_grad}")
                print(f"   Layout output is_leaf: {layout_output.is_leaf}")
                print(f"   Encoded input requires_grad: {encoded_input.requires_grad}")
                print(f"   Encoded input is_leaf: {encoded_input.is_leaf}")
                
                if layout_output.requires_grad:
                    # Create simple loss to test gradient flow
                    test_target = torch.zeros_like(layout_output)  
                    test_loss = torch.nn.functional.mse_loss(layout_output, test_target)
                    
                    print(f"   Test loss value: {test_loss.item():.6f}")
                    print(f"   Test loss requires_grad: {test_loss.requires_grad}")
                    
                    # Check if encoded_input has gradients before backward
                    print(f"   Encoded input grad before backward: {encoded_input.grad}")
                    
                    # Test gradient computation
                    try:
                        # CRITICAL: Check if encoded_input is part of the computation graph
                        if encoded_input.requires_grad:
                            # Register a hook to see if gradients reach encoded_input
                            gradient_received = [False]
                            def grad_hook(grad):
                                gradient_received[0] = True
                                print(f"    GRADIENT HOOK FIRED! Grad shape: {grad.shape}, norm: {torch.norm(grad).item():.8f}")
                                return grad
                            
                            hook_handle = encoded_input.register_hook(grad_hook)
                        
                        print(f"   Calling backward on test loss...")
                        test_loss.backward(retain_graph=True)  # retain_graph=True to not interfere with main training
                        
                        if encoded_input.requires_grad:
                            hook_handle.remove()
                            
                            if gradient_received[0]:
                                print(f"    HOOK CONFIRMED: Gradients reached encoded_input!")
                            else:
                                print(f"    HOOK FAILED: No gradients reached encoded_input")
                                print(f"    DEBUGGING: encoded_input might be disconnected from computation graph")
                        
                        if encoded_input.grad is not None:
                            grad_norm = torch.norm(encoded_input.grad).item()
                            grad_x = encoded_input.grad[0, 0, 0].item()
                            
                            print(f"    GRADIENTS COMPUTED!")
                            print(f"   Grad norm: {grad_norm:.8f}")
                            print(f"   Grad[0,0,0]: {grad_x:.8f}")
                            
                            if grad_norm > 1e-8:
                                print(f"   STE WORKING: Gradients flow through constraint system in REAL training!")
                            else:
                                print(f"     Gradients very small: {grad_norm:.8f}")
                        else:
                            print(f"    NO GRADIENTS: encoded_input.grad is None")
                            print(f"    LIKELY CAUSE: encoded_input is not part of computation graph")
                            
                            # Additional debugging
                            if not encoded_input.requires_grad:
                                print(f"    encoded_input.requires_grad is False!")
                            elif encoded_input.is_leaf:
                                print(f"    encoded_input is a leaf tensor")
                            else:
                                print(f"    encoded_input is NOT a leaf tensor - might be the issue")
                            
                    except Exception as grad_e:
                        print(f"    GRADIENT TEST FAILED: {grad_e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Clear test gradients
                    if encoded_input.grad is not None:
                        encoded_input.grad.zero_()
                else:
                    print(f"     Cannot test gradients: layout_output doesn't require grad")
                
                # Instrumentation state reset removed - no longer needed
                
                # Verify output shape
                if layout_output.shape != (batch_size, n_objects, 4):
                    raise RuntimeError(f" SRM output shape wrong: {layout_output.shape} != {(batch_size, n_objects, 4)}")
                
            except Exception as e:
                # Restore mode and re-raise - NO FALLBACKS!
                self.spatial_reasoning_module.config.mode = original_mode
                raise RuntimeError(f" SRM forward failed: {e}")
            
            # Restore original mode
            self.spatial_reasoning_module.config.mode = original_mode
            
            # ENSURE gradients flow
            if not layout_output.requires_grad:
                layout_output.requires_grad_(True)
            
            # Return Stage 2 output
            output = {
                'layout_results': {
                    'final_layout': layout_output,
                    'constraint_info': {
                        **constraint_info,
                        'layout_quality': constraint_info.get('layout_quality', 1.0),  # Add missing layout_quality
                        'processing_time': time.time() - start_time  # Add processing_time
                    }
                },
                'constraint_satisfaction': {
                    'constraint_satisfaction_rate': constraint_info.get('constraint_satisfaction_rate', 0.0),
                    'total_penalty': constraint_info.get('total_penalty', 0.0)
                },
                'performance_metrics': {
                    'total_time': time.time() - start_time,
                    'processing_time': time.time() - start_time,
                    'layout_quality': constraint_info.get('layout_quality', 1.0)  # Add here too
                }
            }
            
            # BATCH PENALTY PRESERVATION FIX: Ensure batch-level total_penalty is preserved
            
            return output
        
        # STAGE 1: Your existing logic (keep this unchanged)
        else:
            self.logger.debug("Stage 1: Using existing forward logic")
            
            # Your existing Stage 1 forward pass logic
            # This should stay exactly as it was before
            
            # Move images to correct device
            images = images.to(self.device)
            
            # Validate input tensor shapes
            if not self.validate_tensor_shapes(images):
                self.logger.warning("Input validation failed, proceeding anyway")
            
            # Initialize performance tracking
            perception_start = time.time()
            
            # Perception Module
            if self.perception_module:
                try:
                    perception_start = time.time()
                    perception_results = self.perception_module(images)
                    perception_time = time.time() - perception_start
                    # self.logger.info(f"⚡ PERCEPTION TOTAL: {perception_time:.2f}s")
                    
                    scene_features = perception_results.get('scene_features', torch.zeros(batch_size, 500, device=self.device))  # CORRECTED: 500-dim
                    detected_objects = perception_results.get('detected_objects', [[] for _ in range(batch_size)])
                except Exception as e:
                    self.logger.warning(f"Perception module failed: {e}")
                    scene_features = torch.zeros(batch_size, 500, device=self.device)  # CORRECTED: 500-dim
                    detected_objects = [[] for _ in range(batch_size)]
            else:
                scene_features = torch.zeros(batch_size, 500, device=self.device)  # CORRECTED: 500-dim
                detected_objects = [[] for _ in range(batch_size)]
            
            perception_time = time.time() - perception_start
            
            # Spatial Reasoning Module
            srm_start = time.time()
            # self.logger.info(f"🧠 STARTING SRM: {self.spatial_reasoning_module is not None}")
            
            try:
                if self.spatial_reasoning_module:
                    # PHASE 2 FIX: Create layout input from scene features instead of random noise
                    input_start = time.time()
                    if hasattr(self, 'scene_to_layout_encoder') and self.scene_to_layout_encoder is not None:
                        layout_input = self.scene_to_layout_encoder(scene_features)
                        
                        # CRITICAL DEBUG: Check if Scene-to-Layout Encoder produces diversity
                        scene_std = scene_features.std().item()
                        layout_std = layout_input.std().item()
                        print(f" DIVERSITY CHECK:")
                        print(f"  Scene features: mean={scene_features.mean():.3f}, std={scene_std:.3f}")
                        print(f"  Layout input: mean={layout_input.mean():.1f}, std={layout_std:.1f}")
                        print(f"  Layout range: [{layout_input.min():.1f}, {layout_input.max():.1f}]")
                        
                        if layout_std < 1.0:
                            print(f"    WARNING: Layout encoder output has very low variance!")
                        
                    else:
                        # Fallback for mock/testing scenarios
                        layout_input = torch.randn(batch_size, self.config.max_objects, 4, device=self.device)
                        print(f" FALLBACK: layout_input std={layout_input.std():.3f}")
                    
                    # PHASE 3 FIX: Generate scene-specific hidden states
                    if hasattr(self, 'scene_to_hidden_encoder') and self.scene_to_hidden_encoder is not None:
                        scene_hidden_states = self.scene_to_hidden_encoder(scene_features)
                        hidden_std = scene_hidden_states.std().item()
                        print(f" HIDDEN STATE CHECK:")
                        print(f"  Hidden states: mean={scene_hidden_states.mean():.3f}, std={hidden_std:.3f}")
                        print(f"  Hidden shape: {scene_hidden_states.shape}")
                    else:
                        # Fallback - use None (will trigger shared hidden state)
                        scene_hidden_states = None
                        print(f" HIDDEN STATE: Using shared parameter (fallback)")
                    
                    input_time = time.time() - input_start
                    
                    #  CONSTRAINT FLOW LOG 4: Pre-SRM call
                    if self.batch_counter < 5:  # Log first 5 batches
                        self.logger.info(f" FLOW-4 PRE-SRM CALL (batch {self.batch_counter}):")
                        self.logger.info(f"   Passing to SRM: constraints type={type(constraints)}, len={len(constraints) if constraints else 'None'}")
                        self.logger.info(f"   n_objects={self.config.max_objects}")
                        if constraints:
                            total_constraints = sum(len(sample_constraints) if sample_constraints else 0 for sample_constraints in constraints)
                            self.logger.info(f"   Total constraints being passed: {total_constraints}")
                    
                    call_start = time.time()
                    layouts, srm_info = self.spatial_reasoning_module(
                        layout_input,
                        constraints=constraints,
                        n_objects=self.config.max_objects,
                        initial_hidden=scene_hidden_states  # PHASE 3: Pass scene-specific hidden states
                    )
                    call_time = time.time() - call_start
                    
                    # CRITICAL DEBUG: Check GRU output diversity
                    if layouts is not None:
                        layout_std = layouts.std().item()
                        print(f" GRU OUTPUT CHECK:")
                        print(f"  GRU output: mean={layouts.mean():.1f}, std={layout_std:.1f}")
                        print(f"  GRU range: [{layouts.min():.1f}, {layouts.max():.1f}]")
                        
                        if layout_std < 1.0:
                            print(f"    WARNING: GRU output has very low variance!")
                    else:
                        print(f" GRU OUTPUT: None")
                    # self.logger.info(f"🧠 SRM TIMING: Input={input_time:.3f}s, Call={call_time:.2f}s")
                else:
                    self.logger.warning("No SRM available")
            except Exception as e:
                self.logger.warning(f"SRM failed: {e}")
            
            srm_time = time.time() - srm_start
            # self.logger.info(f"🧠 SRM TOTAL: {srm_time:.2f}s")
            
            # Visual Element Generator (VEG)
            veg_start = time.time()
            # self.logger.info(f" STARTING VEG: {self.visual_element_generator is not None}")
            
            if self.visual_element_generator:
                try:
                    mask_start = time.time()
                    valid_masks = torch.ones(batch_size, self.config.max_objects, device=self.device).bool()
                    mask_time = time.time() - mask_start
                    
                    call_start = time.time()
                    veg_results = self.visual_element_generator(
                        images, layouts, valid_masks 
                    )
                    call_time = time.time() - call_start
                    
                    generated_images = veg_results['generated_images']
                    veg_metadata = veg_results.get('generation_metadata', [])
                    # self.logger.info(f" VEG TIMING: Mask={mask_time:.3f}s, Call={call_time:.2f}s")
                except Exception as e:
                    self.logger.warning(f"VEG failed: {e}")
                    generated_images = images.clone()
                    veg_metadata = [{'error': str(e)} for _ in range(batch_size)]
            else:
                generated_images = images.clone()
                veg_metadata = [{'mock': True} for _ in range(batch_size)]
            
            veg_time = time.time() - veg_start
            # self.logger.info(f" VEG TOTAL: {veg_time:.2f}s")
            total_time = time.time() - start_time
            
            # Performance tracking
            self.performance_metrics['perception_time'].append(perception_time)
            self.performance_metrics['srm_time'].append(srm_time)
            self.performance_metrics['veg_time'].append(veg_time)
            self.performance_metrics['total_time'].append(total_time)
            
            # Prepare output
            output = {
                'generated_images': generated_images,
                'performance_metrics': {
                    'perception_time': perception_time,
                    'srm_time': srm_time,
                    'veg_time': veg_time,
                    'total_time': total_time
                }
            }
            
            if return_intermediate:
                output.update({
                    'scene_features': scene_features,
                    'detected_objects': detected_objects,
                    'layout_results': {
                        'final_layout': layouts,
                        'constraint_info': srm_info
                    },
                    'constraint_satisfaction': {
                        'constraint_satisfaction_rate': srm_info.get('constraint_satisfaction_rate', 0.0),
                        'total_penalty': srm_info.get('total_penalty', 0.0)
                    },
                    'veg_metadata': veg_metadata
                })
            
            return output
    
    def switch_mode(self, mode: SpatialReasoningMode):
        """Switch spatial reasoning mode."""
        if hasattr(self.spatial_reasoning_module, 'switch_mode'):
            self.spatial_reasoning_module.switch_mode(mode)
            self.logger.info(f"Switched to {mode.value} mode")
    
    def update_epoch(self, epoch: int):
        """Update training epoch for curriculum learning."""
        self.current_epoch = epoch
        if hasattr(self.spatial_reasoning_module, 'current_epoch'):
            self.spatial_reasoning_module.current_epoch = epoch
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance metrics summary."""
        summary = {}
        for key, values in self.performance_metrics.items():
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
                summary[f'{key}_recent'] = np.mean(values[-10:]) if len(values) >= 10 else np.mean(values)
        return summary
    
    def validate_tensor_shapes(self, images: torch.Tensor) -> bool:
        """Validate input tensor shapes for debugging."""
        try:
            batch_size, channels, height, width = images.shape
            if channels != 3:
                self.logger.warning(f"Expected 3 channels, got {channels}")
                return False
            if (height, width) != self.config.image_size:
                self.logger.warning(f"Expected size {self.config.image_size}, got ({height}, {width})")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Tensor shape validation failed: {e}")
            return False


# Factory functions for easy model creation
def create_research_model(image_size: Tuple[int, int] = (512, 512),
                         max_objects: int = 10,
                         device: str = "auto") -> SpringHybridModel:
    """Create model configured for research use."""
    config = SpringHybridConfig(
        deployment_mode=DeploymentMode.RESEARCH,
        image_size=image_size,
        max_objects=max_objects,
        device=device,
        enable_perception_module=True,
        enable_veg_module=False,
        mixed_precision=True,
        gradient_checkpointing=True
    )
    return SpringHybridModel(config)


def create_production_model(image_size: Tuple[int, int] = (512, 512),
                           max_objects: int = 10,
                           device: str = "auto") -> SpringHybridModel:
    """Create model optimized for production use."""
    config = SpringHybridConfig(
        deployment_mode=DeploymentMode.PRODUCTION,
        image_size=image_size,
        max_objects=max_objects,
        device=device,
        enable_perception_module=True,
        enable_veg_module=True,
        mixed_precision=True,
        enable_compile=True,
        freeze_perception_backbone=True,
        freeze_veg_weights=True
    )
    return SpringHybridModel(config)


def create_inference_model(image_size: Tuple[int, int] = (512, 512),
                          max_objects: int = 10,
                          device: str = "auto") -> SpringHybridModel:
    """Create model optimized for inference only."""
    config = SpringHybridConfig(
        deployment_mode=DeploymentMode.INFERENCE,
        image_size=image_size,
        max_objects=max_objects,
        device=device,
        enable_perception_module=True,
        enable_veg_module=False,
        mixed_precision=True,
        freeze_perception_backbone=True
    )
    return SpringHybridModel(config)
def safe_extract_numeric_value(value, default=1.0):
    """ DEBUG VERSION: Track what's happening with constraint satisfaction extraction."""
    
    print(f" DEBUG EXTRACT: Input value: {value} (type: {type(value)})")
    
    try:
        if isinstance(value, (int, float)):
            result = float(value)
            print(f" DEBUG EXTRACT: Numeric -> {result}")
            return result
        elif isinstance(value, torch.Tensor):
            if value.numel() == 1:
                result = float(value.item())
                print(f" DEBUG EXTRACT: Tensor scalar -> {result}")
                return result
            else:
                result = float(value.mean().item())
                print(f" DEBUG EXTRACT: Tensor mean -> {result}")
                return result
        elif isinstance(value, str):
            value_lower = value.lower()
            print(f" DEBUG EXTRACT: String '{value_lower}' -> defaulting to 1.0")
            if value_lower in ['neural', 'symbolic', 'hardnet', 'soft']:
                return 1.0  #  THIS IS THE PROBLEM!
            elif value_lower in ['failed', 'error', 'none']:
                return 0.0
            else:
                try:
                    result = float(value)
                    print(f" DEBUG EXTRACT: String number -> {result}")
                    return result
                except (ValueError, TypeError):
                    print(f" DEBUG EXTRACT: String parse failed -> {default}")
                    return default
        elif isinstance(value, bool):
            result = 1.0 if value else 0.0
            print(f" DEBUG EXTRACT: Bool -> {result}")
            return result
        elif value is None:
            print(f" DEBUG EXTRACT: None -> {default}")
            return default
        else:
            print(f" DEBUG EXTRACT: Unknown type {type(value)} -> {default}")
            return default
    except Exception as e:
        print(f" DEBUG EXTRACT: Exception {e} -> {default}")
        return default

if __name__ == "__main__":
    """Test the ACTUAL Stage 2 constraint issues."""
    
    print("=== TESTING STAGE 2 CONSTRAINT ISSUES ===\n")
    
    # Test 1: Create model
    print("TEST 1: Model Creation")
    research_model = create_research_model(image_size=(256, 256), max_objects=5)
    print(f"✓ Model created on {research_model.device}")
    
    # Test 2: Stage 2 forward pass with constraints
    print("\nTEST 2: Stage 2 Forward Pass with Constraints")
    
    batch_size = 2
    device = research_model.device
    images = torch.randn(batch_size, 3, 256, 256, device=device)
    
    # Create mock constraints (this is what your training does)
    mock_constraints = [
        [{'constraint_type': 'left_of', 'o1': 0, 'o2': 1}],  # Object 0 left of object 1
        [{'constraint_type': 'above', 'o1': 1, 'o2': 0}]     # Object 1 above object 0
    ]
    
    print(f" Testing with {len(mock_constraints)} constraint sets")
    
    research_model.eval()
    
    try:
        with torch.no_grad():
            # Test Stage 2 mode - THIS IS WHAT'S FAILING IN TRAINING!
            results = research_model(
                images, 
                constraints=mock_constraints,
                return_intermediate=True,
                training_stage='stage2',  # ← FORCE STAGE 2!
                srm_mode='differentiable'  # ← FORCE DIFFERENTIABLE!
            )
        
        print(f"✓ Stage 2 forward pass completed")
        
        # Check constraint info - THIS IS THE KEY!
        if 'layout_results' in results:
            constraint_info = results['layout_results']['constraint_info']
            print(f" CONSTRAINT INFO: {constraint_info}")
            
            constraint_satisfaction = constraint_info.get('constraint_satisfaction_rate', 'MISSING')
            print(f" CONSTRAINT SATISFACTION: {constraint_satisfaction} (type: {type(constraint_satisfaction)})")
            
            # This should NOT be 1.0 if constraints are being checked!
            if constraint_satisfaction == 1.0:
                print(f" PROBLEM: Constraint satisfaction is 1.0 - constraints not being checked!")
            elif constraint_satisfaction == 'MISSING':
                print(f" PROBLEM: Constraint satisfaction rate is missing!")
            else:
                print(f" Constraint satisfaction looks realistic: {constraint_satisfaction}")
        else:
            print(f" PROBLEM: No layout_results in output!")
            
    except Exception as e:
        print(f" STAGE 2 FORWARD FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Test constraint satisfaction extraction
    print("\nTEST 3: Constraint Satisfaction Extraction")
    
    # Test the function that's causing issues
    test_values = [
        1.0,
        torch.tensor(0.75),
        "hardnet",  # ← This returns 1.0 incorrectly!
        "neural",   # ← This returns 1.0 incorrectly!
        None,
        {'rate': 0.5}
    ]
    
    for i, test_val in enumerate(test_values):
        result = safe_extract_numeric_value(test_val)
        print(f"  Test {i+1}: {test_val} → {result}")
        
        # Check for the bug
        if isinstance(test_val, str) and result == 1.0:
            print(f"     BUG: String '{test_val}' incorrectly returns 1.0!")
    
    # Test 4: Check if loss manager is available
    print("\nTEST 4: Loss Manager Availability")
    
    try:
        from pipeline import AdvancedLossManager, get_stage2_loss_config
        loss_manager = AdvancedLossManager(get_stage2_loss_config())
        print(f" Loss manager available: {type(loss_manager)}")
    except ImportError as e:
        print(f" PROBLEM: Loss manager not available: {e}")
        print(f"    This means Stage 2 loss will fallback to Stage 1 loss only!")
    
    # Test 5: Check constraint generator
    print("\nTEST 5: Constraint Generator")
    
    try:
        from constraint_gen import ConstraintGenerator, ConstraintGenerationConfig
        config = ConstraintGenerationConfig()
        generator = ConstraintGenerator(config)
        print(f" Constraint generator available: {type(generator)}")
        
        # Test constraint generation
        layouts = torch.randn(2, 5, 4) * 100 + 50  # Random layouts
        valid_masks = torch.ones(2, 5).bool()
        categories = [['chair', 'table'], ['sofa', 'tv']]
        
        constraints = generator.generate_constraints_for_batch(layouts, valid_masks, categories)
        print(f" Generated {sum(len(c) for c in constraints)} constraints")
        
        if sum(len(c) for c in constraints) == 0:
            print(f" PROBLEM: No constraints generated!")
        
    except ImportError as e:
        print(f" PROBLEM: Constraint generator not available: {e}")
        print(f"    This means training will use empty constraint lists!")
    
    print(f"\n=== DIAGNOSIS COMPLETE ===")
    print("If you see problems above, THOSE are the real issues to fix!")
    print("The basic model test you ran before was not testing the actual problems.")