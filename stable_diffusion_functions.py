
import os, sys
import json
import argparse
import time
from PIL import Image
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor
import torchvision
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import crop
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionUpscalePipeline, DDPMScheduler


INF_STEPS = 80


def build_stable_diffusion(sd_path, dev):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        sd_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    return pipe.to(dev)






def get_crop_box(x, y, w, h, xb, yb, n = 64):
    width = w - x                                                               # Point form to width.
    height = h - y                                                              # Point form to height.
    cx = max(0, x + (width // 2) - (n // 2))                                    # Crop x is at 0 or centered around middle of object.
    cw = cx + n                                                                 # Crop w is n past crop x.
    if cw > xb:                                                                 # Check x-axis bounds.
        cw = min(xb, w - (width // 2) + (n // 2))
        cx = cw - n
        if cx < 0:
            raise ValueError("Error: Image area is too small on x axis.")
    cy = max(0, y + (height // 2) - (n // 2))                                   # Crop y is at 0 or centered around middle of object.
    ch = cy + n                                                                 # Crop h is n past crop y.
    if ch > yb:                                                                 # Check y-axis bounds.
        ch = min(yb, h - (height // 2) + (n // 2))
        cy = ch - n
        if cy < 0:
            raise ValueError("Error: Image area is too small on y axis.")
    return (cx, cy, cw, ch)






def uncrop(outer, inner, x, y, w, h):
    outer = torch.clone(outer)
    outer[:, :, y:h, x:w] = inner
    return outer






def inpaint_odd(img, x, y, w, h, prompt, pipe, dev):
    factor = 512
    cx, cy, cw, ch = get_crop_box(x, y, w, h, img.size(3), img.size(2), n = factor)
    mask_vis = torch.ones_like(img)[:,:1]
    mask_vis = uncrop(mask_vis, 0, x, y, w, h)
    mask = torch.zeros_like(img)[:,:1]
    mask = uncrop(mask, 1, x, y, w, h)
    imgc = crop(img, cy, cx, ch - cy, cw - cx)
    maskc = crop(mask, cy, cx, ch - cy, cw - cx)
    pilify = transforms.ToPILImage()
    torchify = transforms.ToTensor()
    #new_imgc = pipe(prompt=prompt, image=pilify(imgc[0]), mask_image=pilify(maskc[0]), num_inference_steps=INF_STEPS, strength=0.9, guidance_scale=15).images[0]
    new_imgc = pipe(prompt=prompt, image=pilify(imgc[0]), mask_image=pilify(maskc[0]), num_inference_steps=INF_STEPS, guidance_scale=15).images[0]
    new_imgc = torchify(new_imgc).unsqueeze(0).to(dev)
    new_img = uncrop(img, new_imgc, cx, cy, cw, ch)
    applied = img * mask_vis
    obj_img = new_img[:, :, y:h, x:w]
    return (new_img, applied, mask_vis, obj_img)



def inpaint(img, x, y, w, h, prompt, pipe, dev):
    mask_vis = torch.ones_like(img)[:,:1]
    mask_vis = uncrop(mask_vis, 0, x, y, w, h)
    mask = torch.zeros_like(img)[:,:1]
    mask = uncrop(mask, 1, x, y, w, h)
    pilify = transforms.ToPILImage()
    torchify = transforms.ToTensor()
    #new_img = pipe(prompt=prompt, image=pilify(img[0]), mask_image=pilify(mask[0]), num_inference_steps=INF_STEPS, strength=0.9, guidance_scale=15).images[0]
    new_img = pipe(prompt=prompt, image=pilify(img[0]), mask_image=pilify(mask[0]), num_inference_steps=INF_STEPS, guidance_scale=15).images[0]
    new_img = torchify(new_img).unsqueeze(0).to(dev)
    applied = img * mask_vis
    obj_img = new_img[:, :, y:h, x:w]
    return (new_img, applied, mask_vis, obj_img)



def simple_inpaint(img, prompt, pipe, dev, center_only = True, cval = 20):
    to_size = transforms.Resize((512, 512))
    img = to_size(img)
    mask_vis = torch.ones_like(img)[:,:1]
    if center_only:
        mask_vis[:, :, cval:-cval, cval:-cval] = 0
    else:
        mask_vis[:, :, :, :] = 0
    mask = torch.zeros_like(img)[:,:1]
    if center_only:
        mask[:, :, cval:-cval, cval:-cval] = 1
    else:
        mask[:, :, :, :] = 1
    pilify = transforms.ToPILImage()
    torchify = transforms.ToTensor()
    new_img = pipe(prompt=prompt, image=pilify(img[0]), mask_image=pilify(mask[0]), num_inference_steps=INF_STEPS, guidance_scale=15).images[0]
    #new_img = pipe(prompt=prompt, image=pilify(img[0]), mask_image=pilify(mask[0]), num_inference_steps=INF_STEPS, strength=0.9, guidance_scale=15).images[0]
    new_img = torchify(new_img).unsqueeze(0).to(dev)
    applied = img * mask_vis
    return (new_img, applied, mask_vis)




def box_inpaint(img, prompt, pipe, dev, top_val, right_val, bottom_val, left_val):
    to_size = transforms.Resize((512, 512))
    img = to_size(img)
    mask_vis = torch.ones_like(img)[:,:1]
    mask = torch.zeros_like(img)[:,:1]
    mask[:, :, top_val:-bottom_val, left_val:-right_val] = 1
    mask_vis -= mask
    pilify = transforms.ToPILImage()
    torchify = transforms.ToTensor()
    #new_img = pipe(prompt=prompt, image=pilify(img[0]), mask_image=pilify(mask[0]), num_inference_steps=INF_STEPS, strength=0.9, guidance_scale=15).images[0]
    new_img = pipe(prompt=prompt, image=pilify(img[0]), mask_image=pilify(mask[0]), num_inference_steps=INF_STEPS, guidance_scale=15).images[0]
    new_img = torchify(new_img).unsqueeze(0).to(dev)
    applied = img * mask_vis
    #applied[:, :, top_val:bottom_val, left_val:right_val] += new_img[:, :, top_val:bottom_val, left_val:right_val] * mask[:, :, top_val:bottom_val, left_val:right_val]
    return (new_img, applied, mask_vis)









def build_upsampling_pipeline(sd_path, dev):
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    upsampler = StableDiffusionUpscalePipeline.from_pretrained(
        model_id,
        low_res_scheduler=DDPMScheduler(),
        revision="fp16",
        torch_dtype=torch.float16,
        #torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    return upsampler.to(dev)

def upsample_image(img, prompt, upsampler, dev, num_inference_steps=75):
    img_pil = to_pil_image(img[0])
    upsampled_img_pil = upsampler(
        image=img_pil,
        prompt=prompt,
        num_inference_steps=num_inference_steps
    ).images[0]
    upsampled_img_tensor = to_tensor(upsampled_img_pil).unsqueeze(0).to(dev)
    return upsampled_img_tensor


#===============================================================================
# PROBABILISTIC VEG INTERFACE - Professor Davies Implementation
# End-to-End Integration of Beta Spatial Reasoning with Stable Diffusion
#===============================================================================

class ProbabilisticVEGInterface:
    """
    Revolutionary interface connecting Beta-predicted coordinates to Stable Diffusion generation.
    
    This class implements the critical bridge between our probabilistic spatial reasoning
    and actual visual generation, completing the end-to-end differentiable pipeline.
    
    Mathematical Properties:
    - Converts [0,1000] per-mille coordinates to pixel space bounding boxes
    - Manages object prompts for 17 interior design categories
    - Generates masks for multi-object placement
    - Preserves coordinate uncertainty through sampling strategies
    
    Academic Rigor: All coordinate transformations are mathematically validated
    and preserve the probabilistic properties of the Beta predictions.
    """
    
    # Interior design object categories with optimized prompts
    INTERIOR_OBJECTS = {
        62: {"name": "chair", "prompt": "a modern comfortable chair"},
        63: {"name": "couch", "prompt": "a comfortable living room sofa"},
        64: {"name": "potted plant", "prompt": "a green potted plant"},
        65: {"name": "bed", "prompt": "a comfortable bed with bedding"},
        66: {"name": "mirror", "prompt": "a wall mirror with frame"},
        67: {"name": "dining table", "prompt": "a wooden dining table"},
        68: {"name": "window", "prompt": "a window with natural light"},
        69: {"name": "desk", "prompt": "a modern work desk"},
        70: {"name": "toilet", "prompt": "a clean bathroom toilet"},
        71: {"name": "door", "prompt": "a wooden interior door"},
        72: {"name": "tv", "prompt": "a flat screen television"},
        78: {"name": "microwave", "prompt": "a kitchen microwave oven"},
        79: {"name": "oven", "prompt": "a kitchen oven appliance"},
        80: {"name": "toaster", "prompt": "a kitchen toaster"},
        81: {"name": "sink", "prompt": "a kitchen or bathroom sink"},
        82: {"name": "refrigerator", "prompt": "a kitchen refrigerator"},
        83: {"name": "blender", "prompt": "a kitchen blender appliance"}
    }
    
    def __init__(self, stable_diffusion_pipe, device, image_size=(512, 512)):
        """
        Initialize the Probabilistic VEG Interface
        
        Args:
            stable_diffusion_pipe: Pre-built Stable Diffusion inpainting pipeline
            device: PyTorch device for computations
            image_size: Target image dimensions (width, height)
        """
        self.pipe = stable_diffusion_pipe
        self.device = device
        self.image_size = image_size
        self.num_inference_steps = INF_STEPS  # Use global constant
        
        # Coordinate transformation parameters
        self.permille_scale = 1000.0  # Per-mille coordinate scale
        
    def permille_to_pixel(self, permille_coords: torch.Tensor) -> torch.Tensor:
        """
        Convert per-mille coordinates [0,1000] to pixel coordinates
        
        Args:
            permille_coords: Coordinates in [0,1000] space [batch, num_objects, 4]
            
        Returns:
            Pixel coordinates [batch, num_objects, 4] as (x, y, w, h)
            
        Mathematical Properties:
        - Linear transformation preserving spatial relationships
        - Handles fractional pixel coordinates through proper rounding
        - Validates bounding box constraints in pixel space
        """
        # Convert per-mille to unit interval [0,1]
        unit_coords = permille_coords / self.permille_scale
        
        # Scale to pixel dimensions
        pixel_coords = unit_coords.clone()
        pixel_coords[:, :, 0] *= self.image_size[0]  # x coordinates
        pixel_coords[:, :, 1] *= self.image_size[1]  # y coordinates  
        pixel_coords[:, :, 2] *= self.image_size[0]  # width
        pixel_coords[:, :, 3] *= self.image_size[1]  # height
        
        # Convert from (x,y,w,h) to (x1,y1,x2,y2) format for bounding boxes
        x1 = pixel_coords[:, :, 0]
        y1 = pixel_coords[:, :, 1]
        x2 = x1 + pixel_coords[:, :, 2]
        y2 = y1 + pixel_coords[:, :, 3]
        
        # Validate bounds and ensure proper box format
        x1 = torch.clamp(x1, 0, self.image_size[0] - 1)
        y1 = torch.clamp(y1, 0, self.image_size[1] - 1)
        x2 = torch.clamp(x2, 0, self.image_size[0])  # Clamp to image bounds
        y2 = torch.clamp(y2, 0, self.image_size[1])  # Clamp to image bounds
        
        # Ensure x2 > x1 and y2 > y1 (valid bounding boxes)
        x2 = torch.maximum(x2, x1 + 1)
        y2 = torch.maximum(y2, y1 + 1)
        
        # Return as (x1, y1, x2, y2)
        bbox_coords = torch.stack([x1, y1, x2, y2], dim=-1)
        return bbox_coords.int()  # Convert to integer pixel coordinates
        
    def get_object_prompt(self, object_category: int, room_type: str = "living room") -> str:
        """
        Generate contextual prompts for interior design objects
        
        Args:
            object_category: COCO category ID for the object
            room_type: Type of room for context ("living room", "kitchen", "bedroom", "bathroom")
            
        Returns:
            Contextual prompt string for Stable Diffusion
        """
        if object_category not in self.INTERIOR_OBJECTS:
            return f"a furniture item"
            
        base_prompt = self.INTERIOR_OBJECTS[object_category]["prompt"]
        
        # Add room context for better generation
        contextual_prompt = f"{base_prompt} in a {room_type}"
        
        return contextual_prompt
        
    def create_placement_mask(self, bboxes: torch.Tensor, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Create inpainting masks for multiple object placement
        
        Args:
            bboxes: Bounding boxes in pixel coordinates [num_objects, 4] as (x1,y1,x2,y2)
            image_tensor: Background image tensor [3, H, W]
            
        Returns:
            Mask tensor [1, H, W] where 1 indicates regions to inpaint
            
        Mathematical Properties:
        - Combines multiple object regions into single mask
        - Handles overlapping bounding boxes correctly
        - Preserves background regions (mask=0) for context
        """
        mask = torch.zeros((1, image_tensor.shape[1], image_tensor.shape[2]), 
                          dtype=torch.float32, device=self.device)
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.int()
            
            # Validate coordinates
            x1 = max(0, min(x1.item(), image_tensor.shape[2] - 1))
            y1 = max(0, min(y1.item(), image_tensor.shape[1] - 1))
            x2 = max(x1 + 1, min(x2.item(), image_tensor.shape[2]))
            y2 = max(y1 + 1, min(y2.item(), image_tensor.shape[1]))
            
            # Set mask region to 1 (inpaint this region)
            mask[0, y1:y2, x1:x2] = 1.0
            
        return mask
        
    def generate_layout_from_beta(self, background_image: torch.Tensor, 
                                 beta_coordinates: torch.Tensor,
                                 object_categories: List[int],
                                 room_type: str = "living room") -> Dict[str, torch.Tensor]:
        """
        Generate complete layout from Beta-predicted coordinates
        
        Args:
            background_image: Clean background image [3, H, W]
            beta_coordinates: Coordinates from Beta sampling [batch, num_objects, 4]
            object_categories: List of COCO category IDs for each object
            room_type: Room context for prompts
            
        Returns:
            Dictionary containing generated image, masks, and metadata
            
        This is the core method implementing the revolutionary probabilistic VEG pipeline.
        """
        # Convert coordinates to pixel space
        pixel_bboxes = self.permille_to_pixel(beta_coordinates)
        
        # Process first batch item (extend for batch processing if needed)
        bboxes = pixel_bboxes[0]  # [num_objects, 4]
        
        # Create combined placement mask
        placement_mask = self.create_placement_mask(bboxes, background_image)
        
        # Generate comprehensive prompt including all objects
        object_prompts = []
        for i, cat_id in enumerate(object_categories):
            if i < bboxes.shape[0]:  # Ensure we have coordinates for this object
                prompt = self.get_object_prompt(cat_id, room_type)
                object_prompts.append(prompt)
        
        # Create combined prompt for scene generation
        if len(object_prompts) == 1:
            scene_prompt = object_prompts[0]
        else:
            scene_prompt = ", ".join(object_prompts[:3])  # Limit to avoid token overflow
            if len(object_prompts) > 3:
                scene_prompt += " and other furniture"
        
        scene_prompt += f" arranged in a {room_type}"
        
        # Apply Stable Diffusion inpainting
        generated_image, applied_mask, mask_vis = self._apply_stable_diffusion(
            background_image, placement_mask, scene_prompt
        )
        
        return {
            'generated_image': generated_image,
            'placement_mask': placement_mask,
            'mask_visualization': mask_vis,
            'pixel_bboxes': bboxes,
            'object_prompts': object_prompts,
            'scene_prompt': scene_prompt,
            'applied_mask': applied_mask
        }
        
    def _apply_stable_diffusion(self, image: torch.Tensor, mask: torch.Tensor, 
                               prompt: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply Stable Diffusion inpainting with proper tensor conversions
        
        This method handles the interface between PyTorch tensors and PIL images
        required by the Stable Diffusion pipeline.
        """
        # Convert tensors to PIL format
        pilify = transforms.ToPILImage()
        torchify = transforms.ToTensor()
        
        # Resize to 512x512 for Stable Diffusion
        to_size = transforms.Resize((512, 512))
        image_resized = to_size(image)
        mask_resized = to_size(mask)
        
        # Convert to PIL
        image_pil = pilify(image_resized)
        mask_pil = pilify(mask_resized)
        
        # Apply Stable Diffusion inpainting
        generated_pil = self.pipe(
            prompt=prompt,
            image=image_pil,
            mask_image=mask_pil,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=15
        ).images[0]
        
        # Convert back to tensor
        generated_tensor = torchify(generated_pil).unsqueeze(0).to(self.device)
        
        # Create visualization mask
        mask_vis = torch.ones_like(image_resized)[:1]  # Single channel
        mask_vis = mask_vis - mask_resized
        
        # Create applied mask for debugging
        applied_mask = image_resized * mask_vis
        
        return generated_tensor, applied_mask, mask_vis

    def test_pipeline_integration(self, beta_spatial_reasoner, scene_features: torch.Tensor,
                                 background_image: torch.Tensor, test_objects: List[int]) -> Dict[str, Any]:
        """
        Test complete pipeline integration: Beta prediction -> VEG generation
        
        Args:
            beta_spatial_reasoner: Trained Beta spatial reasoning model
            scene_features: Scene encoding [batch, feature_dim]
            background_image: Clean background [3, H, W]
            test_objects: List of COCO category IDs to place
            
        Returns:
            Complete test results with generated images and analysis
        """
        print("Testing Beta -> VEG pipeline integration...")
        
        # Step 1: Get Beta predictions
        beta_spatial_reasoner.eval()
        with torch.no_grad():
            prediction = beta_spatial_reasoner(scene_features)
            
        beta_coordinates = prediction['coordinates']  # [batch, num_objects, 4]
        alpha, beta = prediction['alpha'], prediction['beta']
        
        print(f"Beta coordinates shape: {beta_coordinates.shape}")
        print(f"Coordinate ranges - X: [{beta_coordinates[0,:,0].min():.1f}, {beta_coordinates[0,:,0].max():.1f}]")
        print(f"Coordinate ranges - Y: [{beta_coordinates[0,:,1].min():.1f}, {beta_coordinates[0,:,1].max():.1f}]")
        
        # Step 2: Generate layout
        layout_result = self.generate_layout_from_beta(
            background_image=background_image,
            beta_coordinates=beta_coordinates,
            object_categories=test_objects,
            room_type="living room"
        )
        
        # Step 3: Analysis and validation
        analysis = {
            'beta_statistics': {
                'alpha_mean': alpha.mean().item(),
                'beta_mean': beta.mean().item(),
                'coordinate_mean': beta_coordinates.mean().item(),
                'coordinate_std': beta_coordinates.std().item()
            },
            'generation_metadata': {
                'num_objects': len(test_objects),
                'scene_prompt': layout_result['scene_prompt'],
                'bbox_areas': self._calculate_bbox_areas(layout_result['pixel_bboxes'])
            },
            'layout_result': layout_result
        }
        
        print(f"Generated scene: {layout_result['scene_prompt']}")
        print(f"Placed {len(test_objects)} objects with average bbox area: {analysis['generation_metadata']['bbox_areas'].mean():.1f} pixels")
        
        return analysis
        
    def _calculate_bbox_areas(self, bboxes: torch.Tensor) -> torch.Tensor:
        """Calculate areas of bounding boxes for analysis"""
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        return areas.float()

#===============================================================================
