"""
Generate 10,000 background images using Stable Diffusion for SPRING evaluation.
5,000 kitchen backgrounds and 5,000 living room backgrounds.
NO ERROR HANDLING - Let it crash to identify issues.
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
from pathlib import Path
import time
import json

# Configuration - EXACTLY as specified in SPRING paper
NUM_KITCHEN = 5000
NUM_LIVING = 5000
NUM_INFERENCE_STEPS = 80
GUIDANCE_SCALE = 15.0
IMAGE_SIZE = 512  # Standard SD v1.5 resolution

# Prompts - EXACTLY as specified
KITCHEN_PROMPT = "A clean, empty, kitchen."
LIVING_PROMPT = "A clean, empty, living room."

def generate_backgrounds():
    # Create output directory structure
    output_dir = Path("data/evaluation/backgrounds")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing images to resume from
    existing_kitchen = list(output_dir.glob("kitchen_*.png"))
    existing_living = list(output_dir.glob("living_*.png"))
    
    kitchen_start_idx = len(existing_kitchen)
    living_start_idx = len(existing_living)
    
    if kitchen_start_idx > 0 or living_start_idx > 0:
        print(f"RESUME MODE: Found {kitchen_start_idx} kitchen and {living_start_idx} living room images")
        print(f"Continuing from kitchen_{kitchen_start_idx:04d}.png and living_{living_start_idx:04d}.png")
    
    # Metadata file to track generation parameters
    metadata = {
        "num_kitchen": NUM_KITCHEN,
        "num_living": NUM_LIVING,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "image_size": IMAGE_SIZE,
        "kitchen_prompt": KITCHEN_PROMPT,
        "living_prompt": LIVING_PROMPT,
        "model_id": "runwayml/stable-diffusion-v1-5",
        "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "resume_kitchen_from": kitchen_start_idx,
        "resume_living_from": living_start_idx
    }
    
    # Save metadata
    with open(output_dir / "generation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Starting background generation with parameters:")
    print(f"  Model: runwayml/stable-diffusion-v1-5")
    print(f"  Inference steps: {NUM_INFERENCE_STEPS}")
    print(f"  Guidance scale: {GUIDANCE_SCALE}")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Output directory: {output_dir}")
    
    # Initialize Stable Diffusion pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the model - will crash if model can't be loaded
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)
    
    # Disable safety checker for faster generation (we're generating empty rooms)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    
    print("Model loaded successfully")
    
    # Track timing
    total_start = time.time()
    
    # Generate kitchen backgrounds
    remaining_kitchen = NUM_KITCHEN - kitchen_start_idx
    if remaining_kitchen > 0:
        print(f"\n=== Generating {remaining_kitchen} remaining kitchen backgrounds ({kitchen_start_idx}-{NUM_KITCHEN-1}) ===")
    else:
        print(f"\n=== Kitchen backgrounds already complete ({NUM_KITCHEN} images) ===")
    
    for i in range(kitchen_start_idx, NUM_KITCHEN):
        start = time.time()
        
        # Use sequential seeds for reproducibility
        generator = torch.Generator(device=device).manual_seed(i)
        
        # Generate image - will crash on any generation error
        image = pipe(
            prompt=KITCHEN_PROMPT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            generator=generator
        ).images[0]
        
        # Save image
        filename = output_dir / f"kitchen_{i:04d}.png"
        image.save(filename)
        
        elapsed = time.time() - start
        
        # Progress logging every 100 images
        if (i + 1) % 100 == 0:
            completed_so_far = i + 1 - kitchen_start_idx
            total_time_so_far = time.time() - total_start
            if completed_so_far > 0:
                avg_time = total_time_so_far / completed_so_far
                eta = avg_time * (NUM_KITCHEN - i - 1)
                print(f"  Kitchen {i+1}/{NUM_KITCHEN} - Last: {elapsed:.2f}s - Avg: {avg_time:.2f}s - ETA: {eta/60:.1f}min")
        
        # Clear CUDA cache periodically to prevent memory issues
        if device == "cuda" and (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
    
    if remaining_kitchen > 0:
        print(f"Kitchen backgrounds complete. Time: {(time.time() - total_start)/60:.2f} minutes")
    
    # Generate living room backgrounds
    remaining_living = NUM_LIVING - living_start_idx
    if remaining_living > 0:
        print(f"\n=== Generating {remaining_living} remaining living room backgrounds ({living_start_idx}-{NUM_LIVING-1}) ===")
        living_start = time.time()
    else:
        print(f"\n=== Living room backgrounds already complete ({NUM_LIVING} images) ===")
        living_start = time.time()
    
    for i in range(living_start_idx, NUM_LIVING):
        start = time.time()
        
        # Use sequential seeds (offset by NUM_KITCHEN for uniqueness)
        generator = torch.Generator(device=device).manual_seed(NUM_KITCHEN + i)
        
        # Generate image
        image = pipe(
            prompt=LIVING_PROMPT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            generator=generator
        ).images[0]
        
        # Save image
        filename = output_dir / f"living_{i:04d}.png"
        image.save(filename)
        
        elapsed = time.time() - start
        
        # Progress logging
        if (i + 1) % 100 == 0:
            completed_so_far = i + 1 - living_start_idx
            total_time_so_far = time.time() - living_start
            if completed_so_far > 0:
                avg_time = total_time_so_far / completed_so_far
                eta = avg_time * (NUM_LIVING - i - 1)
                print(f"  Living {i+1}/{NUM_LIVING} - Last: {elapsed:.2f}s - Avg: {avg_time:.2f}s - ETA: {eta/60:.1f}min")
        
        # Clear CUDA cache periodically
        if device == "cuda" and (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
    
    total_time = time.time() - total_start
    print(f"\n=== GENERATION COMPLETE ===")
    print(f"Total images generated: {NUM_KITCHEN + NUM_LIVING}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Average time per image: {total_time/(NUM_KITCHEN + NUM_LIVING):.2f} seconds")
    print(f"Output directory: {output_dir}")
    
    # Final metadata update with completion info
    metadata["completion_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metadata["total_generation_time_minutes"] = total_time / 60
    metadata["average_time_per_image_seconds"] = total_time / (NUM_KITCHEN + NUM_LIVING)
    
    with open(output_dir / "generation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    # No try-except - let it crash if there are issues
    generate_backgrounds()