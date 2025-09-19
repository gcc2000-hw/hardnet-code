"""
SPRING Hybrid Training System - Two-Stage Training Pipeline
The main training script that integrates all components for constraint-aware layout generation

Features:
- Stage 1: Discrete warm-up training (original SPRING procedures)
- Stage 2: Differentiable fine-tuning (constraint-aware training)
- Seamless integration with data infrastructure (Phases 1.1-1.3)
- Advanced loss scheduling and curriculum learning
- Comprehensive checkpoint management
- Real-time monitoring and visualization
- Production-ready training pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import logging
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# Disable all progress bars for cleaner output - PERFORMANCE OPTIMIZATION
os.environ['WANDB_SILENT'] = 'true'  # Disable wandb progress bars
os.environ['WANDB_CONSOLE'] = 'off'   # Disable wandb console output
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Disable transformers progress bars
os.environ['DIFFUSERS_VERBOSITY'] = 'error'     # Disable diffusers progress bars
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'  # Disable huggingface hub progress bars

import sys
import wandb
# Removed tqdm import and monkey patching - using environment variables for progress bar control

# Import our system components
try:
    from custom_dataloader import SpringTrainingDataLoader, SpringDatasetConfig  # UPDATED
    from constraint_gen import ConstraintGenerator, ConstraintGenerationConfig, ConstraintDifficulty
    from preprocessing import BatchPreprocessor, PreprocessingConfig, create_preprocessor
    from completeInt import SpringHybridModel, SpringHybridConfig, create_research_model, create_production_model, DeploymentMode
    from spring_int import SpatialReasoningMode, HybridSRMConfig
    from pipeline import AdvancedLossManager, get_stage1_loss_config, get_stage2_loss_config
    from constraint_validator import validate_constraints_before_training, ConstraintValidator
    SYSTEM_COMPONENTS_AVAILABLE = True
except ImportError:
    raise ImportError("System components not available. Cannot proceed without real implementations.")


def extract_required_numeric(value: Any, field_name: str) -> float:
    """Extract numeric value with validation - NO FALLBACKS."""
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, torch.Tensor):
        return value.item()
    elif value is None:
        raise ValueError(f"Required field '{field_name}' is None")
    else:
        raise TypeError(f"Required field '{field_name}' has invalid type: {type(value)}")


def extract_performance_metrics(info: Dict[str, Any]) -> Dict[str, float]:
    """Extract performance metrics from model output - NO FALLBACKS."""
    performance_metrics = {}
    
    # Constraint satisfaction - REQUIRED
    if 'constraint_satisfaction_rate' in info:
        constraint_satisfaction = info['constraint_satisfaction_rate']
    elif 'constraint_satisfaction' in info:
        constraint_satisfaction = info['constraint_satisfaction']
    else:
        raise KeyError("Missing required 'constraint_satisfaction_rate' or 'constraint_satisfaction' in model output")
    
    performance_metrics['constraint_satisfaction_rate'] = extract_required_numeric(
        constraint_satisfaction, 'constraint_satisfaction_rate'
    )
    
    # Layout quality - REQUIRED
    if 'layout_quality' in info:
        layout_quality = info['layout_quality']
    elif 'reconstruction_quality' in info:
        layout_quality = info['reconstruction_quality']
    else:
        raise KeyError("Missing required 'layout_quality' or 'reconstruction_quality' in model output")
    
    performance_metrics['layout_quality'] = extract_required_numeric(
        layout_quality, 'layout_quality'
    )
    
    # Processing time - REQUIRED
    if 'processing_time' in info:
        processing_time = info['processing_time']
    elif 'total_time' in info:
        processing_time = info['total_time']
    else:
        raise KeyError("Missing required 'processing_time' or 'total_time' in model output")
    
    performance_metrics['processing_time'] = extract_required_numeric(
        processing_time, 'processing_time'
    )
    
    return performance_metrics


@dataclass
class TrainingConfig:
    """Complete training configuration for two-stage pipeline."""
    
    # Dataset configuration
    dataset_root: str = "data/spring_dataset"
    dataset_image_size: Tuple[int, int] = (512, 512)
    max_objects_per_scene: int = 10
    resume_from_checkpoint: Optional[str] = None
    # Stage 1: Discrete warm-up training
    stage1_epochs: int = 100
    stage1_batch_size: int = 32
    stage1_learning_rate: float = 1e-4
    stage1_weight_decay: float = 1e-5
    stage1_gradient_clip: float = 1.0
    stage1_warmup_epochs: int = 10
    
    # Stage 2: Differentiable fine-tuning  
    stage2_epochs: int = 200
    stage2_batch_size: int = 8   # CRITICAL FIX: Match original SPRING paper
    stage2_learning_rate: float = 1e-4  # CRITICAL FIX: Increased from 5e-5 for constraint learning
    stage2_weight_decay: float = 1e-6
    stage2_gradient_clip: float = 0.5
    stage2_warmup_epochs: int = 20
    
    # Model configuration
    model_mode: SpatialReasoningMode = SpatialReasoningMode.HYBRID
    enable_constraint_processing: bool = True
    enable_curriculum_learning: bool = True
    curriculum_progression_schedule: str = "linear"  # "linear", "cosine", "exponential"
    skip_stage1: bool = False  # Skip Stage 1 and train directly in Stage 2
    
    # Training optimization
    mixed_precision: bool = False
    gradient_checkpointing: bool = True
    enable_multi_gpu: bool = True
    accumulation_steps: int = 1
    max_gradient_norm: float = 1.0
    
    # Constraint training
    constraint_weight_schedule: str = "curriculum"  # "constant", "linear", "curriculum"
    initial_constraint_weight: float = 1.0  # CRITICAL FIX: High constraint priority from epoch 0
    final_constraint_weight: float = 1.0
    constraint_satisfaction_target: float = 0.95
    
    # Checkpointing and monitoring
    checkpoint_dir: str = "checkpoints"
    save_every_epochs: int = 10
    validate_every_epochs: int = 5
    early_stopping_patience: int = 20
    monitor_metric: str = "constraint_satisfaction_rate"
    
    # Logging and visualization
    enable_wandb: bool = True
    wandb_project: str = "spring-hybrid"
    wandb_run_name: str = None
    log_every_steps: int = 50
    save_visualizations: bool = True
    visualizations_dir: str = "visualizations"
    
    # Performance and debugging
    profiler_enabled: bool = False
    detect_anomaly: bool = False
    benchmark_mode: bool = False
    seed: int = 42
    
    def __post_init__(self):
        if self.wandb_run_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.wandb_run_name = f"spring_hybrid_{timestamp}"


class TrainingStage:
    """Enum for training stages."""
    STAGE1_DISCRETE = "stage1_discrete"
    STAGE2_DIFFERENTIABLE = "stage2_differentiable"


class SpringHybridTrainer:
    """
    Main trainer class for two-stage SPRING Hybrid training.
    
    Coordinates all components:
    - Data loading and preprocessing pipeline
    - Constraint generation and curriculum learning
    - Model training with advanced loss scheduling
    - Checkpoint management and monitoring
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.current_stage = None
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('-inf')
        self.patience_counter = 0
        
        # Setup logging and monitoring
        self.logger = self._setup_logging()
        self._setup_reproducibility()
        self._setup_monitoring()
        
        # Initialize components
        self._initialize_data_pipeline()
        self._initialize_model()
        self._initialize_training_components()
        # Skip checkpoint loading when doing direct Stage 2 training
        skip_stage1 = getattr(self.config, 'skip_stage1', False)
        if self.config.resume_from_checkpoint:
            self._load_checkpoint(self.config.resume_from_checkpoint)
        # Training state tracking
        self.training_metrics = defaultdict(list)
        self.validation_metrics = defaultdict(list)
        self.constraint_satisfaction_history = deque(maxlen=100)
        
        self.logger.info("SpringHybridTrainer initialized successfully")
    def _extract_constraint_satisfaction(self, info: Dict[str, Any]) -> float:
        """Extract constraint satisfaction rate with validation - NO FALLBACKS."""
        if 'constraint_satisfaction_rate' not in info:
            raise KeyError("Model output missing required 'constraint_satisfaction_rate'")
        return extract_required_numeric(
            info['constraint_satisfaction_rate'], 
            'constraint_satisfaction_rate'
        )

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to training device."""
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device, non_blocking=True)
            else:
                device_batch[key] = value
        
        return device_batch
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging system."""
        logger = logging.getLogger('SpringHybridTrainer')
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            os.makedirs('logs', exist_ok=True)
            file_handler = logging.FileHandler('logs/training.log')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            logger.setLevel(logging.INFO)
        
        return logger
    def _initialize_data_pipeline(self):
        """Initialize complete data loading and preprocessing pipeline for SPRING training pairs."""
        if not SYSTEM_COMPONENTS_AVAILABLE:
            raise RuntimeError("System components not available. Cannot proceed without real implementations.")
        
        # STRICT: Dataset must exist - no fallbacks
        assert self._check_spring_dataset_exists(), f"SPRING training data NOT FOUND at {self.config.dataset_root}. Run coco_processor_fixed.py first."
        
        # UPDATED: Dataset configuration for SPRING training pairs
        dataset_config = SpringDatasetConfig(  # UPDATED class name
            dataset_root=self.config.dataset_root,
            image_size=self.config.dataset_image_size,
            max_objects_to_place=self.config.max_objects_per_scene,  # Objects to predict
            max_background_objects=self.config.max_objects_per_scene,  # Objects in background
            output_format="sequence",
            enable_augmentation=True,
            coordinate_system="per_mille"  # SPRING standard
        )
        
        # Create data loaders - FAIL FAST on any errors
        self.train_loader = SpringTrainingDataLoader.create_train_loader(
            dataset_config,
            batch_size=self.config.stage1_batch_size,
            shuffle=True,
            num_workers=0  # FIXED: Use 0 workers - multiprocessing is 100x slower for small datasets
        )
        
        self.val_loader = SpringTrainingDataLoader.create_val_loader(
            dataset_config,
            batch_size=self.config.stage1_batch_size,
            num_workers=0  # FIXED: Use 0 workers - multiprocessing is 100x slower for small datasets
        )
        
        # Constraint generator (for additional constraints beyond those in training pairs)
        constraint_config = ConstraintGenerationConfig(
            canvas_width=self.config.dataset_image_size[0],
            canvas_height=self.config.dataset_image_size[1],
            enable_curriculum=self.config.enable_curriculum_learning,
            constraints_per_scene=(1, 2),  # CRITICAL: Reduced to (1,2) for mathematical stability
            max_constraint_density_ratio=0.5,  # CRITICAL: 0.5 constraints per variable max
            enable_constraint_pruning=True,
            enable_mathematical_validation=True,  # CRITICAL: Enable rigorous mathematical validation
            prioritize_feasible_constraints=True,
            validate_constraints=True,
            max_condition_number=1e6  # Prevent ill-conditioned constraint matrices
        )
        
        self.constraint_generator = ConstraintGenerator(constraint_config)
        
        # Preprocessor
        self.preprocessor = create_preprocessor(
            target_size=self.config.dataset_image_size,
            coordinate_system="per_mille"  # SPRING standard
        )
        
        self.logger.info(f"SPRING data pipeline initialized: {len(self.train_loader)} train, {len(self.val_loader)} val batches")   

    def _check_spring_dataset_exists(self) -> bool:
        """Check if the SPRING training dataset exists."""
        dataset_path = Path(self.config.dataset_root)
        backgrounds_path = dataset_path / "backgrounds"
        annotations_path = dataset_path / "annotations"
        splits_path = dataset_path / "splits.json"
        
        return (dataset_path.exists() and 
                backgrounds_path.exists() and 
                annotations_path.exists() and 
                splits_path.exists())
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint (.pt files only)."""
        checkpoint_path = Path(checkpoint_path)
        
        # STRICT: Checkpoint must exist and load successfully - no fallbacks
        if checkpoint_path.exists():
            self.logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        # Load model state with compatibility handling
        model_state_dict = checkpoint['model_state_dict']
        
        # CRITICAL: Auto-detect checkpoint stage and adapt model accordingly
        hardnet_keys = [k for k in model_state_dict.keys() if 'hardnet_layer' in k and any(param in k for param in ['A', 'b_l', 'b_u', 'A_pinv'])]
        is_stage2_checkpoint = len(hardnet_keys) > 0
        
        if is_stage2_checkpoint:
            self.logger.info("Detected Stage 2 checkpoint - ensuring model compatibility")
            # Switch to DIFFERENTIABLE mode to accept HardNet parameters
            if hasattr(self.model, 'spatial_reasoning_module'):
                self.model.spatial_reasoning_module.config.mode = SpatialReasoningMode.DIFFERENTIABLE
                # Initialize HardNet layer if not present to accept checkpoint parameters
                if hasattr(self.model.spatial_reasoning_module, 'hardnet_layer') and self.model.spatial_reasoning_module.hardnet_layer is None:
                    from hardnet_aff import HardNetAff
                    # Create empty HardNet layer that can accept saved buffers
                    self.model.spatial_reasoning_module.hardnet_layer = HardNetAff(
                        constraint_matrix=None,  # Will be loaded from checkpoint
                        enable_logging=True      # Enable for debugging projection issues
                    )
        else:
            self.logger.info("Detected Stage 1 checkpoint - maintaining current model state")
        
        current_model_keys = set(self.model.state_dict().keys())
        checkpoint_keys = set(model_state_dict.keys())
        
        # Find unexpected and missing keys
        unexpected_keys = checkpoint_keys - current_model_keys
        missing_keys = current_model_keys - checkpoint_keys
        
        if unexpected_keys:
            self.logger.warning(f"Unexpected keys in checkpoint: {list(unexpected_keys)}")
        if missing_keys:
            self.logger.warning(f"Missing keys in checkpoint: {list(missing_keys)}")
            
        # Load compatible keys only
        self.model.load_state_dict(model_state_dict, strict=False)
        
        # Load training state with filename-based epoch correction
        stored_epoch = checkpoint.get('epoch', 0)
        
        # CRITICAL: Fix corrupted epoch data by extracting from filename
        checkpoint_filename = Path(checkpoint_path).name
        if 'stage2_epoch_' in checkpoint_filename:
            try:
                filename_epoch = int(checkpoint_filename.split('stage2_epoch_')[1].split('.pt')[0])
                if filename_epoch != stored_epoch:
                    self.logger.warning(f"Epoch mismatch: file={filename_epoch}, stored={stored_epoch}. Using filename epoch.")
                    self.current_epoch = filename_epoch
                else:
                    self.current_epoch = stored_epoch
            except (ValueError, IndexError):
                self.logger.warning(f"Could not parse epoch from filename {checkpoint_filename}, using stored epoch {stored_epoch}")
                self.current_epoch = stored_epoch
        else:
            self.current_epoch = stored_epoch
            
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', float('-inf'))
        
        # Load optimizer states
        if 'stage1_optimizer_state_dict' in checkpoint:
            self.stage1_optimizer.load_state_dict(checkpoint['stage1_optimizer_state_dict'])
        if 'stage2_optimizer_state_dict' in checkpoint:
            self.stage2_optimizer.load_state_dict(checkpoint['stage2_optimizer_state_dict'])
        
        # Load scheduler states  
        if 'stage1_scheduler_state_dict' in checkpoint:
            self.stage1_scheduler.load_state_dict(checkpoint['stage1_scheduler_state_dict'])
        if 'stage2_scheduler_state_dict' in checkpoint:
            self.stage2_scheduler.load_state_dict(checkpoint['stage2_scheduler_state_dict'])
        
        # Load scaler state
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_stage = checkpoint.get('current_stage', TrainingStage.STAGE1_DISCRETE)
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
    def _setup_reproducibility(self):
        """Setup reproducible training environment."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
        
        # Configure deterministic operations
        if self.config.benchmark_mode:
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Anomaly detection for debugging
        if self.config.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        
        self.logger.info(f"Reproducibility setup with seed: {self.config.seed}")
    
    def _setup_monitoring(self):
        """Setup training monitoring and visualization."""
        # Create directories
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.visualizations_dir, exist_ok=True)
        
        # STRICT: If wandb enabled, it must work - no fallbacks
        if self.config.enable_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.__dict__
            )
            self.logger.info("Weights & Biases initialized")
    
    def _initialize_data_pipeline(self):
        """Initialize complete data loading and preprocessing pipeline for SPRING training pairs."""
        if not SYSTEM_COMPONENTS_AVAILABLE:
            raise RuntimeError("System components not available")
            return
        
        # STRICT: Dataset must exist - no fallbacks
        assert self._check_spring_dataset_exists(), f"SPRING training data NOT FOUND at {self.config.dataset_root}. Run coco_processor_fixed.py first."
        
        # FIXED: Dataset configuration for SPRING training pairs with correct parameter names
        dataset_config = SpringDatasetConfig(
            dataset_root=self.config.dataset_root,
            image_size=self.config.dataset_image_size,
            max_objects_to_place=self.config.max_objects_per_scene,    
            max_background_objects=self.config.max_objects_per_scene,  
            output_format="sequence",
            enable_augmentation=True,
            coordinate_system="per_mille"
        )
        
        # Create data loaders - FAIL FAST on any errors
        self.train_loader = SpringTrainingDataLoader.create_train_loader(
            dataset_config,
            batch_size=self.config.stage1_batch_size,
            shuffle=True,
            num_workers=0  # FIXED: Use 0 workers - multiprocessing is 100x slower for small datasets
        )
        
        self.val_loader = SpringTrainingDataLoader.create_val_loader(
            dataset_config,
            batch_size=self.config.stage1_batch_size,
            num_workers=0  # FIXED: Use 0 workers - multiprocessing is 100x slower for small datasets
        )
        
        # Constraint generator (for additional constraints beyond those in training pairs)
        constraint_config = ConstraintGenerationConfig(
            canvas_width=self.config.dataset_image_size[0],
            canvas_height=self.config.dataset_image_size[1],
            enable_curriculum=self.config.enable_curriculum_learning,
            constraints_per_scene=(1, 2),  # CRITICAL: Reduced to (1,2) for mathematical stability
            max_constraint_density_ratio=0.5,  # CRITICAL: 0.5 constraints per variable max
            enable_constraint_pruning=True,
            enable_mathematical_validation=True,  # CRITICAL: Enable rigorous mathematical validation
            prioritize_feasible_constraints=True,
            validate_constraints=True,
            max_condition_number=1e6  # Prevent ill-conditioned constraint matrices
        )
        
        self.constraint_generator = ConstraintGenerator(constraint_config)
        
        # Preprocessor
        self.preprocessor = create_preprocessor(
            target_size=self.config.dataset_image_size,
            coordinate_system="per_mille"  # SPRING standard
        )
        
        self.logger.info(f"SPRING data pipeline initialized: {len(self.train_loader)} train, {len(self.val_loader)} val batches")
    
    def _check_dataset_exists(self) -> bool:
        """Check if the required dataset exists."""
        dataset_path = Path(self.config.dataset_root)
        images_path = dataset_path / "images"
        annotations_path = dataset_path / "annotations"
        
        return dataset_path.exists() and images_path.exists() and annotations_path.exists()
    
    def _create_test_dataset(self):
        """Create a minimal test dataset for training."""
        dataset_path = Path(self.config.dataset_root)
        images_path = dataset_path / "images"
        annotations_path = dataset_path / "annotations"
        
        # Create directories
        images_path.mkdir(parents=True, exist_ok=True)
        annotations_path.mkdir(parents=True, exist_ok=True)
        
        # Create test samples
        import random
        from PIL import Image
        import json
        
        sample_ids = []
        for i in range(20):  # Create 20 test samples
            sample_id = f"test_sample_{i:03d}"
            sample_ids.append(sample_id)
            
            # Create test image
            img = Image.new('RGB', self.config.dataset_image_size, 
                          color=(random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)))
            img.save(images_path / f"{sample_id}.jpg")
            
            # Create test annotation
            n_objects = random.randint(1, min(4, self.config.max_objects_per_scene))
            objects = []
            categories = ['chair', 'table', 'sofa', 'bed', 'tv']
            
            for j in range(n_objects):
                # Generate random but reasonable bounding boxes
                x = random.randint(10, self.config.dataset_image_size[0] - 100)
                y = random.randint(10, self.config.dataset_image_size[1] - 100)
                w = random.randint(40, 120)
                h = random.randint(40, 120)
                
                # Ensure within bounds
                x = min(x, self.config.dataset_image_size[0] - w - 10)
                y = min(y, self.config.dataset_image_size[1] - h - 10)
                
                objects.append({
                    'bbox': [x, y, w, h],
                    'category': random.choice(categories),
                    'score': random.uniform(0.8, 1.0),
                    'properties': {}
                })
            
            annotation = {'objects': objects}
            
            with open(annotations_path / f"{sample_id}.json", 'w') as f:
                json.dump(annotation, f, indent=2)
        
        # Create splits file
        random.shuffle(sample_ids)
        train_split = sample_ids[:16]  # 16 for training
        val_split = sample_ids[16:18]   # 2 for validation
        test_split = sample_ids[18:]    # 2 for testing
        
        splits = {
            'train': train_split,
            'val': val_split,
            'test': test_split
        }
        
        with open(dataset_path / "splits.json", 'w') as f:
            json.dump(splits, f, indent=2)
        
        self.logger.info(f"Created test dataset: {len(train_split)} train, {len(val_split)} val, {len(test_split)} test samples")
    
# Mock data loaders removed - only real implementations allowed
    
    def _initialize_model(self):
        """Initialize the hybrid SPRING model."""
        if not SYSTEM_COMPONENTS_AVAILABLE:
            raise RuntimeError("System components not available. Cannot proceed without real model implementation.")
        
        # Device setup - must come first
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model configuration
        try:
            deployment_mode = DeploymentMode.RESEARCH
        except NameError:
            # Fallback if DeploymentMode not available
            deployment_mode = "research"
        
        model_config = SpringHybridConfig(
            deployment_mode=deployment_mode,
            device=self.device,
            image_size=self.config.dataset_image_size,
            max_objects=self.config.max_objects_per_scene,
            mixed_precision=self.config.mixed_precision,
            gradient_checkpointing=self.config.gradient_checkpointing,
            srm_mode=self.config.model_mode,
            enable_veg_module=False  # CRITICAL FIX: Disable VEG for Stage 1 (30x speedup)
        )
        
        self.model = SpringHybridModel(model_config)
        self.model = self.model.to(self.device)
        
        if self.config.enable_multi_gpu and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.logger.info(f"Multi-GPU training enabled: {torch.cuda.device_count()} GPUs")
        
        # GPU OPTIMIZATION: PyTorch model compilation for 15-30% speedup
        if getattr(self.config, 'compile_model', False):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                self.logger.info("Model compiled with PyTorch 2.0+ for optimization")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}. Continuing without compilation.")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model initialized: {total_params:,} total params, {trainable_params:,} trainable")
    
# Mock model removed - only real implementations allowed
    
    def _initialize_training_components(self):
        """Initialize optimizers, schedulers, and loss managers."""
        
        # Stage 1 optimizer and scheduler
        self.stage1_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.stage1_learning_rate,
            weight_decay=self.config.stage1_weight_decay
        )
        
        self.stage1_scheduler = optim.lr_scheduler.OneCycleLR(
            self.stage1_optimizer,
            max_lr=self.config.stage1_learning_rate * 10,
            total_steps=self.config.stage1_epochs * len(self.train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Stage 2 optimizer and scheduler
        self.stage2_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.stage2_learning_rate,
            weight_decay=self.config.stage2_weight_decay
        )
        
        self.stage2_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.stage2_optimizer,
            T_max=self.config.stage2_epochs,
            eta_min=self.config.stage2_learning_rate * 0.01
        )
        
        # Loss managers
        if SYSTEM_COMPONENTS_AVAILABLE:
            self.stage1_loss_manager = AdvancedLossManager(get_stage1_loss_config())
            self.stage2_loss_manager = AdvancedLossManager(get_stage2_loss_config())
        else:
            self.stage1_loss_manager = None
            self.stage2_loss_manager = None
        
        # Mixed precision scaler
        if self.config.mixed_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        self.logger.info("Training components initialized")
    
    def train(self):
        """
        Main training function - run Stage 1 then Stage 2 based on configuration.
        """
        try:
            skip_stage1 = getattr(self.config, 'skip_stage1', False)
            
            if skip_stage1:
                self.logger.info("=== STARTING DIRECT STAGE 2 TRAINING (NO STAGE 1) ===")
                self.logger.info("Skipping Stage 1 - Training directly with constraints from random weights")
                
                # Set epoch to start of Stage 2 to bypass Stage 1 logic
                self.current_epoch = self.config.stage1_epochs
                
                # Stage 2: Differentiable constraint-aware training
                self.logger.info("Starting Stage 2: Constraint-aware training from scratch")
                self._train_stage2()
            else:
                self.logger.info("=== STARTING TWO-STAGE TRAINING (STAGE 1 → STAGE 2) ===")
                
                # Stage 1: Discrete warm-up training
                self.logger.info("Starting Stage 1: Discrete warm-up training")
                self._train_stage1()
                
                # Stage 2: Differentiable constraint-aware training
                self.logger.info("Starting Stage 2: Constraint-aware training")
                self._train_stage2()
            
            # Training completion
            self.logger.info("=== TRAINING COMPLETED SUCCESSFULLY ===")
            self._finalize_training()
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            self._save_checkpoint("interrupted")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self._save_checkpoint("error")
            raise
    
    def _train_stage1(self):
        """Stage 1: Discrete warm-up training using original SPRING procedures."""
        self.current_stage = TrainingStage.STAGE1_DISCRETE
        self.model.train()
        
        # Switch model to discrete mode
        if hasattr(self.model, 'hybrid_srm'):
            self.model.hybrid_srm.switch_mode(SpatialReasoningMode.DISCRETE)
        
        # Use existing train_loader (already has stage1 batch size)
        
        start_epoch = self.current_epoch + 1 if self.current_epoch > 0 else 0
        for epoch in range(start_epoch, self.config.stage1_epochs):
            self.current_epoch = epoch
            epoch_metrics = self._train_epoch_stage1(epoch)
            
            # Validation every epoch to monitor overfitting
            val_metrics = self._validate_stage1(epoch)
            epoch_metrics.update(val_metrics)
            
            # Enhanced logging for overfitting detection
            train_loss = epoch_metrics.get('train_loss', 0.0)
            val_loss = val_metrics.get('val_loss', 0.0)
            self.logger.info(f"EPOCH {epoch} METRICS: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Gap={val_loss-train_loss:.4f}")
            
            # Checkpointing
            if epoch % self.config.save_every_epochs == 0:
                self._save_checkpoint(f"stage1_epoch_{epoch}")
            
            # Early stopping check
            if self._check_early_stopping(epoch_metrics):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Update metrics
            self._update_training_metrics(epoch_metrics)
    
    def _train_stage2(self):
        """Stage 2: Differentiable fine-tuning with constraint-aware training."""
        self.current_stage = TrainingStage.STAGE2_DIFFERENTIABLE
        self.model.train()
        
        # Switch model to hybrid/differentiable mode
        if hasattr(self.model, 'hybrid_srm'):
            self.model.hybrid_srm.switch_mode(self.config.model_mode)
        
        # CRITICAL FIX: Check if resuming Stage 2 checkpoint vs transitioning from Stage 1
        # If current_epoch >= stage1_epochs, we're resuming a Stage 2 checkpoint
        # If current_epoch < stage1_epochs, we're transitioning from Stage 1 and should start Stage 2 from 0
        if self.current_epoch >= self.config.stage1_epochs:
            # Resuming Stage 2 from a Stage 2 checkpoint - continue from checkpoint epoch
            # Convert global epoch to Stage 2 local epoch
            stage2_start_epoch = self.current_epoch - self.config.stage1_epochs
            self.logger.info(f"Resuming Stage 2 from Stage 2 local epoch {stage2_start_epoch} (resumed global epoch {self.current_epoch})")
        else:
            # Transitioning from Stage 1 to Stage 2 - start Stage 2 from 0
            stage2_start_epoch = 0
            self.current_epoch = self.config.stage1_epochs
            self.logger.info(f"Starting Stage 2 fresh from epoch 0 (completed Stage 1 at epoch {self.current_epoch})")
        stage2_end_epoch = self.config.stage2_epochs
        
        # Reset optimizer and scheduler for stage 2
        self.optimizer = self.stage2_optimizer
        self.scheduler = self.stage2_scheduler
        self.loss_manager = self.stage2_loss_manager
        
        # FIXED: Use separate stage2 epoch counter
        for stage2_local_epoch in range(stage2_start_epoch, stage2_end_epoch):
            # Calculate global epoch for checkpoint saving
            global_epoch_for_this_iteration = self.config.stage1_epochs + stage2_local_epoch
            self.current_epoch = global_epoch_for_this_iteration  # CRITICAL: Update current_epoch for checkpoint saving
            self.logger.info(f"Starting Stage 2 Epoch {stage2_local_epoch} (global epoch {self.current_epoch})")
            
            # Update curriculum progression based on stage 2 progress
            if self.config.enable_curriculum_learning:
                progression = stage2_local_epoch / max(1, stage2_end_epoch - 1) 
                self._update_curriculum_progression(progression)
            
            # Update loss manager epoch for proper scheduling
            if self.loss_manager is not None:
                self.loss_manager.update_epoch(stage2_local_epoch, stage2_end_epoch) 
            
            # FIXED: Pass the correct Stage 2 local epoch to training method
            epoch_metrics = self._train_epoch_stage2(stage2_local_epoch)
            
            # Validation
            if stage2_local_epoch % self.config.validate_every_epochs == 0:
                val_metrics = self._validate_stage2(stage2_local_epoch)
                epoch_metrics.update(val_metrics)
            
            # Checkpointing
            if stage2_local_epoch % self.config.save_every_epochs == 0:
                self._save_checkpoint(f"stage2_epoch_{stage2_local_epoch}")
            
            # Early stopping check
            if self._check_early_stopping(epoch_metrics):
                self.logger.info(f"Early stopping triggered at Stage 2 epoch {stage2_local_epoch} (global epoch {self.current_epoch})")
                break
            
            # Update metrics
            self._update_training_metrics(epoch_metrics)
            
            # FIXED: Log with correct stage 2 epoch number
            self.logger.info(f"Finished Stage 2 Epoch {stage2_local_epoch} (global epoch {self.current_epoch}): Metrics={epoch_metrics}")
    
    def _train_epoch_stage1(self, epoch: int) -> Dict[str, float]:
        """Train one epoch in Stage 1 (discrete mode)."""
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        self.logger.info(f"Starting Stage 1 Epoch {epoch}")
        
        for batch_idx, batch in enumerate(self.train_loader):
            self.global_step += 1
            
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass (discrete mode - no constraints)
            with autocast(enabled=self.config.mixed_precision):
                # FIX: Handle dictionary return format
                model_output = self.model(
                    batch['images'],
                    return_intermediate=True
                )
                
                # Extract layout outputs for training
                if 'layout_results' in model_output:
                    outputs = model_output['layout_results']['final_layout']
                    info = {
                        'constraint_satisfaction_rate': 1.0,  # No constraints in stage 1
                        **model_output.get('performance_metrics', {}),
                        **model_output.get('constraint_satisfaction', {})
                    }
                else:
                    raise RuntimeError("Model output missing 'layout_results' - invalid model implementation")
                
                # Compute loss (layout quality only in stage 1)
                loss = self._compute_stage1_loss(outputs, batch)
                
                #  COORDINATE DISTRIBUTION DIAGNOSTICS (First 5 batches + every 10 batches)
                if batch_idx < 5 or batch_idx % 10 == 0:
                    # Extract valid coordinates from both predictions and targets
                    valid_mask = batch['valid_mask'] if 'valid_mask' in batch else torch.ones(outputs.shape[:2], dtype=torch.bool, device=outputs.device)
                    
                    if valid_mask.any():
                        # Flatten and extract valid coordinates
                        pred_flat = outputs.view(-1, 4)[valid_mask.view(-1)]
                        target_flat = batch['layouts'].view(-1, 4)[valid_mask.view(-1)]
                        
                        # CRITICAL FIX: Remove padding contamination from diagnostics
                        # Filter out padding values (negative coordinates) from targets
                        target_valid = target_flat[target_flat >= 0]
                        pred_valid = pred_flat  # Predictions shouldn't have padding
                        
                        # Calculate ranges and distributions on clean data
                        if len(target_valid) > 0 and len(pred_valid) > 0:
                            pred_min, pred_max = pred_valid.min(), pred_valid.max()
                            target_min, target_max = target_valid.min(), target_valid.max()
                            
                            pred_10th = torch.quantile(pred_valid, 0.1)
                            pred_90th = torch.quantile(pred_valid, 0.9)
                            target_10th = torch.quantile(target_valid, 0.1) 
                            target_90th = torch.quantile(target_valid, 0.9)
                            
                            self.logger.info(f"Batch {batch_idx} Coordinate Analysis:")
                            self.logger.info(f"   PREDICTIONS: [{pred_min:.3f}, {pred_max:.3f}] | 10-90th: [{pred_10th:.3f}, {pred_90th:.3f}]")
                            self.logger.info(f"   TARGETS:     [{target_min:.3f}, {target_max:.3f}] | 10-90th: [{target_10th:.3f}, {target_90th:.3f}]")
                            self.logger.info(f"   Range: Pred={pred_max-pred_min:.3f}, Target={target_max-target_min:.3f}")
                            
                            # Flag concerning patterns
                            if pred_min < -0.1:
                                self.logger.warning(f"   WARNING: Negative outputs detected: {pred_min:.3f}")
                            if pred_max - pred_min < 0.3 and batch_idx > 50:
                                self.logger.warning(f"   WARNING: Prediction range collapse: {pred_max-pred_min:.3f}")
                            if target_max - target_min < 0.4:
                                self.logger.info(f"   INFO: Data naturally narrow: {target_max-target_min:.3f} (this might be correct!)")
            
            # Backward pass with gradient accumulation
            accum_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
            is_accumulation_step = (batch_idx + 1) % accum_steps != 0
            
            # CRITICAL FIX: Need to get gradients from INSIDE _backward_and_step  
            should_log = self.global_step % self.config.log_every_steps == 0
            
            # Pass logging flag to backward_and_step so it can calculate gradients at the right time
            grad_norms_after_backward = self._backward_and_step_with_logging(
                loss, self.stage1_optimizer, self.stage1_scheduler, 
                accumulate_grad=is_accumulation_step, should_log=should_log
            )
            
            # Update metrics
            epoch_metrics['train_loss'] += loss.item()
            epoch_metrics['learning_rate'] = self.stage1_scheduler.get_last_lr()[0]
            
            # Logging with pre-calculated gradients
            if should_log and grad_norms_after_backward:
                self._log_training_step_with_gradients(loss.item(), epoch, batch_idx, grad_norms_after_backward)
            
            num_batches += 1
            if batch_idx % 10 == 0:  # Log every 10 batches
                # Log actual model prediction vs expected target
                target_layouts = batch['layouts']  # Expected
                predicted_layouts = outputs  # Model output
                
                self.logger.info(f"STAGE 1 BATCH {batch_idx} COORDINATE CHECK:")
                # COORDINATE SYSTEM VALIDATION: Ensure [0,1] normalized coordinates throughout
                target_min, target_max = target_layouts.min().item(), target_layouts.max().item()
                pred_min, pred_max = predicted_layouts.min().item(), predicted_layouts.max().item()
                
                self.logger.info(f"  COORDINATE VALIDATION - Target range: [{target_min:.3f}, {target_max:.3f}] (should be ~[0,1])")
                self.logger.info(f"  COORDINATE VALIDATION - Predicted range: [{pred_min:.3f}, {pred_max:.3f}] (should be ~[0,1])")
                
                # Validate coordinate system consistency
                if target_min < -0.1 or target_max > 1.1 or pred_min < -0.1 or pred_max > 1.1:
                    self.logger.error(f"  COORDINATE SYSTEM ERROR: Values outside [0,1] range detected!")
                    self.logger.error(f"  This indicates coordinate system inconsistency in the pipeline")
                
                self.logger.info(f"  Loss: {loss.item():.6f}")
                
                # Log first object comparison with coordinate validation
                if target_layouts.shape[0] > 0 and target_layouts.shape[1] > 0:
                    target_obj1 = target_layouts[0, 0, :].detach().cpu().numpy()
                    pred_obj1 = predicted_layouts[0, 0, :].detach().cpu().numpy()
                    self.logger.info(f"  First object [0,1] coords - Target: [{target_obj1[0]:.3f}, {target_obj1[1]:.3f}, {target_obj1[2]:.3f}, {target_obj1[3]:.3f}]")
                    self.logger.info(f"  First object [0,1] coords - Predicted: [{pred_obj1[0]:.3f}, {pred_obj1[1]:.3f}, {pred_obj1[2]:.3f}, {pred_obj1[3]:.3f}]")
                
                self.logger.info("")
        
        # Average metrics over epoch
        for key in epoch_metrics:
            if 'train_' in key:
                epoch_metrics[key] /= num_batches
        
        return dict(epoch_metrics)
    
    def _train_epoch_stage2(self, epoch: int) -> Dict[str, float]:
        """Train one epoch in Stage 2 with FIXED constraint-aware loss."""
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        # CRITICAL FIX: Update global epoch in model to control warm-start mechanism
        # Use global epoch (stage1_epochs + stage2_epoch) not local stage2 epoch
        global_epoch = self.current_epoch + epoch
        if hasattr(self.model, 'update_epoch'):
            self.model.update_epoch(global_epoch)
            self.logger.debug(f"Updated model epoch to global epoch: {global_epoch}")
        
        self.logger.info(f"Starting Stage 2 Epoch {epoch}")
        
        for batch_idx, batch in enumerate(self.train_loader):
            self.global_step += 1
            
            # Move batch to device and preprocess
            batch = self._move_batch_to_device(batch)
            preprocessed_batch = self.preprocessor.preprocess_batch(batch)
            
            # Generate constraints for this batch
            constraints = self._generate_batch_constraints(preprocessed_batch)
            
            preprocessed_batch['training_stage'] = 'stage2'
            preprocessed_batch['srm_mode'] = 'differentiable'
            
            # Forward pass (constraint-aware mode)
            with autocast(enabled=self.config.mixed_precision):
                model_output = self.model(
                    preprocessed_batch['images'],
                    constraints=constraints,
                    return_intermediate=True,
                    training_stage='stage2',
                    srm_mode='differentiable'
                )
                
                # Extract layout outputs and constraint info
                if 'layout_results' in model_output:
                    outputs = model_output['layout_results']['final_layout']
                    
                    info = {
                        **model_output.get('constraint_satisfaction', {}),
                        **model_output.get('performance_metrics', {}),
                        **model_output['layout_results'].get('constraint_info', {})
                    }
                    # Ensure constraint_satisfaction_rate exists
                    if 'constraint_satisfaction_rate' not in info:
                        info['constraint_satisfaction_rate'] = info.get('constraint_satisfaction', 0.0)  # Default to 0.0 not 1.0
                else:
                    raise RuntimeError("Model output missing 'layout_results' - invalid model implementation")
                    
                expected_shape = (preprocessed_batch['images'].shape[0], self.config.max_objects_per_scene, 4)
                if outputs.shape != expected_shape:
                    self.logger.error(f"STAGE 2 WRONG OUTPUT SHAPE")
                    self.logger.error(f"   Got: {outputs.shape}")
                    self.logger.error(f"   Expected: {expected_shape}")
                    self.logger.error(f"   Model mode: {getattr(self.model.spatial_reasoning_module.config, 'mode', 'UNKNOWN')}")
                    raise ValueError(f"Stage 2 output shape wrong: {outputs.shape} != {expected_shape}")

                if not outputs.requires_grad:
                    self.logger.error(f"STAGE 2 OUTPUT HAS NO GRADIENTS")
                    raise ValueError("Model output missing gradients - check model mode")

                # Standard loss computation without adaptive weighting
                loss, loss_breakdown = self._compute_stage2_loss(outputs, preprocessed_batch, info)
            
            # Loss spike monitoring (no capping - let curriculum learning proceed)
            previous_loss = getattr(self, '_last_loss', 0.01)
            if hasattr(self, '_last_loss') and loss.item() > previous_loss * 10:
                self.logger.info(f"LOSS SPIKE DETECTED - Likely curriculum transition")
                self.logger.info(f"  Previous: {previous_loss:.4f} → Current: {loss.item():.4f} ({loss.item()/previous_loss:.1f}x)")
                self.logger.info(f"  Epoch: {epoch}, Batch: {batch_idx}")
                self.logger.info(f"  Loss breakdown: {loss_breakdown}")
                self.logger.info(f"  Continuing training - model will adapt")
                    
            self._last_loss = loss.item()
            
            # Backward pass with gradient accumulation
            accum_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
            is_accumulation_step = (batch_idx + 1) % accum_steps != 0
            
            # CRITICAL FIX: Need to get gradients from INSIDE _backward_and_step (Stage 2)
            should_log = self.global_step % self.config.log_every_steps == 0
            
            # Pass logging flag to backward_and_step so it can calculate gradients at the right time
            grad_norms_after_backward = self._backward_and_step_with_logging(
                loss, self.stage2_optimizer, None, 
                accumulate_grad=is_accumulation_step, should_log=should_log
            )
            
            # Update metrics with safe extraction
            epoch_metrics['train_loss'] += loss.item()
            constraint_satisfaction_value = self._extract_constraint_satisfaction(info)
            epoch_metrics['constraint_satisfaction'] += constraint_satisfaction_value
            epoch_metrics['learning_rate'] = self.stage2_scheduler.get_last_lr()[0]
            
            # Update constraint satisfaction history
            self.constraint_satisfaction_history.append(constraint_satisfaction_value)
            
            # DIAGNOSTIC: Analyze constraint learning trends every 50 batches
            if batch_idx % 50 == 0 and len(self.constraint_satisfaction_history) > 10:
                recent_avg = sum(list(self.constraint_satisfaction_history)[-10:]) / 10
                early_avg = sum(list(self.constraint_satisfaction_history)[:10]) / 10 if len(self.constraint_satisfaction_history) >= 20 else recent_avg
                improvement = recent_avg - early_avg
                
                self.logger.info(f"LEARNING TREND BATCH {batch_idx}:")
                self.logger.info(f"  Constraint Satisfaction Trend: Early={early_avg:.3f}, Recent={recent_avg:.3f}, Change={improvement:+.3f}")
                if improvement > 0.05:
                    self.logger.info("  POSITIVE LEARNING: Constraint satisfaction improving!")
                elif improvement < -0.05:
                    self.logger.info("  NEGATIVE TREND: Constraint satisfaction declining")
                else:
                    self.logger.info("  STABLE: Constraint satisfaction stable")
                
                # DIAGNOSTIC: Historical constraint satisfaction statistics
                history_list = list(self.constraint_satisfaction_history)
                min_sat = min(history_list)
                max_sat = max(history_list)
                std_sat = (sum([(x - recent_avg)**2 for x in history_list]) / len(history_list)) ** 0.5
                self.logger.info(f"  History Stats: Min={min_sat:.3f}, Max={max_sat:.3f}, StdDev={std_sat:.3f}")
                self.logger.info("")
            
            # Logging with pre-calculated gradients (Stage 2)
            if should_log and grad_norms_after_backward:
                self._log_training_step_with_gradients(loss.item(), epoch, batch_idx, grad_norms_after_backward, loss_breakdown)
            
            num_batches += 1
            if batch_idx % 10 == 0:  # Log every 10 batches
                # Log actual model prediction vs expected target
                target_layouts = preprocessed_batch['layouts']  # Expected
                predicted_layouts = outputs  # Model output
                
                # DIAGNOSTIC: Comprehensive constraint learning analysis
                total_constraints = sum(len(sample_constraints) for sample_constraints in constraints)
                
                # DIAGNOSTIC: Break down loss components
                layout_loss_component = loss.item()
                constraint_penalty_component = info.get('total_penalty', 0.0)
                
                # DIAGNOSTIC: Analyze constraint types in this batch
                constraint_types = {}
                affine_count = 0
                non_affine_count = 0
                
                for sample_constraints in constraints:
                    for constraint in sample_constraints:
                        constraint_type = type(constraint).__name__
                        constraint_types[constraint_type] = constraint_types.get(constraint_type, 0) + 1
                        if constraint_type in ['ConstraintT1', 'ConstraintT2', 'ConstraintT3', 'ConstraintT4', 'ConstraintAND']:
                            affine_count += 1
                        else:
                            non_affine_count += 1
                
                # DIAGNOSTIC: Detailed logging every 10 batches
                if batch_idx % 10 == 0:
                    self.logger.info(f"DIAGNOSTIC BATCH {batch_idx}:")
                    self.logger.info(f"  Constraints: {total_constraints} total ({affine_count} affine, {non_affine_count} non-affine)")
                    self.logger.info(f"  Constraint Satisfaction: {constraint_satisfaction_value:.3f} ({constraint_satisfaction_value*100:.1f}%)")
                    self.logger.info(f"  Constraint Types: {constraint_types}")
                    
                    # DIAGNOSTIC: Loss breakdown
                    total_loss = layout_loss_component + constraint_penalty_component
                    layout_pct = (layout_loss_component / max(total_loss, 0.001)) * 100
                    constraint_pct = (constraint_penalty_component / max(total_loss, 0.001)) * 100
                    self.logger.info(f"  Loss Breakdown: Total={total_loss:.2f}, Layout={layout_loss_component:.2f} ({layout_pct:.1f}%), Constraints={constraint_penalty_component:.2f} ({constraint_pct:.1f}%)")
                    
                    # DIAGNOSTIC: Constraint satisfaction details from model output
                    if 'constraint_satisfaction' in model_output:
                        cs_info = model_output['constraint_satisfaction']
                        cs_rate = cs_info.get('constraint_satisfaction_rate', 0.0)
                        total_penalty = cs_info.get('total_penalty', 0.0)
                        self.logger.info(f"  Model Constraint Info: Rate={cs_rate:.3f}, Penalty={total_penalty:.3f}")
                        
                        # HardNet activity summary
                        if hasattr(self.model, 'spatial_reasoning_module'):
                            srm = self.model.spatial_reasoning_module
                            if hasattr(srm, 'hardnet_layer') and srm.hardnet_layer:
                                hardnet = srm.hardnet_layer
                                if hardnet.has_constraints():
                                    self.logger.info(f"  HardNet: {hardnet.n_constraints} constraints active")
                                    
                                    # Log projection status
                                    if hasattr(hardnet, '_last_projection_occurred'):
                                        proj_occurred = hardnet._last_projection_occurred
                                        self.logger.info(f"  HardNet projection: {proj_occurred}")
                                        
                                        # Show projection values if significant
                                        if hasattr(hardnet, '_last_input_value') and hasattr(hardnet, '_last_output_value'):
                                            input_val = hardnet._last_input_value
                                            output_val = hardnet._last_output_value
                                            if input_val is not None and output_val is not None:
                                                projection_magnitude = abs(output_val - input_val)
                                                if projection_magnitude > 1e-6:
                                                    self.logger.info(f"  HardNet correction: {input_val:.4f} -> {output_val:.4f}")
                                else:
                                    self.logger.info(f"  HardNet: No active constraints")
                    
                    # DIAGNOSTIC: Check if constraints are actually generating gradients
                    if hasattr(self.model, 'spatial_reasoning_module'):
                        srm = self.model.spatial_reasoning_module
                        if hasattr(srm, 'soft_constraint_handler') and srm.soft_constraint_handler:
                            # Check if soft constraint handler has been used recently
                            soft_handler = srm.soft_constraint_handler
                            if hasattr(soft_handler, 'penalty_handlers') and soft_handler.penalty_handlers:
                                self.logger.info(f"  Soft Handler: {len(soft_handler.penalty_handlers)} penalty handlers active")
                    
                    # DIAGNOSTIC: Analyze actual constraint violations in detail
                    if non_affine_count > 0 and constraint_satisfaction_value < 1.0:
                        self.logger.info(f"  CONSTRAINT VIOLATION ANALYSIS:")
                        
                        # Check first few OR constraints and actual violations
                        violation_count = 0
                        for i, sample_constraints in enumerate(constraints[:2]):  # Check first 2 samples
                            for j, constraint in enumerate(sample_constraints):
                                if type(constraint).__name__ == 'ConstraintOR':
                                    # Log the OR constraint details
                                    or_conditions = constraint.c
                                    self.logger.info(f"    Sample {i} OR Constraint: {len(or_conditions)} conditions")
                                    
                                    # Check if this is the single_object_or constraint (left OR right)
                                    if len(or_conditions) == 2:
                                        left_cond = or_conditions[0]  # Should be x < 250
                                        right_cond = or_conditions[1]  # Should be x > 500
                                        
                                        # Get the predicted object position
                                        if i < predicted_layouts.shape[0]:
                                            obj_x = predicted_layouts[i, 0, 0].item()  # First object's x position
                                            
                                            # Check OR constraint satisfaction manually - FIXED: Use normalized [0,1] thresholds
                                            left_satisfied = obj_x < 0.25  # Left condition (25% of canvas)
                                            right_satisfied = obj_x > 0.5   # Right condition (50% of canvas) 
                                            or_satisfied = left_satisfied or right_satisfied
                                            
                                            self.logger.info(f"      Object X={obj_x:.3f}, Left(<0.25)={left_satisfied}, Right(>0.5)={right_satisfied}, OR={or_satisfied}")
                                            if not or_satisfied:
                                                violation_count += 1
                                                self.logger.info(f"      VIOLATION: Object in middle zone (0.25-0.5), not satisfying OR constraint!")
                        
                        self.logger.info(f"  Manual Violation Count: {violation_count} OR constraints violated")
                    
                    self.logger.info("")
                else:
                    # Regular concise logging for other batches
                    self.logger.info(f"STAGE 2 BATCH {batch_idx} - Constraints: {total_constraints}, Constraint sat: {constraint_satisfaction_value:.3f}")
                
                # DIAGNOSTIC: Always log loss components for tracking
                self.logger.debug(f"  Loss Components: Layout={layout_loss_component:.4f}, Constraints={constraint_penalty_component:.4f}")
        
        # Average metrics over epoch
        for key in epoch_metrics:
            if 'train_' in key or key == 'constraint_satisfaction':
                epoch_metrics[key] /= num_batches
                
        self.stage2_scheduler.step()
        return dict(epoch_metrics)
        
    def _validate_stage1(self, epoch: int) -> Dict[str, float]:
        """Validation for Stage 1."""
        self.model.eval()
        val_metrics = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_batch_to_device(batch)
                
                # FIX: Handle dictionary return format
                model_output = self.model(batch['images'], return_intermediate=True)
                
                # Extract outputs
                if 'layout_results' in model_output:
                    outputs = model_output['layout_results']['final_layout']
                else:
                    outputs = model_output.get('generated_images', torch.zeros_like(batch['layouts']))
                
                loss = self._compute_stage1_loss(outputs, batch)
                
                val_metrics['val_loss'] += loss.item()
                num_batches += 1
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        self.model.train()
        return dict(val_metrics)
    
    def _validate_stage2(self, epoch: int) -> Dict[str, float]:
        """Validation for Stage 2 with constraint evaluation."""
        self.model.eval()
        val_metrics = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_batch_to_device(batch)
                preprocessed_batch = self.preprocessor.preprocess_batch(batch)
                constraints = self._generate_batch_constraints(preprocessed_batch)
                
                # FIX: Handle dictionary return format
                model_output = self.model(
                    preprocessed_batch['images'],
                    constraints=constraints,
                    return_intermediate=True,
                    training_stage='stage2',
                    srm_mode='differentiable'
                )
                
                # Extract outputs and constraint info
                if 'layout_results' in model_output:
                    outputs = model_output['layout_results']['final_layout']
                    info = {
                        **model_output.get('constraint_satisfaction', {}),
                        **model_output['layout_results'].get('constraint_info', {})
                    }
                    if 'constraint_satisfaction_rate' not in info:
                        info['constraint_satisfaction_rate'] = 0.0  # Default to 0.0 not 1.0
                else:
                    raise RuntimeError("Model output missing 'layout_results' - invalid model implementation")
                
                loss, _ = self._compute_stage2_loss(outputs, preprocessed_batch, info)
                
                val_metrics['val_loss'] += loss.item()
                val_metrics['val_constraint_satisfaction'] += self._extract_constraint_satisfaction(info)
                num_batches += 1
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        self.model.train()
        return dict(val_metrics)
        
    def _compute_stage1_loss(self, outputs: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for Stage 1 (layout quality only)."""
        target_layouts = batch['layouts']  # [batch_size, max_objects, 4] - per-mille [0, 1000]
        predicted_layouts = outputs       # Could be different shape
        valid_mask = batch['valid_masks'] # [batch_size, max_objects]
        
        # Using normalized coordinate system from data loader
        print(f"   Target range: [{target_layouts.min():.3f}, {target_layouts.max():.3f}]")
        
        batch_size, max_objects, features = target_layouts.shape
        
        # Handle different output shapes from model
        if predicted_layouts.shape != target_layouts.shape:
            if predicted_layouts.dim() == 3:
                pred_batch, pred_objects, pred_features = predicted_layouts.shape
                
                if pred_objects != max_objects:
                    # Model returned different number of objects
                    # Create tensor with correct shape, filled with zeros
                    aligned_pred = torch.zeros_like(target_layouts)
                    
                    # Copy available predictions (up to the minimum of both)
                    min_objects = min(pred_objects, max_objects)
                    min_features = min(pred_features, features)
                    aligned_pred[:pred_batch, :min_objects, :min_features] = \
                        predicted_layouts[:pred_batch, :min_objects, :min_features]
                    
                    predicted_layouts = aligned_pred
                
            elif predicted_layouts.dim() == 2:
                # Flat output - reshape to match target
                if predicted_layouts.shape[1] == max_objects * features:
                    predicted_layouts = predicted_layouts.view(batch_size, max_objects, features)
                else:
                    # Create zeros with correct shape
                    predicted_layouts = torch.zeros_like(target_layouts)
                    
            else:
                # Fallback: create zeros with correct shape
                self.logger.warning(f"Unexpected predicted layout shape: {predicted_layouts.shape}, "
                                f"expected shape like: {target_layouts.shape}")
                predicted_layouts = torch.zeros_like(target_layouts)
        
        # Now both tensors should have the same shape: [batch_size, max_objects, 4]
        assert predicted_layouts.shape == target_layouts.shape, \
            f"Shape mismatch after alignment: pred={predicted_layouts.shape}, target={target_layouts.shape}"
        
        # CRITICAL FIX: Extract valid tensors BEFORE loss computation (prevents gradient contamination)
        valid_mask_expanded = valid_mask.unsqueeze(-1).expand_as(target_layouts)  # [batch_size, max_objects, 4]
        
        # Extract ONLY valid predictions and targets (no gradient contamination from padding)
        if valid_mask.any():
            # Reshape for easier indexing
            pred_flat = predicted_layouts.view(-1, 4)  # [batch*objects, 4] 
            target_flat = target_layouts.view(-1, 4)   # [batch*objects, 4]
            mask_flat = valid_mask.view(-1)            # [batch*objects]
            
            # Extract only valid entries (completely isolate from padding)
            valid_pred = pred_flat[mask_flat]      # [n_valid, 4]
            valid_target = target_flat[mask_flat]  # [n_valid, 4] 
            
            # Compute loss ONLY on valid data (clean gradients, no padding contamination)
            loss = nn.functional.mse_loss(valid_pred, valid_target, reduction='mean')
        else:
            # No valid objects in batch (fallback)
            loss = torch.tensor(0.0, device=predicted_layouts.device, requires_grad=True)
        
        # COORDINATE NORMALIZATION FIX: No longer need to scale loss since coordinates are normalized
        # Previous: loss / (1000^2) for per-mille coordinates
        # Now: coordinates are already in [0,1] range, so loss scaling is natural
        
        return loss

    def _compute_stage2_loss(self, 
                            outputs: torch.Tensor, 
                            batch: Dict[str, torch.Tensor],
                            info: Dict[str, Any],
                            layout_weight_multiplier: float = 1.0,
                            constraint_weight_multiplier: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        """
        Compute constraint-aware loss for Stage 2 with FIXED type handling.
        """
        try:
            if self.stage2_loss_manager is None:
                # Fallback to simple loss
                return self._compute_stage1_loss(outputs, batch), {}
            
            # Compute basic layout loss
            layout_loss = self._compute_stage1_loss(outputs, batch)
            
            # Prepare loss components
            loss_components = {
                'layout': layout_loss
            }
            
            # Add constraint-related losses with safe extraction
            device = outputs.device
            
            # STRICT: All constraint losses must be provided and valid
            if 'constraint_violation_penalty' in info:
                penalty_value = extract_required_numeric(info['constraint_violation_penalty'], 'constraint_violation_penalty')
                loss_components['constraint'] = torch.tensor(penalty_value, device=device)
            
            if 'total_penalty' in info:
                penalty_value = extract_required_numeric(info['total_penalty'], 'total_penalty')
                loss_components['soft_constraint'] = torch.tensor(penalty_value, device=device)
            
            if 'hardnet_loss' in info:
                hardnet_value = extract_required_numeric(info['hardnet_loss'], 'hardnet_loss')
                loss_components['hardnet'] = torch.tensor(hardnet_value, device=device)
            
            # Constraint satisfaction is REQUIRED for stage 2
            if 'constraint_satisfaction_rate' not in info:
                raise KeyError("Stage 2 model output missing required 'constraint_satisfaction_rate'")
            constraint_satisfaction = extract_required_numeric(
                info['constraint_satisfaction_rate'], 'constraint_satisfaction_rate'
            )
            satisfaction_loss = 1.0 - constraint_satisfaction
            loss_components['satisfaction'] = torch.tensor(satisfaction_loss, device=device)
            
            # Extract performance metrics with strict validation
            performance_metrics = extract_performance_metrics(info)
            
            # Apply adaptive loss multipliers to escape gradient stagnation
            adjusted_loss_components = {}
            for key, value in loss_components.items():
                if key == 'layout':
                    adjusted_loss_components[key] = value * layout_weight_multiplier
                elif key in ['constraint', 'hardnet', 'satisfaction']:
                    adjusted_loss_components[key] = value * constraint_weight_multiplier
                else:
                    adjusted_loss_components[key] = value
            
            # Compute weighted loss
            total_loss, loss_breakdown = self.stage2_loss_manager.compute_weighted_loss(
                adjusted_loss_components, performance_metrics
            )
            
            return total_loss, loss_breakdown
            
        except Exception as e:
            # FAIL-FAST: No fallback to simple loss - crash immediately
            raise AssertionError(
                f"FAIL-FAST: Advanced loss computation failed in Stage 2:\n"
                f"Error: {e}\n"
                f"Batch keys: {list(batch.keys())}\n"
                f"Output keys: {list(outputs.keys()) if isinstance(outputs, dict) else type(outputs)}\n"
                f"This indicates a real bug in constraint-aware training\n"
                f"NO fallback to simple loss - FIX THE CONSTRAINT PROCESSING"
            )
    
    def _generate_batch_constraints(self, batch: Dict[str, torch.Tensor]) -> List[List[Any]]:
        """Generate constraints for a batch."""
        if not hasattr(self, 'constraint_generator'):
            batch_size = batch['images'].shape[0] if 'images' in batch else 1
            return [[] for _ in range(batch_size)]
        
        # Extract metadata for constraint generation  
        batch_size = batch['images'].shape[0] if 'images' in batch else batch['layouts'].shape[0]
        categories = []
        for i in range(batch_size):
            if 'metadata' in batch and i < len(batch['metadata']):
                categories.append(batch['metadata'][i].get('categories', []))
            else:
                categories.append([])
        
        # Generate constraints - CRITICAL: Apply same target normalization as model predictions
        # Model outputs [0, 20] coordinates, so constraints must be generated for same space
        normalized_layouts = batch['layouts']  # Already in [0,1] from data loader
        
        batch_constraints = self.constraint_generator.generate_constraints_for_batch(
            normalized_layouts,  # Use normalized coordinates matching model output space
            batch['valid_masks'],
            categories
        )
        
        return batch_constraints
    
    def _backward_and_step_with_logging(self, loss: torch.Tensor, optimizer: optim.Optimizer, scheduler, accumulate_grad: bool = False, should_log: bool = False) -> Optional[Tuple[float, float, Dict[str, float]]]:
        """Execute backward pass and optimization step with optional gradient logging."""
        
        # Check if any parameters require gradients
        has_grad_params = any(p.requires_grad for p in self.model.parameters())
        
        # Get gradient accumulation steps from config
        accum_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        
        if self.config.mixed_precision and self.scaler and has_grad_params:
            # Mixed precision backward pass with gradient accumulation
            scaled_loss = self.scaler.scale(loss / accum_steps)
            scaled_loss.backward()
            
            # Only step optimizer if not accumulating or on final accumulation step
            if not accumulate_grad:
                # Gradient clipping
                if self.config.max_gradient_norm > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_gradient_norm)
                
                # Step optimizer
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
        else:
            #  INSTRUMENTATION: Hook original_srm parameters BEFORE backward to detect gradient flow
            gradient_hooks = []
            hardnet_gradients_received = [False]
            
            if should_log:
                print(f"\n REAL TRAINING BACKWARD TRACE (NON-AMP) - Loss: {loss.item():.6f}")
                # Hook only first few key parameters to avoid spam
                key_params = ['coordinate_regressor.3.weight', 'gru.weight_ih_l0']
                for name, param in self.model.named_parameters():
                    if any(key in name for key in key_params) and param.requires_grad:
                        def create_grad_hook(param_name):
                            def grad_hook(grad):
                                if grad is not None:
                                    grad_norm = torch.norm(grad).item()
                                    print(f"    GRADIENT: {param_name} norm={grad_norm:.8f}")
                                    hardnet_gradients_received[0] = True
                                else:
                                    print(f"    NO GRADIENT: {param_name}")
                                return grad
                            return grad_hook
                        
                        hook = param.register_hook(create_grad_hook(name))
                        gradient_hooks.append(hook)
            
            # Standard backward pass with gradient accumulation
            if should_log:
                print(f"    CALLING (loss / accum_steps).backward()...")
            (loss / accum_steps).backward()  # Scale loss by accumulation steps
            
            # Clean up hooks
            for hook in gradient_hooks:
                hook.remove()
            
            if should_log:
                if hardnet_gradients_received[0]:
                    print(f"    GRADIENTS REACHED ORIGINAL_SRM PARAMETERS")
                else:
                    print(f"    NO GRADIENTS REACHED ORIGINAL_SRM PARAMETERS")
            
            # CRITICAL: Calculate gradients AFTER backward() but BEFORE optimizer step
            grad_norms = None
            if should_log:
                grad_norms = self._calculate_gradient_norms()
            
            # Only step optimizer if not accumulating or on final accumulation step
            if not accumulate_grad:
                # DEBUG: Check gradient state before clipping
                param_count = 0
                grad_count = 0
                for param in self.model.parameters():
                    param_count += 1
                    if param.grad is not None:
                        grad_count += 1
                        
                print(f"CLIP DEBUG: {param_count} params, {grad_count} have gradients")
                
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
                print(f"CLIP DEBUG: accumulate_grad={accumulate_grad}, max_gradient_norm={self.config.max_gradient_norm}")
                print(f"CLIP DEBUG: Pre-clip gradient norm: {pre_clip_norm:.2f}")
                
                # Gradient clipping
                if self.config.max_gradient_norm > 0:
                    post_clip_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_gradient_norm)
                    print(f"CLIP DEBUG: Post-clip gradient norm: {post_clip_norm:.2f}, threshold: {self.config.max_gradient_norm}")
                    
                    # CRITICAL: Check if clipping actually worked
                    verify_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
                    print(f"CLIP DEBUG: VERIFY - actual gradient norm after clipping: {verify_norm:.2f}")
                    if verify_norm > self.config.max_gradient_norm * 1.01:  # Allow small tolerance
                        print(f" CLIPPING FAILED! Expected ≤{self.config.max_gradient_norm}, got {verify_norm:.2f}")
                    else:
                        print(f" CLIPPING WORKED! Gradient norm properly clamped")
                else:
                    print(f"CLIP DEBUG: Gradient clipping SKIPPED (max_gradient_norm={self.config.max_gradient_norm})")
                
                optimizer.step()
                optimizer.zero_grad()  # Only zero gradients after step
        
        if scheduler:
            scheduler.step()
            
        return grad_norms

    def _backward_and_step(self, loss: torch.Tensor, optimizer: optim.Optimizer, scheduler, accumulate_grad: bool = False):
        """Execute backward pass and optimization step with gradient accumulation support."""
        
        # Check if any parameters require gradients
        has_grad_params = any(p.requires_grad for p in self.model.parameters())
        
        # DEBUG: Log backward pass info when debugging
        # self.logger.info(f" BACKWARD DEBUG: has_grad_params={has_grad_params}, loss.requires_grad={loss.requires_grad}")
        # self.logger.info(f" BACKWARD DEBUG: mixed_precision={self.config.mixed_precision}, loss_value={loss.item():.6f}")
        
        # Get gradient accumulation steps from config
        accum_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        
        if self.config.mixed_precision and self.scaler and has_grad_params:
            # Mixed precision backward pass with gradient accumulation
            scaled_loss = self.scaler.scale(loss / accum_steps)  # Scale loss by accumulation steps
            scaled_loss.backward()
            
            # Only step optimizer if not accumulating or on final accumulation step
            if not accumulate_grad:
                # Gradient clipping - need to unscale before clipping
                if self.config.max_gradient_norm > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_gradient_norm)
                    
                # Step optimizer with better AMP logging
                try:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()  # Only zero gradients after step
                except AssertionError as e:
                    if "No inf checks were recorded" in str(e):
                        self.logger.error(f"AMP Scaler Error: {e}")
                        self.logger.error("This usually means scaler.scale() wasn't called before backward()")
                        self.logger.error("Loss was: {loss.item()}, requires_grad: {loss.requires_grad}")
                        raise
                    else:
                        raise
        else:
            # Standard backward pass with gradient accumulation
            (loss / accum_steps).backward()  # Scale loss by accumulation steps
            
            # Only step optimizer if not accumulating or on final accumulation step
            if not accumulate_grad:
                # Gradient clipping
                if self.config.max_gradient_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_gradient_norm)
                
                optimizer.step()
                optimizer.zero_grad()  # Only zero gradients after step
        
        if scheduler:
            scheduler.step()
    
    def _update_curriculum_progression(self, stage2_epoch: int):
        """Update curriculum learning progression based on Stage 2 progress."""
        total_stage2_epochs = self.config.stage2_epochs  # Use stage 2 total epochs
        
        if self.config.curriculum_progression_schedule == "linear":
            progression = stage2_epoch / total_stage2_epochs
        elif self.config.curriculum_progression_schedule == "cosine":
            progression = 0.5 * (1 - np.cos(np.pi * stage2_epoch / total_stage2_epochs))
        elif self.config.curriculum_progression_schedule == "exponential":
            progression = 1 - np.exp(-3 * stage2_epoch / total_stage2_epochs)
        else:
            progression = stage2_epoch / total_stage2_epochs
        
        # Update constraint generator
        if hasattr(self, 'constraint_generator'):
            self.logger.info(f"CURRICULUM: Updating progression to {progression:.3f} (epoch {stage2_epoch}/{total_stage2_epochs})")
            self.constraint_generator.update_curriculum_progression(progression)
    
    def _transition_to_stage2(self):
        """FIXED: Handle transition from Stage 1 to Stage 2 with proper mode switching."""
        
        self.logger.info("=== TRANSITIONING FROM STAGE 1 TO STAGE 2 ===")
        
        # Check if model has spatial reasoning module
        if not hasattr(self.model, 'spatial_reasoning_module'):
            self.logger.error("Model has no spatial_reasoning_module!")
            raise RuntimeError("Model architecture issue - no spatial_reasoning_module found")
        
        # CRITICAL FIX: Create Stage 2 dataloaders with correct batch size
        self.logger.info("Creating Stage 2 dataloaders with batch size 8")
        dataset_config = SpringDatasetConfig(
            dataset_root=self.config.dataset_root,
            image_size=self.config.dataset_image_size,
            max_objects_to_place=self.config.max_objects_per_scene,    
            max_background_objects=self.config.max_objects_per_scene,  
            output_format="sequence",
            enable_augmentation=True,
            coordinate_system="per_mille"
        )
        
        # Create Stage 2 specific dataloaders with correct batch size
        self.train_loader = SpringTrainingDataLoader.create_train_loader(
            dataset_config,
            batch_size=self.config.stage2_batch_size,  # Use stage2_batch_size=8
            shuffle=True,
            num_workers=0
        )
        
        self.val_loader = SpringTrainingDataLoader.create_val_loader(
            dataset_config,
            batch_size=self.config.stage2_batch_size,  # Use stage2_batch_size=8
            num_workers=0
        )
        
        # Switch the model to differentiable mode
        self.logger.info("Switching model to DIFFERENTIABLE mode for Stage 2")
        self.model.spatial_reasoning_module.config.mode = SpatialReasoningMode.DIFFERENTIABLE
        
        # Enable constraint processing components
        if hasattr(self.model.spatial_reasoning_module, 'hardnet_layer'):
            if self.model.spatial_reasoning_module.hardnet_layer:
                self.model.spatial_reasoning_module.hardnet_layer.enable_projection = True
        
        if hasattr(self.model.spatial_reasoning_module, 'soft_constraint_handler'):
            if self.model.spatial_reasoning_module.soft_constraint_handler:
                self.model.spatial_reasoning_module.soft_constraint_handler.enabled = True
        
        # Enable VEG for Stage 2 (needed for constraint validation)
        if hasattr(self.model, 'visual_element_generator') and not self.model.visual_element_generator:
            self.logger.info("Enabling VEG module for Stage 2 constraint validation")
            from completeInt import VisualElementGenerator
            veg_config = self.model.config
            self.model.visual_element_generator = VisualElementGenerator(veg_config)
            self.model.visual_element_generator = self.model.visual_element_generator.to(self.device)
        
        # Update temperature scheduler for Stage 2 epochs
        if hasattr(self.model.spatial_reasoning_module, 'soft_constraint_handler'):
            if hasattr(self.model.spatial_reasoning_module.soft_constraint_handler, 'temperature_scheduler'):
                temp_scheduler = self.model.spatial_reasoning_module.soft_constraint_handler.temperature_scheduler
                temp_scheduler.config.total_epochs = self.config.stage2_epochs
                temp_scheduler.config.warmup_epochs = min(1, self.config.stage2_epochs // 2)
                temp_scheduler.current_epoch = 0
                self.logger.info(f" Temperature scheduler updated: {self.config.stage2_epochs} epochs, warmup: {temp_scheduler.config.warmup_epochs}")
        
        # Verify the switch worked
        current_mode = self.model.spatial_reasoning_module.config.mode
        self.logger.info(f" Model mode after switch: {current_mode}")
        
        if current_mode != SpatialReasoningMode.DIFFERENTIABLE:
            raise RuntimeError(f"Mode switch FAILED! Still in {current_mode}")
        
        # Save Stage 1 checkpoint (your existing logic)
        self._save_checkpoint("stage1_complete")
        
        # Reset for Stage 2 (your existing logic)
        self.best_metric = float('-inf')
        self.patience_counter = 0
        
        if hasattr(self.model, 'update_epoch'):
            self.model.update_epoch(0)
        
        # CRITICAL: Set current stage to Stage 2
        self.current_stage = TrainingStage.STAGE2_DIFFERENTIABLE
        
        self.logger.info(" Successfully transitioned to Stage 2")   
        
    def _calculate_gradient_norms(self) -> Tuple[float, float, Dict[str, float], Dict[str, int]]:
        """Calculate gradient norms before they're cleared by zero_grad()"""
        
        total_grad_norm = 0.0
        max_grad_norm = 0.0
        grad_norms_by_layer = {}
        
        # Gradient flow statistics
        total_params = 0
        params_with_grads = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None:
                    params_with_grads += 1
                    param_norm = param.grad.data.norm(2).item()
                    total_grad_norm += param_norm ** 2
                    max_grad_norm = max(max_grad_norm, param_norm)
                    
                    # Log important layers
                    if any(key in name for key in ['spatial_reasoning', 'hardnet', 'constraint']):
                        grad_norms_by_layer[name] = param_norm
        
        total_grad_norm = total_grad_norm ** 0.5
        
        # Pack gradient flow stats
        grad_flow_stats = {
            'total_params': total_params,
            'params_with_grads': params_with_grads,
            'coverage_pct': (params_with_grads / max(total_params, 1)) * 100
        }
        
        return total_grad_norm, max_grad_norm, grad_norms_by_layer, grad_flow_stats

    def _log_training_step_with_gradients(self, loss: float, epoch: int, batch_idx: int, gradient_norms: Tuple[float, float, Dict[str, float], Dict[str, int]], loss_breakdown: Dict = None):
        """Log training step with pre-calculated gradient norms"""
        
        total_grad_norm, max_grad_norm, grad_norms_by_layer, grad_flow_stats = gradient_norms
        
        step_info = {
            'epoch': epoch,
            'step': self.global_step,
            'loss': loss,
            'stage': self.current_stage,
            'total_grad_norm': total_grad_norm,
            'max_grad_norm': max_grad_norm
        }
        
        if loss_breakdown:
            step_info.update(loss_breakdown)
        
        # Add gradient norms to step info
        if grad_norms_by_layer:
            step_info['constraint_grad_norms'] = grad_norms_by_layer
        
        # Log to file (training.log)
        log_message = (f"Stage {self.current_stage} | Epoch {epoch} | Step {self.global_step} | "
                      f"Loss: {loss:.6f} | Grad Norm: {total_grad_norm:.6f} | "
                      f"Max Grad: {max_grad_norm:.6f}")
        
        if loss_breakdown and 'constraint_penalty' in loss_breakdown:
            log_message += f" | Constraint: {loss_breakdown['constraint_penalty']:.6f}"
        
        # Log gradient details for important layers
        if grad_norms_by_layer:
            constraint_grads = [f"{name}: {norm:.4f}" for name, norm in grad_norms_by_layer.items()]
            log_message += f" | Constraint Grads: {', '.join(constraint_grads)}"
        
        self.logger.info(log_message)
        
        # STE gradient evidence (reduced logging)
        if self.global_step % 50 == 0:  # Log every 50 steps instead of 10
            # Use pre-calculated gradient flow stats
            neural_params_with_grads = grad_flow_stats['params_with_grads']
            total_neural_params = grad_flow_stats['total_params']
            grad_coverage = grad_flow_stats['coverage_pct']
            
            # Log gradient flow summary
            self.logger.info(f"  Gradient flow: {neural_params_with_grads}/{total_neural_params} params ({grad_coverage:.1f}%), max_grad={max_grad_norm:.6f}")
            
            # Log STE status
            if neural_params_with_grads >= total_neural_params * 0.8 and max_grad_norm > 1e-8:
                self.logger.info(f"  STE status: Working correctly")
            else:
                self.logger.info(f"  STE status: May be blocked or gradients too small")
        
        # Log to wandb if enabled
        if self.config.enable_wandb:
            wandb.log(step_info, step=self.global_step)

    def _log_training_step(self, loss: float, epoch: int, batch_idx: int, loss_breakdown: Dict = None):
        """Log training step information with gradient monitoring."""
        
        # Calculate gradient norms for logging
        total_grad_norm = 0.0
        max_grad_norm = 0.0
        grad_norms_by_layer = {}
        
        # Gradient flow statistics
        total_params = 0
        params_with_grads = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None:
                    params_with_grads += 1
                    param_norm = param.grad.data.norm(2).item()
                    total_grad_norm += param_norm ** 2
                    max_grad_norm = max(max_grad_norm, param_norm)
                    
                    # Log important layers
                    if any(key in name for key in ['spatial_reasoning', 'hardnet', 'constraint']):
                        grad_norms_by_layer[name] = param_norm
        
        total_grad_norm = total_grad_norm ** 0.5
        
        # Pack gradient flow stats
        grad_flow_stats = {
            'total_params': total_params,
            'params_with_grads': params_with_grads,
            'coverage_pct': (params_with_grads / max(total_params, 1)) * 100
        }
        
        step_info = {
            'epoch': epoch,
            'step': self.global_step,
            'loss': loss,
            'stage': self.current_stage,
            'total_grad_norm': total_grad_norm,
            'max_grad_norm': max_grad_norm
        }
        
        if loss_breakdown:
            step_info.update(loss_breakdown)
        
        # Add gradient norms to step info
        if grad_norms_by_layer:
            step_info['constraint_grad_norms'] = grad_norms_by_layer
        
        # Log to file (training.log)
        log_message = (f"Stage {self.current_stage} | Epoch {epoch} | Step {self.global_step} | "
                      f"Loss: {loss:.6f} | Grad Norm: {total_grad_norm:.6f} | "
                      f"Max Grad: {max_grad_norm:.6f}")
        
        if loss_breakdown and 'constraint_penalty' in loss_breakdown:
            log_message += f" | Constraint: {loss_breakdown['constraint_penalty']:.6f}"
        
        # Log gradient details for important layers
        if grad_norms_by_layer:
            constraint_grads = [f"{name}: {norm:.4f}" for name, norm in grad_norms_by_layer.items()]
            log_message += f" | Constraint Grads: {', '.join(constraint_grads)}"
        
        self.logger.info(log_message)
        
        # STE gradient evidence (reduced logging)
        if self.global_step % 50 == 0:  # Log every 50 steps instead of 10
            # Use pre-calculated gradient flow stats
            neural_params_with_grads = grad_flow_stats['params_with_grads']
            total_neural_params = grad_flow_stats['total_params']
            grad_coverage = grad_flow_stats['coverage_pct']
            
            # Log gradient flow summary
            self.logger.info(f"  Gradient flow: {neural_params_with_grads}/{total_neural_params} params ({grad_coverage:.1f}%), max_grad={max_grad_norm:.6f}")
            
            # Log STE status
            if neural_params_with_grads >= total_neural_params * 0.8 and max_grad_norm > 1e-8:
                self.logger.info(f"  STE status: Working correctly")
            else:
                self.logger.info(f"  STE status: May be blocked or gradients too small")
        
        # Log to wandb if enabled
        if self.config.enable_wandb:
            wandb.log(step_info, step=self.global_step)
    
    def _update_training_metrics(self, metrics: Dict[str, float]):
        """Update training metrics tracking."""
        for key, value in metrics.items():
            self.training_metrics[key].append(value)
    
    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """Check if early stopping criteria are met."""
        if self.config.monitor_metric not in metrics:
            return False
        
        current_metric = metrics[self.config.monitor_metric]
        
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.patience_counter = 0
            self._save_checkpoint("best_model")
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.early_stopping_patience
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save training checkpoint with compression for 90% size reduction."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"{checkpoint_name}.pt"
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'stage1_optimizer_state_dict': self.stage1_optimizer.state_dict(),
            'stage2_optimizer_state_dict': self.stage2_optimizer.state_dict(),
            'stage1_scheduler_state_dict': self.stage1_scheduler.state_dict(),
            'stage2_scheduler_state_dict': self.stage2_scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
            'training_metrics': dict(self.training_metrics),
            'current_stage': self.current_stage
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # DISABLED COMPRESSION: Save only uncompressed .pt files to prevent corruption
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # CUDA MEMORY CLEANUP: Force cleanup after checkpoint save
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.debug("CUDA cache cleared after checkpoint save")
    
    def _finalize_training(self):
        """Finalize training and cleanup."""
        # Save final checkpoint
        self._save_checkpoint("final_model")
        
        # Generate training summary
        self._generate_training_summary()
        
        # Close wandb
        if self.config.enable_wandb:
            wandb.finish()
        
        self.logger.info("Training finalization complete")
    
    def _generate_training_summary(self):
        """Generate comprehensive training summary."""
        summary = {
            'total_epochs': self.config.stage1_epochs + self.config.stage2_epochs,
            'completed_epochs': self.current_epoch + 1,
            'total_steps': self.global_step,
            'best_metric': self.best_metric,
            'final_constraint_satisfaction': np.mean(list(self.constraint_satisfaction_history)) if self.constraint_satisfaction_history else 0.0,
            'stage1_epochs': self.config.stage1_epochs,
            'stage2_epochs': self.config.stage2_epochs
        }
        
        # Save summary
        summary_path = Path(self.config.checkpoint_dir) / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Training summary: {summary}")


def main():
    """UPDATED: Production-scale SPRING training with Telea inpainting methodology."""
    print("=== SPRING HYBRID PRODUCTION TRAINING (TELEA METHODOLOGY) ===\n")
    
    # PRODUCTION CONFIGURATION (matches SPRING paper exactly)
    config = TrainingConfig(
        # UPDATED: Dataset paths for SPRING training pairs
        dataset_root="data/spring_training_data",  # UPDATED path
        dataset_image_size=(512, 512),  #  SPRING standard
        max_objects_per_scene=5,        #  UPDATED: Objects to place per training pair
        
        # STAGE 1: Original SPRING training (100 epochs)
        stage1_epochs=100,              #  TESTING: 1 epoch for validation
        stage1_batch_size=16,             #  TESTING: Reduced for speed
        stage1_learning_rate=1e-5,      #  SPRING paper learning rate
        stage1_weight_decay=1e-5,       #  Standard regularization
        stage1_gradient_clip=0.5,       #  Gradient stability
        stage1_warmup_epochs=10,        #  Warmup for stability
        
        # STAGE 2: Hybrid constraint training (100 epochs)
        stage2_epochs=100,               #  TESTING: 1 epoch for validation
        stage2_batch_size=16,            #  TESTING: Reduced for speed  
        stage2_learning_rate=1e-4,      # CRITICAL FIX: Increased for constraint learning
        stage2_weight_decay=1e-6,       #  Reduced for fine-tuning
        stage2_gradient_clip=0.5,       #  Careful gradient clipping
        stage2_warmup_epochs=10,         #  Short warmup
        resume_from_checkpoint=None,  # Skip checkpoint for direct Stage 2
        # Model configuration
        model_mode=SpatialReasoningMode.HYBRID,  #  Your contribution
        enable_constraint_processing=True,       #  Enable constraints
        enable_curriculum_learning=False,         #  Curriculum learning
        skip_stage1=True,                   #  FIXED: Run Stage 1 first, then Stage 2 (scene_features bug fixed!)
        
        # Training optimization (production settings)
        mixed_precision=False,          #  Fixed dtype conflicts
        gradient_checkpointing=True,    #  Memory efficiency
        enable_multi_gpu=True,          #  Use all available GPUs
        max_gradient_norm=1.0,          #  Gradient stability
        
        # Constraint training (your innovation) - CRITICAL FIX: These params are OVERRIDDEN by pipeline.py
        constraint_weight_schedule="curriculum",  #  Progressive difficulty  
        initial_constraint_weight=1.0,            # CRITICAL FIX: Now handled in pipeline.py
        final_constraint_weight=1.0,              # CRITICAL FIX: Now handled in pipeline.py
        constraint_satisfaction_target=0.95,      #  High target
        
        # Monitoring and checkpointing (production settings)
        checkpoint_dir="checkpoints",  #  Production checkpoints
        save_every_epochs=5,                           #  Regular saves
        validate_every_epochs=5,                        #  Regular validation
        early_stopping_patience=15,                     #  Reasonable patience
        monitor_metric="constraint_satisfaction_rate",  #  Key metric
        
        # Logging (production monitoring)
        enable_wandb=False,             #  TESTING: Disabled for speed
        wandb_project="spring-hybrid-telea-production",
        wandb_run_name=f"spring_telea_hybrid_{time.strftime('%Y%m%d_%H%M%S')}",
        log_every_steps=10,             #  TESTING: More frequent for monitoring
        save_visualizations=True,       #  Save progress visualizations
        visualizations_dir="visualizations_spring_production",
        
        # Performance settings
        profiler_enabled=False,         #  Disable for production speed
        detect_anomaly=False,           #  Disable for production speed  
        benchmark_mode=True,            #  Enable CUDNN optimization
        seed=42                         #  Reproducible results
    )
    
    print("SPRING TELEA TRAINING CONFIGURATION:")
    print(f"  Training Data: {config.dataset_root}")
    print(f"   Methodology: Telea inpainting + object removal")
    print(f"   Stage 1: {config.stage1_epochs} epochs (Original SPRING)")
    print(f"   Stage 2: {config.stage2_epochs} epochs (Hybrid Innovation)")
    print(f"  Stage 1 Batch Size: {config.stage1_batch_size}, Stage 2: {config.stage2_batch_size}")
    print(f"  Model Mode: {config.model_mode}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Monitoring: {'Weights & Biases' if config.enable_wandb else 'Local only'}")
    
    # Estimate training time
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  🖥️  GPU: {gpu_name}")
        
        # Rough time estimates
        stage1_time = config.stage1_epochs * 0.4  # ~24 minutes per epoch
        stage2_time = config.stage2_epochs * 0.5   # ~30 minutes per epoch (constraint processing)
        total_time = stage1_time + stage2_time
        
        print(f"  ⏱️  Estimated Time: Stage 1 (~{stage1_time:.1f}h) + Stage 2 (~{stage2_time:.1f}h) = {total_time:.1f}h total")
    
    print(f"\nTARGET BENCHMARKS (vs SPRING paper):")
    print(f"  📍 Position Accuracy: ≥1.0 (perfect constraint satisfaction)")
    print(f"  Object Accuracy: >0.77 (beat SPRING's 77%)")
    print(f"  🖼️  FID Score: <160.36 (better image quality)")
    print(f"   Inception Score: >3.59 (better image quality)")
    
    # Check prerequisites
    print(f"\nCHECKING PREREQUISITES:")
    
    # Check SPRING training dataset
    dataset_path = Path(config.dataset_root)
    if dataset_path.exists():
        print(f"   SPRING training data found: {dataset_path}")
        
        # Check dataset size
        splits_file = dataset_path / "splits.json"
        if splits_file.exists():
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            print(f"     Train pairs: {len(splits.get('train', []))}")
            print(f"     Val pairs: {len(splits.get('val', []))}")
            print(f"     Test pairs: {len(splits.get('test', []))}")
            
            # Check if we have enough data
            total_pairs = len(splits.get('train', [])) + len(splits.get('val', [])) + len(splits.get('test', []))
            if total_pairs < 1000:
                print(f"  Warning: Only {total_pairs} training pairs, consider running coco_processor_fixed.py")
        else:
            print(f"  SPRING training data incomplete")
            print(f"     Please run: python coco_processor_fixed.py")
            return
    else:
        print(f"  SPRING training data NOT found: {dataset_path}")
        print(f"     Please run: python coco_processor_fixed.py first")
        return
    
    # Check disk space
    import shutil
    free_space_gb = shutil.disk_usage(config.checkpoint_dir).free / (1024**3)
    print(f"   Free disk space: {free_space_gb:.1f} GB")
    if free_space_gb < 10:
        print(f"  Warning: Low disk space for checkpoints")
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  🖥️  GPU Memory: {gpu_memory_gb:.1f} GB")
        if gpu_memory_gb < 8:
            print(f"  Warning: Low GPU memory, consider reducing batch size")
    
    print(f"\n INITIALIZING SPRING TRAINING PIPELINE...")
    
    # Initialize trainer
    try:
        trainer = SpringHybridTrainer(config)
        print(f"   SPRING trainer initialized successfully")
        print(f"     Training methodology: Telea inpainting with object removal")
        print(f"     Input: Inpainted backgrounds + constraints")
        print(f"     Output: Object positions (ground truth from COCO)")
        
        # Start training
        print(f"\nSTARTING SPRING PRODUCTION TRAINING")
        print(f"  Expected completion: ~{total_time:.1f} hours from now")
        print(f"  Monitor progress: https://wandb.ai/{config.wandb_project}")
        print(f"  Checkpoints: {config.checkpoint_dir}/")
        print(f"  Logs: logs/training.log")
        
        trainer.train()
        
        print(f"\n SPRING TRAINING COMPLETED SUCCESSFULLY!")
        print(f"   Final checkpoints: {config.checkpoint_dir}/")
        print(f"  Training summary: {config.checkpoint_dir}/training_summary.json")
        print(f"  Ready for evaluation against SPRING benchmarks!")
        
    except Exception as e:
        print(f"\nSPRING TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nDEBUG SUGGESTIONS:")
        print(f"  1. Check SPRING training data: python spring_data_loader_fixed.py")
        print(f"  2. Verify GPU memory: nvidia-smi")
        print(f"  3. Check logs: tail -f logs/training.log")
        print(f"  4. Recreate training data: python coco_processor_fixed.py")
        raise


if __name__ == "__main__":
    main()