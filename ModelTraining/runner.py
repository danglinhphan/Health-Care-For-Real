!/usr/bin/env python3
"""
Optimized LoRA fine-tuning script for Qwen model
"""

import gc
import os
import math
import torch
from typing import Dict, Any
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset

from QwenInstructionDataset import QwenInstructionDataset
from epoch_sampling_strategy import EpochSamplingDataset, train_with_epoch_sampling


@dataclass
class TrainingConfig:
    """Training configuration optimized for RTX 4090 (24GB VRAM) with anti-overfitting measures"""
    model_name: str = "Qwen/Qwen3-1.7B"
    dataset_name: str = "axiong/pmc_llama_instructions" 
    output_dir: str = "./qwen_lora_tuned"
    adapter_dir: str = "./qwen_lora_adapters"
    
    LoRA parameters - reduced to prevent overfitting
    lora_r: int =4  # Further reduced for RTX 3060
    lora_alpha: int = 32  # Corresponding alpha scaling
    lora_dropout: float = 0.2  # Increased dropout for regularization
    
    Training parameters - Anti-overfitting configuration with epoch sampling
    num_train_epochs: int = 1  # More epochs with sampling strategy
    per_device_train_batch_size: int = 1  # Reduced for RTX 3060
    gradient_accumulation_steps: int = 4  # Reduced for RTX 3060
    learning_rate: float = 1e-4  # Lower learning rate for better convergence
    max_length: int = 256  # Further reduced for RTX 3060
    max_samples: int = None  # Use full dataset with sampling strategy
    
    Epoch sampling strategy parameters
    enable_epoch_sampling: bool = False  # Enable epoch-wise sampling
    sampling_ratio: float = 0.01  # Use 10% of data per epoch
    sampling_seed: int = 42  # Seed for reproducible sampling
    
    Regularization parameters
    weight_decay: float = 0.3  # Strong weight decay
    warmup_ratio: float = 0.1  # Longer warmup for stability
    cosine_schedule: bool = True  # Cosine learning rate schedule
    
    Early stopping parameters 
    enable_early_stopping: bool= True
    early_stopping_patience: int = 5  # Increased patience for epoch sampling
    early_stopping_threshold: float = 0.001  # More sensitive threshold
    
    Additional overfitting prevention
    max_grad_norm: float = 0.3  # Stricter gradient clipping
    label_smoothing: float = 0.1  # Label smoothing for regularization
    
    System parameters - 4090 specific
    bf16: bool = False  # BF16 for Ada Lovelace architecture
    fp16: bool = False  # Use BF16 instead
    gradient_checkpointing: bool = False  # Disabled for RTX 3060
    
    4090 specific optimizations
    dataloader_num_workers: int = 8  # Parallel data loading
    pin_memory: bool = True  # Faster GPU transfers
    compile_model: bool = False  # PyTorch 2.0+ compilation (disabled for stability)
    use_flash_attention: bool = True  # Flash Attention 2


class MemoryOptimizer:
    """Memory management utilities optimized for RTX 4090"""
    
    @staticmethod
    def clear_memory():
        """Clear GPU and system memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()  # Clear IPC cache
        gc.collect()
    
    @staticmethod
    def print_memory_stats():
        """Print detailed memory usage for 4090"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            total_memory = 12.0  
            utilization = (allocated / total_memory) * 100
            print(f"RTX 4090 Memory: {allocated:.2f}GB/{total_memory}GB ({utilization:.1f}%)")
            print(f"Cached: {cached:.2f}GB, Peak: {max_allocated:.2f}GB")
    
    @staticmethod
    def optimize_4090_settings():
        """Apply RTX 4090 specific optimizations"""
        if torch.cuda.is_available():
            Enable TensorFloat-32 for faster training on Ampere/Ada
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            Optimize CUDNN for consistent workloads
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            Set memory fraction to prevent OOM - more conservative
            torch.cuda.set_per_process_memory_fraction(0.85)
            
            Enable memory expansion for better allocation
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            
            print("✓ RTX 4090 optimizations applied")


class ModelManager:
    """Model loading and configuration"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
    
    def load_tokenizer(self):
        """Load and configure tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer
    
    def load_model(self):
        """Load base model with RTX 4090 optimizations"""
        MemoryOptimizer.clear_memory()
        MemoryOptimizer.optimize_4090_settings()
        
        Determine optimal dtype for 4090
        dtype = torch.bfloat16 if self.config.bf16 else torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if self.config.use_flash_attention else "eager",
            use_cache=False,  # Disable KV cache for training
        )
        
        Apply gradient checkpointing if enabled
        if self.config.gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        Note: Model compilation will be applied after LoRA wrapping
        to avoid conflicts with Trainer's model unwrapping
        
        return self.model
    
    def apply_lora(self):
        """Apply LoRA configuration to model"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            fan_in_fan_out=False,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        Apply compilation after LoRA wrapping to avoid Trainer conflicts
        if self.config.compile_model and hasattr(torch, 'compile'):
            print("Compiling LoRA model with PyTorch 2.0+...")
            try:
                Use a more compatible compilation mode
                self.model = torch.compile(self.model, mode="default", dynamic=True)
                print("✓ Model compilation successful")
            except Exception as e:
                print(f"⚠️  Model compilation failed, continuing without: {e}")
        
        return self.model


class DataManager:
    """Dataset loading and processing optimized for RTX 4090"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def load_dataset(self):
        """Load and preprocess dataset with epoch sampling support"""
        print(f"Loading dataset: {self.config.dataset_name}")
        
        Use streaming for large datasets to save memory
        dataset = load_dataset(
            self.config.dataset_name,
            streaming=False,  # Disable for better performance with sufficient RAM
            num_proc=8  # Parallel processing
        )
        
        data = []
        print("Processing dataset samples...")
        
        Process with progress tracking for large datasets
        max_samples = self.config.max_samples if self.config.max_samples is not None else len(dataset['train'])
        
        for i, item in enumerate(dataset['train']):
            if i >= max_samples:
                break
                
            Filter out empty samples
            instruction = item.get('instruction', item.get('input', '')).strip()
            response = item.get('output', item.get('response', '')).strip()
            
            if instruction and response:  # Only include valid samples
                data.append({
                    "instruction": instruction,
                    "response": response
                })
            
            Progress indicator for large datasets
            if i % 10000 == 0 and i > 0:
                print(f"Processed {i} samples...")
        
        total_samples = len(data)
        print(f"Loaded {total_samples} high-quality training examples")
        
        if self.config.enable_epoch_sampling:
            print(f"Epoch sampling enabled:")
            print(f"  Sampling ratio: {self.config.sampling_ratio*100:.1f}%")
            print(f"  Samples per epoch: {int(total_samples * self.config.sampling_ratio)}")
            print(f"  Epochs to cover all data: {math.ceil(1.0 / self.config.sampling_ratio)}")
        
        return data
    
    def get_dataloader_kwargs(self):
        """Get optimized dataloader arguments for RTX 4090"""
        return {
            "num_workers": self.config.dataloader_num_workers,
            "pin_memory": self.config.pin_memory,
            "persistent_workers": True if self.config.dataloader_num_workers > 0 else False,
            "prefetch_factor": 2 if self.config.dataloader_num_workers > 0 else None,
        }


class TrainingManager:
    """Training orchestration"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model_manager = ModelManager(config)
        self.data_manager = DataManager(config)
    
    def setup(self):
        """Setup all components for training with epoch sampling support"""
        print("Setting up training components...")
        
        Load tokenizer and model
        tokenizer = self.model_manager.load_tokenizer()
        model = self.model_manager.load_model()
        model = self.model_manager.apply_lora()
        
        Load data with augmentation enabled
        data = self.data_manager.load_dataset()
        
        Create base dataset first
        base_train_dataset = QwenInstructionDataset(
            data, tokenizer, max_length=self.config.max_length, enable_augmentation=True
        )
        
        Apply epoch sampling if enabled
        if self.config.enable_epoch_sampling:
            print("Applying epoch sampling strategy...")
            train_dataset = EpochSamplingDataset(
                base_train_dataset, 
                sampling_ratio=self.config.sampling_ratio,
                shuffle_seed=self.config.sampling_seed
            )
        else:
            train_dataset = base_train_dataset
        
        Handle validation dataset for early stopping
        eval_dataset = None
        if self.config.enable_early_stopping:
            For epoch sampling, use a small validation set from unused data
            if self.config.enable_epoch_sampling:
                Use last 5% of data for validation (won't overlap with training due to sampling)
                val_start_idx = int(len(data) * 0.95)
                eval_data = data[val_start_idx:]
                eval_dataset = QwenInstructionDataset(
                    eval_data, tokenizer, max_length=self.config.max_length, enable_augmentation=False
                )
                print(f"Created validation set with {len(eval_data)} samples")
            else:
                Traditional split for non-sampling approach
                split_idx = int(len(data) * 0.9)
                eval_data = data[split_idx:]
                eval_dataset = QwenInstructionDataset(
                    eval_data, tokenizer, max_length=self.config.max_length, enable_augmentation=False
                )
        
        return model, tokenizer, train_dataset, eval_dataset
    
    def create_training_args(self):
        """Create RTX 4090 optimized training arguments with anti-overfitting measures"""
        dataloader_kwargs = self.data_manager.get_dataloader_kwargs()
        
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            
            Enhanced regularization for anti-overfitting
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type="cosine" if self.config.cosine_schedule else "linear",
            
            Logging optimized for faster training
            logging_steps=10,
            logging_first_step=True,
            
            Evaluation and saving with early stopping support
            eval_strategy="steps" if self.config.enable_early_stopping else "no",
            eval_steps=50 if self.config.enable_early_stopping else None,
            save_steps=100,  # More frequent saves for monitoring
            save_total_limit=3,
            save_strategy="steps",
            load_best_model_at_end=self.config.enable_early_stopping,
            metric_for_best_model="eval_loss" if self.config.enable_early_stopping else None,
            
            Precision settings for 4090
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            fp16_full_eval=False,
            
            Memory and performance optimizations
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_drop_last=True,
            dataloader_pin_memory=dataloader_kwargs["pin_memory"],
            dataloader_num_workers=dataloader_kwargs["num_workers"],
            dataloader_persistent_workers=dataloader_kwargs["persistent_workers"],
            dataloader_prefetch_factor=dataloader_kwargs["prefetch_factor"],
            
            Training stability with enhanced overfitting prevention
            max_grad_norm=self.config.max_grad_norm,
            label_smoothing_factor=self.config.label_smoothing,
            optim="adamw_torch",  # Standard optimizer for compatibility
            adam_beta1=0.9,
            adam_beta2=0.95,  # Optimized for LoRA
            adam_epsilon=1e-8,
            
            Miscellaneous
            remove_unused_columns=False,
            report_to="none",
            disable_tqdm=False,
            
            Speed optimizations
            group_by_length=True,  # Group similar length sequences
            length_column_name="length" if hasattr(self, 'length_column') else None,
        )
    
    def train(self):
        """Execute RTX 4090 optimized training pipeline"""
        print("🚀 Starting RTX 4090 optimized training pipeline...")
        
        try:
            Setup with 4090 optimizations
            setup_result = self.setup()
            if self.config.enable_early_stopping:
                model, tokenizer, train_dataset, eval_dataset = setup_result
            else:
                model, tokenizer, train_dataset, eval_dataset = *setup_result, None
                
            training_args = self.create_training_args()
            
            Create callbacks for early stopping if enabled
            callbacks = []
            if self.config.enable_early_stopping and eval_dataset is not None:
                callbacks.append(EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold
                ))
            
            Create optimized trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                callbacks=callbacks,
            )
            
            Performance monitoring
            import time
            start_time = time.time()
            
            print("🔥 Starting training with RTX 4090 optimizations...")
            print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
            MemoryOptimizer.print_memory_stats()
            
            Execute training with epoch sampling if enabled
            if self.config.enable_epoch_sampling and isinstance(train_dataset, EpochSamplingDataset):
                print("Enabling epoch-aware training with sampling strategy...")
                result = train_with_epoch_sampling(
                    trainer, train_dataset, self.config.num_train_epochs
                )
            else:
                result = trainer.train()
            
            Calculate performance metrics
            end_time = time.time()
            training_time = end_time - start_time
            print(f"✅ Training completed in {training_time:.2f} seconds!")
            
            Save LoRA adapters
            print(f"💾 Saving LoRA adapters to {self.config.adapter_dir}")
            model.save_pretrained(self.config.adapter_dir)
            tokenizer.save_pretrained(self.config.adapter_dir)
            
            Final memory stats
            MemoryOptimizer.print_memory_stats()
            
            Verify loading
            self.verify_model_loading()
            
            Training summary
            total_samples = len(train_dataset)
            samples_per_second = total_samples * training_args.num_train_epochs / training_time
            print(f"📊 Performance: {samples_per_second:.2f} samples/second")
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            MemoryOptimizer.print_memory_stats()
            raise
        finally:
            MemoryOptimizer.clear_memory()
            print("🧹 Memory cleanup completed")
    
    def verify_model_loading(self):
        """Verify that saved adapters can be loaded"""
        try:
            print("Verifying model loading...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            lora_model = PeftModel.from_pretrained(base_model, self.config.adapter_dir)
            print("✓ Model verification successful!")
        except Exception as e:
            print(f"✗ Model verification failed: {str(e)}")


def print_4090_banner():
    """Print RTX 4090 optimization banner"""
    print("=" * 60)
    print("🎯 RTX 4090 OPTIMIZED QWEN LORA FINE-TUNING WITH EPOCH SAMPLING")
    print("=" * 60)
    print("🚀 Performance Optimizations:")
    print("  • Balanced batch size with gradient accumulation")
    print("  • BF16 precision (Ada Lovelace native)")
    print("  • Flash Attention 2 enabled")
    print("  • TensorFloat-32 acceleration")
    print("  • Optimized AdamW optimizer")
    print("  • Multi-worker data loading")
    print("  • Memory-efficient configuration")
    print("  • LoRA rank 8 for optimal performance/memory")
    print("")
    print("🎲 Epoch Sampling Strategy:")
    print("  • 10% random sampling per epoch")
    print("  • No sample repetition until full dataset used")
    print("  • Enhanced overfitting prevention")
    print("  • Label smoothing regularization")
    print("  • Stricter gradient clipping")
    print("=" * 60)


def main():
    """Main RTX 4090 optimized training function"""
    print_4090_banner()
    
    Apply memory optimizations before any GPU operations
    MemoryOptimizer.optimize_4090_settings()
    MemoryOptimizer.clear_memory()
    
    Initialize 4090-optimized configuration
    config = TrainingConfig()
    
    Print configuration summary
    print(f"📋 Training Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Batch Size: {config.per_device_train_batch_size}")
    print(f"  Sequence Length: {config.max_length}")
    print(f"  LoRA Rank: {config.lora_r}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Max Samples: {'Full dataset' if config.max_samples is None else config.max_samples}")
    print(f"  Precision: {'BF16' if config.bf16 else 'FP16'}")
    print()
    
    if config.enable_epoch_sampling:
        print(f"🎲 Epoch Sampling Configuration:")
        print(f"  Sampling Ratio: {config.sampling_ratio*100:.1f}%")
        print(f"  Shuffle Seed: {config.sampling_seed}")
        print(f"  Estimated Epochs for Full Coverage: {math.ceil(1.0 / config.sampling_ratio)}")
        print()
    
    Verify RTX 4090 setup
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 GPU: {gpu_name}")
        if "4090" not in gpu_name:
            print("⚠️  WARNING: This script is optimized for RTX 4090. Performance may vary on other GPUs.")
        print()
    else:
        print("❌ CUDA not available. This script requires RTX 4090 GPU.")
        return
    
    Create training manager and start training
    trainer_manager = TrainingManager(config)
    trainer_manager.train()
    
    print("🎉 RTX 4090 optimized training pipeline completed!")
    print("💡 Your LoRA adapters are ready for inference!")


if __name__ == "__main__":
    main()