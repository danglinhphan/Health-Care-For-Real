#!/usr/bin/env python3
"""
Simple inference testing script for LoRA fine-tuned Qwen model
ASCII-only version for Windows compatibility
"""

import torch
import gc
import time
import os
import sys
from typing import Dict, Any, List

# Add error handling for import issues
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("[OK] Transformers imported successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import transformers: {e}")
    sys.exit(1)

try:
    from peft import PeftModel
    print("[OK] PEFT imported successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import PEFT: {e}")
    print("Please install PEFT: pip install peft")
    sys.exit(1)


class QwenInferenceTest:
    """Simple inference engine for LoRA fine-tuned Qwen model"""
    
    def __init__(self, base_model_name: str = "Qwen/Qwen3-1.7B", adapter_path: str = "./qwen_lora_adapters"):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[INFO] Initializing Qwen Inference on {self.device}")
        self._setup_environment()
        self._load_model()
    
    def _setup_environment(self):
        """Setup environment and optimizations"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("[OK] TensorFloat-32 optimizations enabled")
            except:
                print("[WARN] TF32 optimizations not available")
        else:
            print("[WARN] CUDA not available, using CPU")
    
    def _load_model(self):
        """Load the base model and LoRA adapters"""
        try:
            print(f"[INFO] Loading tokenizer: {self.base_model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("[OK] Pad token set to EOS token")
            
            print(f"[INFO] Loading base model: {self.base_model_name}")
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            print(f"[INFO] Loading LoRA adapters from: {self.adapter_path}")
            
            # Check if adapter path exists
            if not os.path.exists(self.adapter_path):
                raise FileNotFoundError(f"LoRA adapter path not found: {self.adapter_path}")
            
            # Load LoRA adapters
            self.model = PeftModel.from_pretrained(
                base_model,
                self.adapter_path,
                torch_dtype=torch.float16
            )
            
            self.model.eval()
            
            print("[SUCCESS] Model loaded successfully!")
            self._print_memory_stats()
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {str(e)}")
            print(f"[DEBUG] Error type: {type(e).__name__}")
            raise
    
    def _print_memory_stats(self):
        """Print current GPU memory usage"""
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                print(f"[MEMORY] GPU: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
            except:
                print("[WARN] Could not get memory stats")
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Dict[str, Any]:
        """Generate text for a given prompt"""
        
        print(f"[GENERATE] Processing: {prompt[:50]}...")
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate response
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(prompt):].strip()
            
            generation_time = time.time() - start_time
            tokens_generated = outputs[0].shape[0] - inputs.shape[1]
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            print(f"[RESPONSE] Generated {tokens_generated} tokens in {generation_time:.2f} seconds")

            return {
                "prompt": prompt,
                "response": response,
                "generation_time": generation_time,
                "tokens_generated": tokens_generated,
                "tokens_per_second": tokens_per_second
            }
            
        except Exception as e:
            print(f"[ERROR] Generation failed: {str(e)}")
            return {
                "prompt": prompt,
                "response": f"Error: {str(e)}",
                "generation_time": 0,
                "tokens_generated": 0,
                "tokens_per_second": 0
            }
    
    def run_basic_tests(self):
        """Run basic test prompts"""
        test_prompts = [
            "What is artificial intelligence?",
            "Explain machine learning in simple terms.",
            "How do neural networks learn?",
            "What is the difference between AI and ML?",
            "Describe deep learning."

        ]
        
        print(f"\n[TEST] Running {len(test_prompts)} basic tests...")
        print("=" * 60)
        
        results = []
        total_time = 0
        total_tokens = 0
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n[TEST {i}/{len(test_prompts)}] {prompt}")
            result = self.generate_text(prompt, max_length=128)
            results.append(result)
            
            print(f"[RESPONSE] {result['response'][:100]}...")
            print(f"[SPEED] {result['tokens_per_second']:.1f} tokens/second")
            
            total_time += result['generation_time']
            total_tokens += result['tokens_generated']
        
        # Summary
        avg_speed = total_tokens / total_time if total_time > 0 else 0
        print(f"\n[SUMMARY] Test Results:")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Average speed: {avg_speed:.1f} tokens/second")
        print(f"  Tests passed: {len([r for r in results if 'Error:' not in r['response']])}/{len(results)}")
        
        return results
    
    def interactive_test(self):
        """Interactive testing mode"""
        print("\n[INTERACTIVE] Mode started (type 'quit' to exit)")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("[INFO] Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                result = self.generate_text(user_input)
                print(f"Assistant: {result['response']}")
                print(f"[PERF] {result['tokens_generated']} tokens, {result['tokens_per_second']:.1f} tok/s")
                
            except KeyboardInterrupt:
                print("\n[INFO] Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"[ERROR] {str(e)}")


def main():
    """Main function"""
    print("QWEN LORA INFERENCE TESTER")
    print("=" * 40)
    
    try:
        # Check if adapter directory exists before initializing
        adapter_path = "./qwen_lora_adapters"
        if not os.path.exists(adapter_path):
            print(f"[ERROR] LoRA adapter directory not found: {adapter_path}")
            print("[INFO] Please run the training script first to create adapters")
            return
        
        # Initialize inference
        inference = QwenInferenceTest()
        
        print("\nTest Options:")
        print("1. Run basic test suite")
        print("2. Interactive mode")  
        print("3. Single prompt test")
        print("4. Exit")
        
        while True:
            choice = input("\nSelect (1-4): ").strip()
            
            if choice == "1":
                inference.run_basic_tests()
                
            elif choice == "2":
                inference.interactive_test()
                
            elif choice == "3":
                prompt = input("Enter prompt: ").strip()
                if prompt:
                    result = inference.generate_text(prompt)
                    print(f"\nResponse: {result['response']}")
                    print(f"Speed: {result['tokens_per_second']:.1f} tokens/second")
                
            elif choice == "4":
                print("[INFO] Goodbye!")
                break
                
            else:
                print("[ERROR] Invalid choice. Please select 1-4.")
    
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure LoRA adapters exist in ./qwen_lora_adapters/")
        print("2. Check if training completed successfully")
        print("3. Verify PyTorch/CUDA installation")


if __name__ == "__main__":
    main()