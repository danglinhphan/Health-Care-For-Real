#!/usr/bin/env python3
"""
Qwen LoRA Inference Engine
"""

import asyncio
import torch
import gc
import logging
from typing import Dict, List, Optional, Tuple, AsyncGenerator, Any
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TextStreamer,
    GenerationConfig
)
from peft import PeftModel
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class QwenInferenceEngine:
    """Inference engine for Qwen LoRA models"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-1.7B",
                 adapter_path: str = "./models/qwen_lora_adapters",
                 device: str = "auto",
                 torch_dtype: torch.dtype = torch.bfloat16):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        
    async def initialize(self):
        """Initialize the model and tokenizer"""
        logger.info("Initializing Qwen LoRA inference engine...")
        
        try:
            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            logger.info(f"Loading base model from {self.model_name}")
            # Check if flash attention is available
            try:
                import flash_attn
                attn_implementation = "flash_attention_2"
            except ImportError:
                attn_implementation = "eager"
                logger.info("Flash attention not available, using eager attention")
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
            )
            
            # Load LoRA adapters
            logger.info(f"Loading LoRA adapters from {self.adapter_path}")
            self.model = PeftModel.from_pretrained(
                base_model, 
                self.adapter_path,
                torch_dtype=self.torch_dtype
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            # Default generation config
            self.generation_config = GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                max_new_tokens=1024,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                repetition_penalty=1.1,
            )
            
            logger.info("Qwen LoRA inference engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize inference engine: {e}")
            raise
    
    def _prepare_messages(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to chat format"""
        formatted_messages = []
        
        # Balanced system prompt 
        system_prompt = """You are a helpful AI assistant with medical knowledge. Respond naturally to all questions. For casual conversations and greetings, be friendly and conversational. For medical questions, provide helpful information while recommending consulting healthcare professionals for diagnosis and treatment. Keep responses clear and informative."""
        
        formatted_messages.append({"role": "system", "content": system_prompt})
        
        # Only use the last few messages to avoid context contamination
        recent_messages = messages[-3:] if len(messages) > 3 else messages
        
        for msg in recent_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                formatted_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                formatted_messages.append({"role": "assistant", "content": content})
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def _clean_response(self, response_text: str) -> str:
        """Clean response from medical artifacts and unwanted content"""
        import re
        
        # Store original response for fallback
        original_response = response_text.strip()
        
        # Remove <think> tags and content
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        
        # Only remove medical patterns if they clearly exist (more specific matching)
        if '###Rationale:' in response_text and '###Answer:' in response_text:
            response_text = re.sub(r'###Rationale:.*?###Answer:\s*', '', response_text, flags=re.DOTALL)
        
        # Remove standalone ###Answer: only if followed by single letter options
        response_text = re.sub(r'###Answer:\s*[A-Z]\.?\s*$', '', response_text, flags=re.MULTILINE)
        
        # Remove medical diagnostic patterns - more comprehensive
        response_text = re.sub(r'\bOPTION [A-Z] IS CORRECT\.?\s*', '', response_text, flags=re.MULTILINE | re.IGNORECASE)
        response_text = re.sub(r'\bThe answer is [A-Z]\.?\s*', '', response_text, flags=re.MULTILINE | re.IGNORECASE)
        response_text = re.sub(r'\bAnswer:\s*[A-Z]\.?\s*', '', response_text, flags=re.MULTILINE | re.IGNORECASE)
        response_text = re.sub(r'\b[A-Z]\.\s*is\s*(the\s*)?(correct|right)\s*(answer|option)\.?\s*', '', response_text, flags=re.MULTILINE | re.IGNORECASE)
        response_text = re.sub(r'^\s*[A-Z]\.?\s*$', '', response_text, flags=re.MULTILINE)
        response_text = re.sub(r'###\s*Answer\s*:?\s*[A-Z]\.?\s*', '', response_text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Clean up whitespace
        response_text = re.sub(r'\s+', ' ', response_text).strip()

        # If cleaning removed everything, use original response
        if not response_text and original_response:
            response_text = original_response
        
        # Only provide default if completely empty
        if not response_text:
            response_text = "I'm here to help! What would you like to know?"
        
        return response_text
    
    def _generate_sync(self, 
                      messages: List[Dict[str, str]],
                      temperature: float = 0.7,
                      top_p: float = 0.9,
                      top_k: int = 40,
                      max_tokens: int = 1024,
                      stop_sequences: Optional[List[str]] = None) -> Tuple[str, Dict[str, int]]:
        """Synchronous generation"""
        with self._lock:
            try:
                # Clear GPU cache to avoid contamination
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Prepare input
                prompt = self._prepare_messages(messages)
                prompt_tokens = self._count_tokens(prompt)
                
                # Tokenize
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(self.model.device)
                
                # Update generation config
                gen_config = GenerationConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    repetition_penalty=1.1,
                )
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=gen_config,
                        use_cache=True
                    )
                
                # Decode response
                response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                response_text = self.tokenizer.decode(
                    response_tokens, 
                    skip_special_tokens=True
                ).strip()
                
                # Clean response from medical artifacts
                response_text = self._clean_response(response_text)
                
                # Apply stop sequences
                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if stop_seq in response_text:
                            response_text = response_text.split(stop_seq)[0]
                
                completion_tokens = len(response_tokens)
                total_tokens = prompt_tokens + completion_tokens
                
                token_counts = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
                
                return response_text, token_counts
                
            except Exception as e:
                logger.error(f"Generation error: {e}")
                raise
            finally:
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    async def generate(self, 
                      messages: List[Dict[str, str]],
                      temperature: float = 0.7,
                      top_p: float = 0.9,
                      top_k: int = 40,
                      max_tokens: int = 1024,
                      stop_sequences: Optional[List[str]] = None) -> Tuple[str, Dict[str, int]]:
        """Asynchronous generation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._generate_sync,
            messages,
            temperature,
            top_p,
            top_k,
            max_tokens,
            stop_sequences
        )
    
    def _stream_generate_sync(self, 
                             messages: List[Dict[str, str]],
                             temperature: float = 0.7,
                             top_p: float = 0.9,
                             top_k: int = 40,
                             max_tokens: int = 1024,
                             stop_sequences: Optional[List[str]] = None):
        """Synchronous streaming generation"""
        with self._lock:
            try:
                # Prepare input
                prompt = self._prepare_messages(messages)
                prompt_tokens = self._count_tokens(prompt)
                
                # Tokenize
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(self.model.device)
                
                # Update generation config
                gen_config = GenerationConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    repetition_penalty=1.1,
                )
                
                # Setup streaming
                response_text = ""
                input_length = inputs['input_ids'].shape[1]
                
                # Generate with streaming
                with torch.no_grad():
                    for output in self.model.generate(
                        **inputs,
                        generation_config=gen_config,
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_scores=True,
                        streamer=None  # We'll handle streaming manually
                    ):
                        # Decode new tokens
                        new_tokens = output[input_length:]
                        chunk_text = self.tokenizer.decode(
                            new_tokens, 
                            skip_special_tokens=True
                        )
                        
                        # Check for stop sequences
                        should_stop = False
                        if stop_sequences:
                            for stop_seq in stop_sequences:
                                if stop_seq in chunk_text:
                                    chunk_text = chunk_text.split(stop_seq)[0]
                                    should_stop = True
                                    break
                        
                        response_text = chunk_text
                        
                        # Yield chunk
                        yield chunk_text, should_stop, None
                        
                        if should_stop:
                            break
                
                # Final token counts
                completion_tokens = self._count_tokens(response_text)
                total_tokens = prompt_tokens + completion_tokens
                
                token_counts = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
                
                # Yield final chunk with token counts
                yield response_text, True, token_counts
                
            except Exception as e:
                logger.error(f"Streaming generation error: {e}")
                yield f"Error: {str(e)}", True, None
            finally:
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    async def stream_generate(self, 
                             messages: List[Dict[str, str]],
                             temperature: float = 0.7,
                             top_p: float = 0.9,
                             top_k: int = 40,
                             max_tokens: int = 1024,
                             stop_sequences: Optional[List[str]] = None) -> AsyncGenerator[Tuple[str, bool, Optional[Dict[str, int]]], None]:
        """Asynchronous streaming generation"""
        loop = asyncio.get_event_loop()
        
        # Create a queue for communication between threads
        queue = asyncio.Queue()
        
        def run_stream():
            try:
                for chunk_text, is_final, token_counts in self._stream_generate_sync(
                    messages, temperature, top_p, top_k, max_tokens, stop_sequences
                ):
                    asyncio.run_coroutine_threadsafe(
                        queue.put((chunk_text, is_final, token_counts)), 
                        loop
                    )
                
                # Signal completion
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)
                
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    queue.put((f"Error: {str(e)}", True, None)), 
                    loop
                )
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)
        
        # Start streaming in executor
        await loop.run_in_executor(None, lambda: threading.Thread(target=run_stream).start())
        
        # Yield results
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up inference engine...")
        
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Garbage collection
        gc.collect()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Inference engine cleanup completed")
    
    def is_ready(self) -> bool:
        """Check if the inference engine is ready"""
        return self.model is not None and self.tokenizer is not None