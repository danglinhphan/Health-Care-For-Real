
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
from tqdm import tqdm
from datasets import load_dataset

class QwenInstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048, enable_augmentation=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_augmentation = enable_augmentation
        
        # Data augmentation templates for response diversity
        self.response_starters = [
            "",  # Original
            "Dựa trên câu hỏi của bạn, ",
            "Để trả lời câu hỏi này, ",
            "Tôi có thể giải thích như sau: ",
            "Theo hiểu biết của tôi, ",
            "Đây là câu trả lời: ",
        ]
        
        self.instruction_variations = [
            "",  # Original
            "Bạn có thể ",
            "Hãy giúp tôi ",
            "Xin hãy ",
            "Tôi muốn biết ",
            "Làm thế nào để ",
        ]

    def __len__(self):
        return len(self.data)

    def augment_text(self, instruction, response):
        """Apply data augmentation to increase diversity"""
        if not self.enable_augmentation or random.random() > 0.3:
            return instruction, response
            
        # Randomly augment instruction
        if random.random() < 0.4:
            starter = random.choice(self.instruction_variations[1:])
            if not instruction.lower().startswith(starter.lower()):
                instruction = starter + instruction.lower()
        
        # Randomly augment response
        if random.random() < 0.4:
            starter = random.choice(self.response_starters[1:])
            response = starter + response
            
        return instruction, response

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Apply augmentation
        instruction, response = self.augment_text(
            item["instruction"], 
            item["response"]
        )

        # Diverse message formats to avoid overfitting
        message_formats = [
            # Standard format
            [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response}
            ],
            # With system message variation
            [
                {"role": "system", "content": "Bạn là một trợ lý AI hữu ích."},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response}
            ],
            # Conversational format
            [
                {"role": "user", "content": f"Câu hỏi: {instruction}"},
                {"role": "assistant", "content": f"Trả lời: {response}"}
            ]
        ]
        
        # Randomly select format
        messages = random.choice(message_formats) if self.enable_augmentation else message_formats[0]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten()
        }