"""LLM-based generation for Task B."""

import json
import torch
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

from src.models.generation.prompt_templates import get_prompt_template


@dataclass
class GenerationConfig:
    """Configuration for generation."""
    model_name: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    do_sample: bool = True
    num_beams: int = 1
    repetition_penalty: float = 1.1
    use_history: bool = True


class LLMGenerator:
    """Generator using HuggingFace LLMs."""
    
    def __init__(self,
                 config: GenerationConfig,
                 quantization: str = "4bit",
                 device_map: str = "auto"):
        """
        Args:
            config: Generation configuration
            quantization: "4bit", "8bit", or None
            device_map: Device mapping for model
        """
        self.config = config
        self.device_map = device_map
        
        # Setup quantization
        self.quantization_config = None
        if quantization == "4bit":
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == "8bit":
            self.quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load model and tokenizer
        self._load_model()
        
        # Get prompt template
        self.prompt_template = get_prompt_template(config.model_name)
    
    def _load_model(self):
        """Load model and tokenizer."""
        print(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            padding_side="left",
            trust_remote_code=True
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=self.quantization_config,
            device_map=self.device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        self.model.eval()
        print("Model loaded successfully")
    
    def format_prompt(self,
                      question: str,
                      passages: List[Dict[str, str]],
                      conversation_history: Optional[str] = None) -> str:
        """Format prompt for generation.
        
        Args:
            question: Current user question
            passages: List of relevant passages
            conversation_history: Optional conversation history
            
        Returns:
            Formatted prompt string
        """
        # Extract passage texts
        passage_texts = [p["text"] if isinstance(p, dict) else p for p in passages]
        
        # Use template
        return self.prompt_template.format(
            question=question,
            passages=passage_texts,
            history=conversation_history
        )
    
    def generate(self,
                 question: str,
                 passages: List[Dict[str, str]],
                 conversation_history: Optional[str] = None,
                 **kwargs) -> str:
        """Generate response for a single example.
        
        Args:
            question: Current user question
            passages: List of relevant passages
            conversation_history: Optional conversation history
            **kwargs: Override generation config
            
        Returns:
            Generated response text
        """
        # Format prompt
        prompt = self.format_prompt(question, passages, conversation_history)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096 - self.config.max_new_tokens
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
                top_k=kwargs.get("top_k", self.config.top_k),
                do_sample=kwargs.get("do_sample", self.config.do_sample),
                num_beams=kwargs.get("num_beams", self.config.num_beams),
                repetition_penalty=kwargs.get("repetition_penalty", self.config.repetition_penalty),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def generate_batch(self,
                       examples: List[Any],
                       passage_lookup: Dict[str, str],
                       use_history: bool = True) -> Dict[str, str]:
        """Generate for multiple examples.
        
        Args:
            examples: List of MTRAGExample objects
            passage_lookup: Mapping from passage ID to text
            use_history: Whether to include conversation history
            
        Returns:
            Dictionary mapping task_id to response
        """
        results = {}
        
        for ex in examples:
            # Get passage texts from IDs
            passages = []
            if ex.relevant_passages:
                for pid in ex.relevant_passages:
                    if pid in passage_lookup:
                        passages.append({"text": passage_lookup[pid]})
            
            # Format history if needed
            history = None
            if use_history:
                history_texts = []
                for turn in ex.conversation[:-1]:
                    role = "User" if turn["role"] == "user" else "Assistant"
                    history_texts.append(f"{role}: {turn['text']}")
                history = "\n".join(history_texts)
            
            # Generate
            response = self.generate(
                question=ex.final_question,
                passages=passages,
                conversation_history=history
            )
            
            results[ex.task_id] = response
        
        return results