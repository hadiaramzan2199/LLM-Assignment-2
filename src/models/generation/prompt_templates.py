"""Prompt templates for different LLMs."""

from typing import Dict, List, Optional


def get_prompt_template(model_name: str):
    """Get appropriate prompt template for model."""
    model_name_lower = model_name.lower()
    
    if "llama" in model_name_lower:
        return LlamaTemplate()
    elif "qwen" in model_name_lower:
        return QwenTemplate()
    elif "mistral" in model_name_lower:
        return MistralTemplate()
    else:
        return DefaultTemplate()


class BaseTemplate:
    """Base prompt template."""
    
    def format(self, question: str, passages: List[str], history: Optional[str] = None) -> str:
        raise NotImplementedError


class LlamaTemplate(BaseTemplate):
    """Template for Llama-3 instruct models."""
    
    def format(self, question: str, passages: List[str], history: Optional[str] = None) -> str:
        system_msg = """You are a helpful assistant that answers questions based ONLY on the provided passages. 
If the passages do not contain the answer, say "I cannot answer this question based on the provided passages."
Be concise and faithful to the source material."""
        
        # Format passages
        passages_text = ""
        for i, p in enumerate(passages, 1):
            passages_text += f"[{i}] {p}\n\n"
        
        # Build conversation
        if history:
            conversation = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>\n\n"
            conversation += f"<|start_header_id|>user<|end_header_id|>\n\n{history}<|eot_id|>\n\n"
            conversation += f"<|start_header_id|>assistant<|end_header_id|>\n\nI understand the conversation history.<|eot_id|>\n\n"
            conversation += f"<|start_header_id|>user<|end_header_id|>\n\nGiven these passages:\n\n{passages_text}\nAnswer this question: {question}<|eot_id|>\n\n"
            conversation += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            conversation = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>\n\n"
            conversation += f"<|start_header_id|>user<|end_header_id|>\n\nGiven these passages:\n\n{passages_text}\nAnswer this question: {question}<|eot_id|>\n\n"
            conversation += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return conversation


class QwenTemplate(BaseTemplate):
    """Template for Qwen-2.5 instruct models."""
    
    def format(self, question: str, passages: List[str], history: Optional[str] = None) -> str:
        system_msg = """You are a helpful assistant that answers questions based ONLY on the provided passages. 
If the passages do not contain the answer, say "I cannot answer this question based on the provided passages."
Be concise and faithful to the source material."""
        
        # Format passages
        passages_text = ""
        for i, p in enumerate(passages, 1):
            passages_text += f"Passage {i}: {p}\n\n"
        
        if history:
            prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            prompt += f"<|im_start|>user\n{history}\n\n{passages_text}\nQuestion: {question}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            prompt += f"<|im_start|>user\n{passages_text}\nQuestion: {question}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
        
        return prompt


class MistralTemplate(BaseTemplate):
    """Template for Mistral models."""
    
    def format(self, question: str, passages: List[str], history: Optional[str] = None) -> str:
        system_msg = "[INST] <<SYS>>\nYou are a helpful assistant that answers questions based ONLY on the provided passages. If the passages do not contain the answer, say 'I cannot answer this question based on the provided passages.' Be concise and faithful to the source material.\n<</SYS>>\n\n"
        
        # Format passages
        passages_text = ""
        for i, p in enumerate(passages, 1):
            passages_text += f"Passage {i}: {p}\n\n"
        
        if history:
            prompt = f"{system_msg}{history}\n\nGiven these passages:\n\n{passages_text}Answer this question: {question} [/INST]"
        else:
            prompt = f"{system_msg}Given these passages:\n\n{passages_text}Answer this question: {question} [/INST]"
        
        return prompt


class DefaultTemplate(BaseTemplate):
    """Default template for other models."""
    
    def format(self, question: str, passages: List[str], history: Optional[str] = None) -> str:
        system_msg = "You are a helpful assistant that answers questions based ONLY on the provided passages. If the passages do not contain the answer, say 'I cannot answer this question based on the provided passages.' Be concise and faithful to the source material."
        
        # Format passages
        passages_text = ""
        for i, p in enumerate(passages, 1):
            passages_text += f"Passage {i}: {p}\n\n"
        
        if history:
            prompt = f"{system_msg}\n\nConversation history:\n{history}\n\nGiven these passages:\n\n{passages_text}Question: {question}\n\nAnswer:"
        else:
            prompt = f"{system_msg}\n\nGiven these passages:\n\n{passages_text}Question: {question}\n\nAnswer:"
        
        return prompt