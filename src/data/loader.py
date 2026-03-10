"""MTRAG dataset loader with proper formatting for multi-turn conversations."""

import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader


@dataclass
class MTRAGExample:
    """Single example from MTRAG dataset."""
    task_id: str
    conversation: List[Dict[str, str]]  # List of {"role": "user/agent", "text": str}
    final_question: str
    corpus: List[Dict[str, str]]  # List of {"doc_id": str, "text": str}
    relevant_passages: Optional[List[str]] = None  # Passage IDs for Task B
    metadata: Optional[Dict[str, Any]] = None
    
    # Metadata that is NOT used during inference
    question_type: Optional[str] = None  # factoid, explanatory, etc. (hidden)
    answerability: Optional[bool] = None  # True if answerable (hidden)
    multi_turn_type: Optional[str] = None  # follow-up, clarification (hidden)
    domain: Optional[str] = None  # ClapNQ, Govt, etc. (provided)


class MTRAGDataset:
    """Loader for MTRAG dataset with conversation formatting."""
    
    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.examples = []
        self._load_data()
    
    def _load_data(self):
        """Load dataset from JSON files."""
        file_path = self.data_dir / f"{self.split}.json"
        if not file_path.exists():
            # Try alternative naming
            file_path = self.data_dir / f"{self.split}_data.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
        
        # Parse each example
        for item in raw_data:
            # Extract conversation turns
            conversation = []
            for turn in item.get("conversation", []):
                role = turn.get("role", "user")  # user or agent
                text = turn.get("text", "")
                conversation.append({"role": role, "text": text})
            
            # Final question is the last user turn
            final_question = conversation[-1]["text"] if conversation else ""
            
            # Process corpus passages
            corpus = []
            for doc in item.get("corpus", []):
                if isinstance(doc, dict):
                    corpus.append({
                        "doc_id": doc.get("id", doc.get("doc_id", str(len(corpus)))),
                        "text": doc.get("text", doc.get("content", ""))
                    })
                elif isinstance(doc, str):
                    corpus.append({"doc_id": str(len(corpus)), "text": doc})
            
            # Get relevant passages for Task B (if available)
            relevant = item.get("relevant_passages", [])
            if relevant and isinstance(relevant[0], dict):
                relevant = [r.get("id", r.get("doc_id", "")) for r in relevant]
            
            # Metadata (some hidden during evaluation)
            metadata = {
                "domain": item.get("domain", "unknown"),
                "question_type": item.get("question_type", None),
                "answerability": item.get("answerable", None),
                "multi_turn_type": item.get("multi_turn_type", None)
            }
            
            example = MTRAGExample(
                task_id=item.get("task_id", item.get("id", str(len(self.examples)))),
                conversation=conversation,
                final_question=final_question,
                corpus=corpus,
                relevant_passages=relevant,
                metadata=metadata,
                question_type=metadata["question_type"],
                answerability=metadata["answerability"],
                multi_turn_type=metadata["multi_turn_type"],
                domain=metadata["domain"]
            )
            self.examples.append(example)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def to_hf_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset format."""
        data = {
            "task_id": [ex.task_id for ex in self.examples],
            "conversation": [ex.conversation for ex in self.examples],
            "final_question": [ex.final_question for ex in self.examples],
            "corpus": [ex.corpus for ex in self.examples],
            "relevant_passages": [ex.relevant_passages for ex in self.examples],
            "domain": [ex.domain for ex in self.examples],
            "question_type": [ex.question_type for ex in self.examples],
            "answerability": [ex.answerability for ex in self.examples],
            "multi_turn_type": [ex.multi_turn_type for ex in self.examples],
        }
        return Dataset.from_dict(data)
    
    def filter_by_domain(self, domain: str) -> "MTRAGDataset":
        """Filter examples by domain."""
        filtered = MTRAGDataset.__new__(MTRAGDataset)
        filtered.data_dir = self.data_dir
        filtered.split = self.split
        filtered.examples = [ex for ex in self.examples if ex.domain == domain]
        return filtered
    
    def get_conversation_text(self, idx: int, include_history: bool = True) -> str:
        """Format conversation as text for prompting."""
        example = self.examples[idx]
        
        if not include_history:
            return example.final_question
        
        lines = []
        for turn in example.conversation[:-1]:  # Exclude final question
            role = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['text']}")
        
        # Add final question
        lines.append(f"User: {example.final_question}")
        
        return "\n".join(lines)