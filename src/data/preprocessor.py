"""Preprocessing utilities for MTRAG dataset."""

import hashlib
import json
import os
import re
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from src.data.loader import MTRAGDataset, MTRAGExample


class MTRAGPreprocessor:
    """Preprocessor for MTRAG data with chunking and cleaning."""
    
    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 128,
                 min_chunk_length: int = 50,
                 remove_headers: bool = True):
        """
        Args:
            chunk_size: Maximum tokens per passage chunk
            chunk_overlap: Overlap between chunks
            min_chunk_length: Minimum characters for a chunk
            remove_headers: Whether to remove common headers
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.remove_headers = remove_headers
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common headers/footers if enabled
        if self.remove_headers:
            # Remove page numbers
            text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
            # Remove common header patterns
            text = re.sub(r'(?:©|Copyright|All rights reserved).*?(?=\n|$)', '', text, flags=re.IGNORECASE)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        
        return text.strip()
    
    def chunk_document(self, text: str, doc_id: str) -> List[Dict[str, str]]:
        """Split document into overlapping chunks."""
        # First clean the text
        text = self.clean_text(text)
        
        if len(text) < self.min_chunk_length:
            return [{"doc_id": doc_id, "text": text, "chunk_id": f"{doc_id}_0"}]
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in sentences:
            sent_length = len(sent)
            
            if current_length + sent_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunk_id = f"{doc_id}_{len(chunks)}"
                chunks.append({
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "chunk_id": chunk_id
                })
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-self.chunk_overlay_sentences:]
                current_chunk = overlap_sentences + [sent]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sent)
                current_length += sent_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_id = f"{doc_id}_{len(chunks)}"
            chunks.append({
                "doc_id": doc_id,
                "text": chunk_text,
                "chunk_id": chunk_id
            })
        
        return chunks
    
    @property
    def chunk_overlay_sentences(self):
        """Number of sentences to keep for overlap based on token overlap."""
        # Rough estimate: 1 sentence ≈ 20 tokens, so overlap_sentences = chunk_overlap/20
        return max(1, self.chunk_overlap // 20)
    
    def process_example(self, example: MTRAGExample) -> MTRAGExample:
        """Process a single example with chunking."""
        # Chunk all corpus documents
        chunked_corpus = []
        for doc in example.corpus:
            chunks = self.chunk_document(doc["text"], doc["doc_id"])
            chunked_corpus.extend(chunks)
        
        # Update example with chunked corpus
        example.corpus = chunked_corpus
        
        # Clean conversation text
        for turn in example.conversation:
            turn["text"] = self.clean_text(turn["text"])
        
        example.final_question = self.clean_text(example.final_question)
        
        return example
    
    def process_dataset(self, dataset: MTRAGDataset) -> MTRAGDataset:
        """Process entire dataset."""
        processed_examples = []
        for ex in tqdm(dataset.examples, desc="Processing examples"):
            processed_examples.append(self.process_example(ex))
        
        dataset.examples = processed_examples
        return dataset
    
    def build_passage_lookup(self, dataset: MTRAGDataset) -> Dict[str, str]:
        """Build lookup dictionary from passage ID to text."""
        lookup = {}
        for ex in dataset.examples:
            for passage in ex.corpus:
                pid = passage.get("chunk_id", passage.get("doc_id"))
                lookup[pid] = passage["text"]
        return lookup
    
    def save_processed(self, dataset: MTRAGDataset, output_path: str):
        """Save processed dataset to JSON."""
        output = []
        for ex in dataset.examples:
            output.append({
                "task_id": ex.task_id,
                "conversation": ex.conversation,
                "final_question": ex.final_question,
                "corpus": ex.corpus,
                "relevant_passages": ex.relevant_passages,
                "domain": ex.domain,
                "question_type": ex.question_type,
                "answerability": ex.answerability,
                "multi_turn_type": ex.multi_turn_type
            })
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)