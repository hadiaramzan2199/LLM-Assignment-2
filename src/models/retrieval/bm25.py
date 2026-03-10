"""BM25 retrieval baseline for Task A."""

import json
import pickle
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.data.loader import MTRAGDataset, MTRAGExample


class BM25Retriever:
    """BM25 retriever for multi-turn RAG."""
    
    def __init__(self, 
                 k1: float = 1.5,
                 b: float = 0.75,
                 tokenizer=None):
        """
        Args:
            k1: BM25 parameter controlling term frequency saturation
            b: BM25 parameter controlling document length normalization
            tokenizer: Tokenizer function (default: simple whitespace)
        """
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or self._default_tokenizer
        self.index = None
        self.passage_ids = []
        self.passage_texts = []
    
    def _default_tokenizer(self, text: str) -> List[str]:
        """Simple whitespace tokenizer."""
        return text.lower().split()
    
    def build_index(self, corpus: List[Dict[str, str]]):
        """Build BM25 index from corpus passages.
        
        Args:
            corpus: List of passages with "doc_id" and "text" keys
        """
        print(f"Building BM25 index with {len(corpus)} passages...")
        
        # Extract texts and IDs
        self.passage_texts = [p["text"] for p in corpus]
        self.passage_ids = [p.get("chunk_id", p["doc_id"]) for p in corpus]
        
        # Tokenize all passages
        tokenized_corpus = [self.tokenizer(text) for text in tqdm(self.passage_texts, desc="Tokenizing")]
        
        # Build BM25 index
        self.index = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        
        print(f"Index built with {self.index.corpus_size} documents")
    
    def retrieve(self, 
                 query: str, 
                 conversation_history: Optional[str] = None,
                 top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve top-k passages for query.
        
        Args:
            query: Current user query
            conversation_history: Optional formatted conversation history
            top_k: Number of passages to retrieve
            
        Returns:
            List of (passage_id, score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Combine query with history if provided
        search_query = query
        if conversation_history:
            # Simple concatenation (can be improved with query rewriting)
            search_query = f"{conversation_history} {query}"
        
        # Tokenize query
        tokenized_query = self.tokenizer(search_query)
        
        # Get scores
        scores = self.index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Return passage IDs and scores
        results = [(self.passage_ids[i], float(scores[i])) for i in top_indices]
        
        return results
    
    def retrieve_batch(self,
                       examples: List[MTRAGExample],
                       top_k: int = 10,
                       use_history: bool = True) -> Dict[str, List[str]]:
        """Retrieve for multiple examples.
        
        Returns:
            Dictionary mapping task_id to list of passage IDs
        """
        results = {}
        
        for ex in tqdm(examples, desc="Retrieving"):
            # Format conversation history if needed
            history = None
            if use_history:
                # Simple concatenation of all previous turns
                history_texts = []
                for turn in ex.conversation[:-1]:  # Exclude final question
                    role = "User" if turn["role"] == "user" else "Assistant"
                    history_texts.append(f"{role}: {turn['text']}")
                history = " ".join(history_texts)
            
            retrieved = self.retrieve(
                query=ex.final_question,
                conversation_history=history,
                top_k=top_k
            )
            
            results[ex.task_id] = [pid for pid, _ in retrieved]
        
        return results
    
    def save(self, path: Union[str, Path]):
        """Save index to disk."""
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        
        # Save metadata
        metadata = {
            "k1": self.k1,
            "b": self.b,
            "passage_ids": self.passage_ids,
            "num_passages": len(self.passage_ids)
        }
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Save BM25 index (need to pickle since not JSON serializable)
        with open(path / "bm25_index.pkl", 'wb') as f:
            pickle.dump(self.index, f)
        
        # Save passage texts
        with open(path / "passage_texts.json", 'w') as f:
            json.dump(self.passage_texts, f)
    
    def load(self, path: Union[str, Path]):
        """Load index from disk."""
        path = Path(path)
        
        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.k1 = metadata["k1"]
        self.b = metadata["b"]
        self.passage_ids = metadata["passage_ids"]
        
        with open(path / "bm25_index.pkl", 'rb') as f:
            self.index = pickle.load(f)
        
        with open(path / "passage_texts.json", 'r') as f:
            self.passage_texts = json.load(f)
        
        return self