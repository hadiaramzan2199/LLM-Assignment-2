"""Dense retrieval baseline using sentence transformers."""

import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util
import torch.nn.functional as F

from src.data.loader import MTRAGExample


class DenseRetriever:
    """Dense retriever using sentence transformers."""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 device: Optional[str] = None,
                 batch_size: int = 32):
        """
        Args:
            model_name: HuggingFace model name
            device: cuda or cpu
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading dense retriever {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        self.passage_ids = []
        self.passage_embeddings = None
        self.passage_texts = []
    
    def encode_passages(self, passages: List[Dict[str, str]]) -> np.ndarray:
        """Encode passages into embeddings.
        
        Args:
            passages: List of passage dicts with "text" key
            
        Returns:
            numpy array of embeddings
        """
        texts = [p["text"] for p in passages]
        
        # Encode in batches
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding passages"):
            batch_texts = texts[i:i+self.batch_size]
            embeddings = self.model.encode(
                batch_texts, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
            all_embeddings.append(embeddings.cpu())
        
        if all_embeddings:
            self.passage_embeddings = torch.cat(all_embeddings, dim=0)
        else:
            self.passage_embeddings = torch.tensor([])
        
        return self.passage_embeddings.numpy()
    
    def build_index(self, corpus: List[Dict[str, str]]):
        """Build dense index from corpus."""
        print(f"Building dense index with {len(corpus)} passages...")
        
        self.passage_ids = [p.get("chunk_id", p["doc_id"]) for p in corpus]
        self.passage_texts = [p["text"] for p in corpus]
        
        # Encode passages
        self.encode_passages(corpus)
        
        print(f"Index built with {len(self.passage_ids)} passages")
    
    def retrieve(self,
                 query: str,
                 conversation_history: Optional[str] = None,
                 top_k: int = 10,
                 return_scores: bool = True) -> List[Tuple[str, float]]:
        """Retrieve top-k passages for query.
        
        Args:
            query: Current user query
            conversation_history: Optional formatted conversation history
            top_k: Number of passages to retrieve
            return_scores: Whether to return scores
            
        Returns:
            List of (passage_id, score) tuples
        """
        if self.passage_embeddings is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Combine query with history
        search_query = query
        if conversation_history:
            search_query = f"{conversation_history} {query}"
        
        # Encode query
        query_embedding = self.model.encode(
            search_query, 
            convert_to_tensor=True,
            device=self.device
        )
        
        # Compute similarities
        scores = util.cos_sim(query_embedding, self.passage_embeddings)[0]
        
        # Get top-k
        top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
        
        results = []
        for score, idx in zip(top_scores, top_indices):
            results.append((self.passage_ids[idx], float(score)))
        
        return results
    
    def retrieve_batch(self,
                       examples: List[MTRAGExample],
                       top_k: int = 10,
                       use_history: bool = True) -> Dict[str, List[str]]:
        """Retrieve for multiple examples."""
        results = {}
        
        for ex in tqdm(examples, desc="Dense retrieval"):
            history = None
            if use_history:
                history_texts = []
                for turn in ex.conversation[:-1]:
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
            "model_name": self.model_name,
            "passage_ids": self.passage_ids,
            "num_passages": len(self.passage_ids)
        }
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Save embeddings
        if self.passage_embeddings is not None:
            torch.save(self.passage_embeddings, path / "embeddings.pt")
        
        # Save passage texts
        with open(path / "passage_texts.json", 'w') as f:
            json.dump(self.passage_texts, f)
    
    def load(self, path: Union[str, Path]):
        """Load index from disk."""
        path = Path(path)
        
        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.model_name = metadata["model_name"]
        self.passage_ids = metadata["passage_ids"]
        
        # Reload model (in case different)
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
        # Load embeddings
        emb_path = path / "embeddings.pt"
        if emb_path.exists():
            self.passage_embeddings = torch.load(emb_path)
        
        with open(path / "passage_texts.json", 'r') as f:
            self.passage_texts = json.load(f)
        
        return self