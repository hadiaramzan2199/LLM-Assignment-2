"""Tests for retrieval models."""

import unittest
import tempfile
from pathlib import Path

from src.models.retrieval.bm25 import BM25Retriever


class TestBM25Retriever(unittest.TestCase):
    
    def setUp(self):
        self.retriever = BM25Retriever()
        self.corpus = [
            {"doc_id": "1", "text": "The capital of France is Paris."},
            {"doc_id": "2", "text": "Berlin is the capital of Germany."},
            {"doc_id": "3", "text": "Rome is the capital of Italy and has ancient history."},
            {"doc_id": "4", "text": "Madrid is the capital of Spain."},
        ]
    
    def test_build_index(self):
        self.retriever.build_index(self.corpus)
        self.assertEqual(self.retriever.index.corpus_size, 4)
    
    def test_retrieve(self):
        self.retriever.build_index(self.corpus)
        results = self.retriever.retrieve("What is the capital of France?", top_k=2)
        
        self.assertEqual(len(results), 2)
        # First result should be about France
        self.assertEqual(results[0][0], "1")
    
    def test_save_load(self):
        self.retriever.build_index(self.corpus)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.retriever.save(tmpdir)
            
            new_retriever = BM25Retriever()
            new_retriever.load(tmpdir)
            
            self.assertEqual(new_retriever.index.corpus_size, 4)
            self.assertEqual(new_retriever.passage_ids, ["1", "2", "3", "4"])


if __name__ == "__main__":
    unittest.main()