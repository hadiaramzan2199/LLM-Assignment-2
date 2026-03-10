"""Evaluation metrics for Tasks A and B."""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict

from sklearn.metrics import precision_score, recall_score, f1_score


def ndcg_at_k(relevance_scores: List[int], k: int) -> float:
    """Compute nDCG@k.
    
    Args:
        relevance_scores: List of relevance scores (graded) in ranked order
        k: Cutoff
        
    Returns:
        nDCG@k score
    """
    if not relevance_scores or k == 0:
        return 0.0
    
    # Take first k
    relevance_scores = relevance_scores[:k]
    
    # DCG
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
    
    # Ideal DCG (sort in descending order)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_scores))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Compute Precision@k."""
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    return len([doc for doc in retrieved_k if doc in relevant]) / len(retrieved_k)


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Compute Recall@k."""
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    return len([doc for doc in retrieved_k if doc in relevant]) / len(relevant)


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """Compute Reciprocal Rank."""
    for i, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def hit_rate_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Compute Hit Rate@k (whether any relevant doc in top-k)."""
    retrieved_k = retrieved[:k]
    return 1.0 if any(doc in relevant for doc in retrieved_k) else 0.0


def evaluate_retrieval(predictions: Dict[str, List[str]],
                       ground_truth: Dict[str, List[str]],
                       ks: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
    """Evaluate retrieval performance.
    
    Args:
        predictions: Dict mapping task_id to list of retrieved passage IDs
        ground_truth: Dict mapping task_id to list of relevant passage IDs
        
    Returns:
        Dictionary of metrics
    """
    metrics = defaultdict(list)
    
    for task_id, retrieved in predictions.items():
        if task_id not in ground_truth:
            continue
        
        relevant_set = set(ground_truth[task_id])
        
        # Compute graded relevance for nDCG (binary relevance)
        relevance_scores = [1 if doc in relevant_set else 0 for doc in retrieved]
        
        # nDCG
        for k in ks:
            if k <= len(relevance_scores):
                metrics[f"ndcg@{k}"].append(ndcg_at_k(relevance_scores, k))
        
        # Precision, Recall, Hit Rate
        for k in ks:
            metrics[f"precision@{k}"].append(precision_at_k(retrieved, relevant_set, k))
            metrics[f"recall@{k}"].append(recall_at_k(retrieved, relevant_set, k))
            metrics[f"hit_rate@{k}"].append(hit_rate_at_k(retrieved, relevant_set, k))
        
        # MRR
        metrics["mrr"].append(reciprocal_rank(retrieved, relevant_set))
    
    # Average metrics
    avg_metrics = {}
    for key, values in metrics.items():
        if values:
            avg_metrics[key] = np.mean(values)
        else:
            avg_metrics[key] = 0.0
    
    return avg_metrics


def evaluate_generation(predictions: Dict[str, str],
                       ground_truth: Dict[str, str],
                       metric_names: List[str] = ["rouge1", "rouge2", "rougeL", "bertscore"]) -> Dict[str, float]:
    """Evaluate generation performance.
    
    Note: For MTRAGEval Task B, the official metric is harmonic mean of R_LF, RB_llm, RB_alg.
    This is a placeholder for lexical metrics. The official evaluation will use the task's scripts.
    """
    # This is a simplified version - actual implementation would use
    # the official MTRAGEval evaluation scripts
    
    from evaluate import load
    
    metrics = {}
    
    if "rouge1" in metric_names or "rouge" in metric_names:
        rouge = load("rouge")
        rouge_scores = rouge.compute(
            predictions=list(predictions.values()),
            references=[ground_truth[tid] for tid in predictions.keys() if tid in ground_truth]
        )
        metrics.update(rouge_scores)
    
    if "bertscore" in metric_names:
        bertscore = load("bertscore")
        bert_scores = bertscore.compute(
            predictions=list(predictions.values()),
            references=[ground_truth[tid] for tid in predictions.keys() if tid in ground_truth],
            lang="en"
        )
        metrics["bertscore"] = np.mean(bert_scores["f1"])
    
    return metrics