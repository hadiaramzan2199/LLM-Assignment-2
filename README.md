# MTRAGEval Baseline System for SemEval 2026 Task 8

## Project Overview
This repository contains baseline implementations for Tasks A (Retrieval Only) and B (Generation with Reference Passages) of the MTRAGEval shared task at SemEval 2026. The system implements:
- BM25 and dense retrieval baselines for Task A
- Prompt-based generation with Llama-3 and Qwen-2.5 for Task B
- Reproducible experimental protocol with fixed seeds
- Comprehensive evaluation metrics as specified by the task organizers

## Team Members
- Hadia Ramzan (hramzan.msai24secs@secs.edu.pk)
- Hareem Fatima Nagra (hnagra.mscs24secs@secs.edu.pk)

## Setup Instructions

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended: 24GB+ VRAM for 70B models)
- 16GB+ RAM

### Installation

1. Clone the repository:
```bash
git clone <https://github.com/hadiaramzan2199/LLM-Assignment-2>
cd project