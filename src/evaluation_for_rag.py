# Retrieval: Precision, Recall, F1-score, MRR
# Generation: BLEU, ROUGE, METEOR, Factual accuracy
# Pipeline: Latency, Throughput, Resource utilization, Token cost, Hallucination rate, User satisfaction

import time
import psutil
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
from datasets import load_metric

# ===== Retrieval Metrics =====
def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant: return 0.0
    hits = sum(1 for doc in retrieved[:k] if doc in relevant)
    return hits / len(relevant)

def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if k == 0: return 0.0
    hits = sum(1 for doc in retrieved[:k] if doc in relevant)
    return hits / k

def f1_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0: return 0.0
    return 2 * p * r / (p + r)

def reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0

# ===== Generation Metrics =====
bleu_metric = load_metric("bleu")
rouge_metric = load_metric("rouge")
meteor_metric = load_metric("meteor")

def compute_generation_metrics(answer: str, references: List[str]) -> Dict[str, float]:
    if not references:
        return {"BLEU": 0.0, "ROUGE-L": 0.0, "METEOR": 0.0, "FactualAcc": 0.0}

    # BLEU
    bleu = bleu_metric.compute(predictions=[answer.split()], references=[[ref.split()] for ref in references])["bleu"]

    # ROUGE
    rouge = rouge_metric.compute(predictions=[answer], references=[references[0]])
    rouge_l = rouge["rougeL"].mid.fmeasure

    # METEOR
    meteor = meteor_metric.compute(predictions=[answer], references=[references])["meteor"]

    # Factual accuracy: the overlap ratio of keywords (simple)
    answer_tokens = set(answer.lower().split())
    ref_tokens = set(" ".join(references).lower().split())
    factual_acc = len(answer_tokens & ref_tokens) / max(1, len(ref_tokens))

    return {"BLEU": bleu, "ROUGE-L": rouge_l, "METEOR": meteor, "FactualAcc": factual_acc}

# ===== Pipeline Metrics =====
def pipeline_metrics(latencies: List[float], token_counts: List[int], answers: List[str], references: List[List[str]]) -> Dict[str, float]:
    throughput = len(latencies) / sum(latencies) if latencies else 0.0
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    avg_tokens = float(np.mean(token_counts)) if token_counts else 0.0

    # Resource utilization
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent

    # Hallucination rate: the answer does not match any reference
    hallucinations = 0
    total = len(answers)
    for ans, refs in zip(answers, references):
        if not any(ref.lower() in ans.lower() for ref in refs):
            hallucinations += 1
    halluc_rate = hallucinations / total if total > 0 else 0.0

    # User satisfaction (placeholder = 1 - halluc_rate)
    satisfaction = 1 - halluc_rate

    return {
        "Latency": avg_latency,
        "Throughput": throughput,
        "Resource_CPU%": cpu,
        "Resource_RAM%": ram,
        "TokenCost": avg_tokens,
        "HallucRate": halluc_rate,
        "UserSatisfaction": satisfaction
    }