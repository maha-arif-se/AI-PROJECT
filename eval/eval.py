"""
eval.py
-------
Runs every question in eval_set.json through the RAG pipeline, scores each
answer using Groq as an LLM judge, and reports aggregate numbers:
average score, latency, rough cost, and failure cases.

Run from the rag-project root folder:
    python eval/eval.py
"""

import json
import os
import sys
import time

# Allow importing rag_pipeline.py and config.py from the parent folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_pipeline import RAGPipeline
import config
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

EVAL_SET_PATH = os.path.join(os.path.dirname(__file__), "eval_set.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval_results.json")

JUDGE_MODEL = "openai/gpt-oss-20b"  # same model family used for generation

JUDGE_PROMPT_TEMPLATE = """You are grading whether a system's answer is correct.

Question: {question}
Expected answer: {expected}
System's actual answer: {actual}

Score the system's answer from 0 to 10, where:
- 10 = fully correct and matches the expected meaning
- 5 = partially correct or missing key details
- 0 = wrong, or contradicts the expected answer

If the expected answer is "NOT_IN_DOCUMENTS", the system should have said it
doesn't have enough information. Score 10 if it correctly declined to answer,
0 if it made up an answer instead.

Respond with ONLY a single number from 0 to 10, nothing else.
"""


def load_eval_set(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


import re


def judge_answer(client, question, expected, actual):
    """Uses Groq as an LLM judge. Returns an int score 0-10, or None on failure."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, expected=expected, actual=actual)
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        text = response.choices[0].message.content.strip()
        matches = re.findall(r'\b(10|[0-9])\b', text)
        if matches:
            score = int(matches[-1])
            return max(0, min(10, score))
        print(f"  [warn] Could not parse a score from judge response: '{text[:100]}'")
        return None
    except Exception as e:
        print(f"  [warn] Judge failed for this question: {e}")
        return None


def run_eval():
    print("Loading eval set...")
    eval_set = load_eval_set(EVAL_SET_PATH)
    print(f"  Loaded {len(eval_set)} question(s).\n")

    print("Loading RAG pipeline...")
    pipeline = RAGPipeline()

    judge_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    results = []

    for i, item in enumerate(eval_set, start=1):
        question = item["question"]
        expected = item["expected_answer"]
        print(f"[{i}/{len(eval_set)}] {question}")

        start_time = time.time()
        result = pipeline.answer(question)
        latency = time.time() - start_time

        actual = result["answer"]
        score = judge_answer(judge_client, question, expected, actual)

        results.append({
            "question": question,
            "expected_answer": expected,
            "actual_answer": actual,
            "sources": result["sources"],
            "score": score,
            "latency_seconds": round(latency, 2),
        })

        print(f"  Score: {score}/10 | Latency: {latency:.2f}s\n")

    return results


def summarize(results):
    scored = [r for r in results if r["score"] is not None]
    scores = [r["score"] for r in scored]
    latencies = [r["latency_seconds"] for r in results]

    avg_score = sum(scores) / len(scores) if scores else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0

    # Rough cost estimate: Groq free tier is $0 per token on most models,
    # but we estimate token usage for awareness in case you're on a paid tier.
    approx_tokens_per_request = 800  # rough context + response estimate
    total_requests = len(results) * 2  # 1 generation + 1 judge call per question
    approx_total_tokens = approx_tokens_per_request * total_requests

    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total questions:      {len(results)}")
    print(f"Average score:        {avg_score:.1f} / 10")
    print(f"Average latency:      {avg_latency:.2f}s")
    print(f"Min / Max latency:    {min_latency:.2f}s / {max_latency:.2f}s")
    print(f"Approx. total tokens: {approx_total_tokens} (rough estimate)")
    print(f"Approx. cost:         $0.00 (Groq free tier)")
    print()

    # Failure cases: score < 6
    failures = [r for r in results if r["score"] is not None and r["score"] < 6]
    print(f"Failure cases (score < 6): {len(failures)}")
    print("-" * 60)
    for f in failures:
        print(f"Q: {f['question']}")
        print(f"  Expected: {f['expected_answer']}")
        print(f"  Got:      {f['actual_answer'][:150]}...")
        print(f"  Score:    {f['score']}/10")
        print()

    return {
        "total_questions": len(results),
        "average_score": round(avg_score, 2),
        "average_latency_seconds": round(avg_latency, 2),
        "min_latency_seconds": round(min_latency, 2),
        "max_latency_seconds": round(max_latency, 2),
        "approx_total_tokens": approx_total_tokens,
        "approx_cost_usd": 0.00,
        "failure_count": len(failures),
    }


if __name__ == "__main__":
    results = run_eval()
    summary = summarize(results)

    output = {"summary": summary, "results": results}
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Full results saved to '{RESULTS_PATH}'")