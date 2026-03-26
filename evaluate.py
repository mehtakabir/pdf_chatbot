"""
evaluate.py — Step 2: Evaluate Q&A using LangSmith built-in evaluators

Judge LLM: Claude via AWS Bedrock (same as chatbot)
No OpenAI API key needed.

Run this AFTER:
    1. Running collect_qa.py
    2. Filling in "reference_answer" in qa_logs.json

Usage:
    python evaluate.py
"""

import os
import json
import warnings
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langchain_aws import ChatBedrockConverse

from vectorstore import get_vectorstore
from chain import build_retriever, build_rag_chain
from config import AWS_REGION, MODEL_ID


# -------------------------------------------------------
# Settings
# -------------------------------------------------------

LOG_FILE        = "qa_logs.json"
DATASET_NAME    = "pdf_chatbot_eval_dataset"
EXPERIMENT_NAME = "rag_evaluation_v1"


# -------------------------------------------------------
# Step 1 — Load and Validate qa_logs.json
# -------------------------------------------------------

def load_and_validate_logs() -> list:
    """
    Load qa_logs.json and check all reference_answers are filled in.
    """

    if not Path(LOG_FILE).exists():
        raise FileNotFoundError(
            f"'{LOG_FILE}' not found.\n"
            "Please run: python collect_qa.py first."
        )

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        logs = json.load(f)

    if not logs:
        raise ValueError("qa_logs.json is empty. Run collect_qa.py first.")

    # Check all reference answers are filled in
    missing = []
    for log in logs:
        if not log.get("reference_answer", "").strip():
            missing.append(log["id"])

    if missing:
        raise ValueError(
            f"reference_answer is empty for entries: {missing}\n"
            f"Open '{LOG_FILE}' and fill in reference_answer for all entries."
        )

    print(f"Loaded {len(logs)} validated entries from {LOG_FILE}")
    return logs


# -------------------------------------------------------
# Step 2 — Upload Dataset to LangSmith
# -------------------------------------------------------

def create_dataset(client: Client, logs: list) -> str:
    """
    Upload QA logs as a dataset to LangSmith.

    inputs  : question + context
    outputs : reference_answer (written by you)
    """

    existing = [d.name for d in client.list_datasets()]

    if DATASET_NAME in existing:
        print(f"Dataset '{DATASET_NAME}' already exists. Deleting and recreating...")
        old = client.read_dataset(dataset_name=DATASET_NAME)
        client.delete_dataset(dataset_id=old.id)

    print(f"Creating dataset '{DATASET_NAME}' in LangSmith...")

    dataset = client.create_dataset(
        dataset_name = DATASET_NAME,
        description  = "PDF Chatbot RAG evaluation dataset with human reference answers"
    )

    inputs  = []
    outputs = []

    for log in logs:
        inputs.append({
            "question": log["question"],
            "context" : log["context"]
        })
        outputs.append({
            "answer": log["reference_answer"]
        })

    client.create_examples(
        inputs     = inputs,
        outputs    = outputs,
        dataset_id = dataset.id
    )

    print(f"Uploaded {len(logs)} examples to LangSmith.")
    return DATASET_NAME


# -------------------------------------------------------
# Step 3 — RAG Pipeline for Evaluation
# -------------------------------------------------------

def create_rag_pipeline():
    """
    Create RAG pipeline function for evaluation.
    LangSmith calls this with each example from the dataset.
    """

    print("Loading RAG pipeline...")
    vectorstore, all_chunks = get_vectorstore()
    retriever               = build_retriever(vectorstore, all_chunks)
    rag_chain               = build_rag_chain(retriever)

    def run_rag(inputs: dict) -> dict:
        answer = rag_chain.invoke(inputs["question"])
        return {"answer": answer}

    return run_rag


# -------------------------------------------------------
# Step 4 — Define Evaluators using Claude as Judge
# -------------------------------------------------------

def get_evaluators(judge_llm) -> list:
    """
    Define 3 LangSmith evaluators using Claude as the judge LLM.

    Why Claude as judge?
    --------------------
    Default LangSmith evaluators use GPT-3.5-turbo which needs
    an OpenAI API key. Since we don't have one, we pass our
    existing Claude model as the judge instead.

    Same AWS Bearer Token — no extra cost, no extra setup.

    Evaluators:
        1. Correctness  - answer vs your reference answer
        2. Faithfulness - answer vs retrieved context (no hallucination)
        3. Relevance    - answer vs question

    Args:
        judge_llm: ChatBedrockConverse instance used as judge

    Returns:
        list of evaluator objects
    """

    # ---- Evaluator 1: Correctness ----
    # Compares chatbot answer vs YOUR written reference answer
    # Uses your reference_answer as the ground truth
    correctness = LangChainStringEvaluator(
        "qa",
        config = {"llm": judge_llm},      # Claude as judge
        prepare_data = lambda run, example: {
            "prediction": run.outputs["answer"],
            "reference" : example.outputs["answer"],
            "input"     : example.inputs["question"]
        }
    )

    # ---- Evaluator 2: Faithfulness ----
    # Checks if answer is grounded in retrieved context
    # Score 0 if model adds info not present in chunks (hallucination)
    faithfulness = LangChainStringEvaluator(
        "context_qa",
        config = {"llm": judge_llm},      # Claude as judge
        prepare_data = lambda run, example: {
            "prediction": run.outputs["answer"],
            "reference" : example.inputs["context"],
            "input"     : example.inputs["question"]
        }
    )

    # ---- Evaluator 3: Relevance ----
    # Checks if answer actually addresses the question
    # Score 0 if answer is off-topic or doesn't answer the question
    relevance = LangChainStringEvaluator(
        "criteria",
        config = {
            "llm"     : judge_llm,        # Claude as judge
            "criteria": {
                "relevance": (
                    "Is the answer relevant and directly addressing the question? "
                    "Score 1 if yes, 0 if the answer is off-topic or does not answer the question."
                )
            }
        },
        prepare_data = lambda run, example: {
            "prediction": run.outputs["answer"],
            "input"     : example.inputs["question"]
        }
    )

    return [correctness, faithfulness, relevance]


# -------------------------------------------------------
# Step 5 — Run Evaluation
# -------------------------------------------------------

def run_evaluation():
    """Main evaluation function."""

    print("=" * 55)
    print("  LangSmith RAG Evaluator")
    print("  Judge LLM: Claude via AWS Bedrock")
    print("=" * 55)

    # Check API key
    if not os.environ.get("LANGCHAIN_API_KEY"):
        raise ValueError(
            "LANGCHAIN_API_KEY not found in .env\n"
            "Get your key from smith.langchain.com"
        )

    # Connect to LangSmith
    client = Client()
    print("Connected to LangSmith.\n")

    # Create Claude judge LLM
    # Uses same Bearer Token from .env — no extra setup needed
    print("Setting up Claude as judge LLM...")
    judge_llm = ChatBedrockConverse(
        model       = MODEL_ID,
        region_name = AWS_REGION,
        temperature = 0    # 0 = consistent scoring, no randomness
    )
    print("Judge LLM ready.\n")

    # Step 1 — Load logs
    logs = load_and_validate_logs()

    # Step 2 — Upload dataset
    dataset_name = create_dataset(client, logs)

    # Step 3 — Create RAG pipeline
    rag_pipeline = create_rag_pipeline()

    # Step 4 — Get evaluators with Claude as judge
    evaluators = get_evaluators(judge_llm)

    # Step 5 — Run evaluation
    print(f"\nRunning experiment: '{EXPERIMENT_NAME}'")
    print("Running chatbot on each question and scoring answers...")
    print("Please wait...\n")

    evaluate(
        rag_pipeline,
        data              = dataset_name,
        evaluators        = evaluators,
        experiment_prefix = EXPERIMENT_NAME,
        metadata = {
            "judge_llm"    : "Claude via AWS Bedrock",
            "retriever"    : "EnsembleRetriever (BM25 + MMR)",
            "model"        : "claude-3-5-sonnet via AWS Bedrock",
            "chunk_size"   : 600,
            "top_k"        : 3,
            "bm25_weight"  : 0.4,
            "vector_weight": 0.6
        }
    )

    print("\n" + "=" * 55)
    print("  Evaluation Complete!")
    print("=" * 55)
    print(f"\nTotal questions evaluated : {len(logs)}")
    print(f"View results at           : https://smith.langchain.com")
    print(f"Project                   : chatbot_bedrock")
    print(f"Experiment                : {EXPERIMENT_NAME}")
    print("\nScores: 1.0 = best  |  0.0 = worst")
    print("\nEvaluators:")
    print("  Correctness  : chatbot answer vs your reference answer")
    print("  Faithfulness : chatbot answer vs retrieved context")
    print("  Relevance    : chatbot answer vs question")


if __name__ == "__main__":
    run_evaluation()