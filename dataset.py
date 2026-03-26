"""
collect_qa.py — Step 1: Collect Q&A data for evaluation

Run this file to collect questions and answers from your chatbot.
After running, open qa_logs.json and add "reference_answer" for each entry.
Then run evaluate.py for scoring.

Usage:
    python collect_qa.py
"""

import os
import json
import warnings
import logging
from datetime import datetime
from pathlib import Path

# Suppress ChromaDB warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from vectorstore import get_vectorstore
from chain import build_retriever, build_rag_chain, format_docs


LOG_FILE = "qa_logs.json"


def load_logs() -> list:
    """Load existing logs or return empty list."""
    if Path(LOG_FILE).exists():
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_log(question: str, context: str, answer: str):
    """
    Save one Q&A entry to qa_logs.json.

    Each entry has:
        id               : auto incremented number
        time             : when the question was asked
        question         : what you asked
        context          : chunks retrieved from vector DB
        answer           : chatbot answer
        reference_answer : YOU fill this in manually after collection
    """
    logs = load_logs()

    entry = {
        "id"              : len(logs) + 1,
        "time"            : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question"        : question,
        "context"         : context,
        "answer"          : answer,
        "reference_answer": ""   # ← you fill this in manually
    }

    logs.append(entry)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    print(f"  [Saved] Entry #{entry['id']} saved to {LOG_FILE}")


def main():

    print("=" * 55)
    print("  Q&A Data Collection for Evaluation")
    print("=" * 55)
    print(f"Answers will be saved to: {LOG_FILE}")
    print("After collection, open the file and fill in")
    print("'reference_answer' for each entry manually.\n")

    # Load RAG pipeline
    vectorstore, all_chunks = get_vectorstore()
    retriever               = build_retriever(vectorstore, all_chunks)
    rag_chain               = build_rag_chain(retriever)

    print("\nChatbot ready! Ask your questions.")
    print("Type 'done' when finished.\n")
    print("Tip: Ask 8-10 questions covering different topics")
    print("     from your PDF notes for better evaluation.\n")

    while True:

        question = input("Q: ").strip()

        if question.lower() in ("done", "exit", "quit"):
            break

        if not question:
            print("Please type a question.\n")
            continue

        print("\nThinking...\n")

        # Get relevant chunks
        docs    = retriever.invoke(question)
        context = format_docs(docs)

        # Get answer from chatbot
        answer  = rag_chain.invoke(question)

        print(f"A: {answer}\n")
        print("-" * 55 + "\n")

        # Save to file
        save_log(question, context, answer)

    # Show final instructions
    total = len(load_logs())
    print("\n" + "=" * 55)
    print(f"  Collection Complete! {total} questions saved.")
    print("=" * 55)
    print(f"\nNext steps:")
    print(f"  1. Open '{LOG_FILE}'")
    print(f"  2. For each entry, fill in 'reference_answer'")
    print(f"     with your own correct answer (1-2 sentences)")
    print(f"  3. Save the file")
    print(f"  4. Run: python evaluate.py")
    print()
    print("Example of what to fill in:")
    print('''  "reference_answer": "Pandas is a Python library for''')
    print('''   data manipulation with Series and DataFrame structures."''')


if __name__ == "__main__":
    main()