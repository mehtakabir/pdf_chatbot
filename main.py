import os
import warnings
import logging

os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from langsmith import traceable
from vectorstore import get_vectorstore
from chain import build_retriever, build_rag_chain


@traceable(name="answer_question")
def answer_question(rag_chain, question: str) -> str:
    """Traced — runs full RAG chain for one question."""
    return rag_chain.invoke(question)


def main():

    print("=" * 50)
    print("  Data Science PDF Chatbot")
    print("=" * 50)

    vectorstore, all_chunks = get_vectorstore()
    retriever               = build_retriever(vectorstore, all_chunks)
    rag_chain               = build_rag_chain(retriever)

    print("\nChatbot is ready! Type your question below.")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("Q: ").strip()

        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if not question:
            print("Please type a question.\n")
            continue

        print("\nThinking...\n")
        answer = answer_question(rag_chain, question)
        print(f"A: {answer}\n")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()