from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from config import TOP_K, FETCH_K, LAMBDA, BM25_WEIGHT, VECTOR_WEIGHT
from model import get_llm


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs[:TOP_K])


def build_retriever(vectorstore, all_chunks):
    """
    EnsembleRetriever combining both BM25 + Vector search.

    BM25   → finds exact keyword matches 
    Vector → finds semantically similar chunks
    

    Args:
        vectorstore : Chroma vectorstore object
        all_chunks  : list of all Document chunks (needed by BM25)

    Returns:
        EnsembleRetriever object
    """

    # BM25 — keyword search
    bm25_retriever   = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = TOP_K

    # Vector — semantic search with MMR
    vector_retriever = vectorstore.as_retriever(
        search_type   = "mmr",
        search_kwargs = {
            "k"          : TOP_K,
            "fetch_k"    : FETCH_K,
            "lambda_mult": LAMBDA
        }
    )

    # Combine both
    ensemble_retriever = EnsembleRetriever(
        retrievers = [bm25_retriever, vector_retriever],
        weights    = [BM25_WEIGHT, VECTOR_WEIGHT]
    )

    return ensemble_retriever


def build_rag_chain(retriever):
    """
    Build the RAG chain using LangChain components.

    Flow:
        Question
          -> EnsembleRetriever   finds relevant chunks (BM25 + Vector)
          -> format_docs         joins TOP_K best chunks into context
          -> ChatPromptTemplate  fills {context} and {question}
          -> ChatBedrockConverse sends to Claude, gets response
          -> StrOutputParser     returns plain string answer

    Returns:
        rag_chain: LangChain Runnable chain
    """

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Data Science tutor.

    Strict Instructions:
    1. Answer ONLY using the provided context.
    2. Do NOT use any external knowledge.
    3. If the answer is not present in the context, reply exactly:
    "I don't know based on the provided notes."
    4. Do NOT add extra explanations, assumptions, or examples.
    5. Keep answers concise and to the point.

    Formatting Rules:
    - If a definition is asked, return the exact definition from the context.
    - If a comparison or difference is asked, return the answer in a clear tabular format.
    - If the answer is factual, return it as short bullet points if needed.

    Important:
    - Do NOT hallucinate.
    - Do NOT rephrase unnecessarily.
    - Stay fully grounded in the context.
    """
        ),
        (
            "human",
            """Context:
    {context}

    Question:
    {question}
    """
        )
    ])

    rag_chain = (
        {
            "context" : retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain