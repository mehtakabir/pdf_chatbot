import os
from dotenv import load_dotenv

load_dotenv()


# LangSmith Tracing
# -------------------------------------------------------
os.environ["LANGCHAIN_PROJECT"] = "chatbot_bedrock"


# Folder Paths
# -------------------------------------------------------
PDF_FOLDER  = "Data_science_notes"
CHROMA_PATH = "chroma_db_pdf"


# Chunking Settings
# -------------------------------------------------------
CHUNK_SIZE    = 600
CHUNK_OVERLAP = 100


# Retrieval Settings
# -------------------------------------------------------
TOP_K    = 3
FETCH_K  = 20
LAMBDA   = 0.8

BM25_WEIGHT   = 0.4
VECTOR_WEIGHT = 0.6

# bedrock api 
# ---------------------------------------------------------
AWS_REGION = os.environ["AWS_DEFAULT_REGION"]
MODEL_ID   = "apac.anthropic.claude-3-5-sonnet-20240620-v1:0"