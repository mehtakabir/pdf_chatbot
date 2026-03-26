from langchain_aws import ChatBedrockConverse
from config import AWS_REGION, MODEL_ID


def get_llm():
    llm = ChatBedrockConverse(
        model       = MODEL_ID,
        region_name = AWS_REGION,
        temperature = 0   # 0 = consistent answers, no randomness
    )

    return llm