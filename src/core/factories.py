from functools import lru_cache
from src.config.config_parser import settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from src.logging.logger import logger


class ModelFactory:
    @staticmethod
    @lru_cache(maxsize=1)
    def get_embedding():
        if settings.embedding_provider.lower() == "huggingface":
            logger.info(
                f"Initializing Embeddings Model (Provider: {settings.embedding_provider})..."
            )

            return HuggingFaceEmbeddings(
                model_name=settings.embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        else:
            raise ValueError(
                f"Unknown embedding provider: {settings.embedding_provider}"
            )

    @staticmethod
    @lru_cache(maxsize=1)
    def get_llm():
        if settings.llm_provider.lower() == "gemini":
            logger.info(
                f"Initializing LLM "
                f"(Provider: {settings.llm_provider}, "
                f"Model: {settings.llm_model_name})..."
            )

            return ChatGoogleGenerativeAI(
                model=settings.llm_model_name,
                max_output_tokens=settings.max_output_tokens,
                thinking_level=settings.thinking_level,
            )

        raise ValueError(
            f"Unsupported LLM provider: {settings.llm_provider}"
        )