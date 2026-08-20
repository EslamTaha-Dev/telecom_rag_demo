import os
from langchain_community.vectorstores import FAISS
from src.logging.logger import logger
from src.config.config_parser import settings
from src.core.factories import ModelFactory


class VectorDatabase:
    cache = None

    def __init__(self):
        self.embeddings = ModelFactory.get_embedding()
        self.index_path = settings.vector_index_path

    def save_index(self, vectorstore: FAISS):
        logger.info(f"Saving vector database to {self.index_path}...")
        vectorstore.save_local(self.index_path)
        VectorDatabase.cache = vectorstore

    def load_index(self):
        if VectorDatabase.cache is not None:
            logger.info("Using in-memory cached vector database...")
            return VectorDatabase.cache

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"FAISS index folder '{self.index_path}' not found! Upload a file via /api/v1/ingest first."
            )

        logger.info(f"Loading vector database from {self.index_path}...")

        vectorstore = FAISS.load_local(
            self.index_path, self.embeddings, allow_dangerous_deserialization=True
        )

        VectorDatabase.cache = vectorstore
        return vectorstore

    def add_documents(self, documents: list, batch_size: int = None) -> FAISS:
        batch_size = batch_size or settings.batch_size
        total_chunks = len(documents)
        total_batches = (total_chunks + batch_size - 1) // batch_size

        logger.info(
            f"Indexing {total_chunks} chunks into FAISS in {total_batches} batches (batch_size={batch_size})..."
        )
        vectorstore = None

        for i in range(0, total_chunks, batch_size):
            batch = documents[i : i + batch_size]
            current_batch_num = i // batch_size + 1
            logger.info(
                f"Embedding & indexing batch {current_batch_num}/{total_batches} ({len(batch)} chunks)..."
            )

            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, self.embeddings)
            else:
                vectorstore.add_documents(batch)

        self.save_index(vectorstore)
        return vectorstore
