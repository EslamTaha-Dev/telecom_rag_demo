from fastapi import UploadFile
from src.logging.logger import logger
from src.vectors.database import VectorDatabase
from src.config.config_parser import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class IngestService:
    def __init__(self):
        self.repo = VectorDatabase()

    async def process_uploaded_file(
        self, file: UploadFile, chunk_size: int = 500, chunk_overlap: int = 100
    ):
        logger.info(f"Reading uploaded file: {file.filename}...")
        content = await file.read()

        text_content = content.decode("utf-8").replace("\r\n", "\n")

        logger.info(
            f"Splitting text into {chunk_size} , {chunk_overlap} character chunks..."
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\r\n\r\n", "\r\n", "\n", " ", ""],
        )
        docs = text_splitter.create_documents(
            texts=[text_content], metadatas=[{"source": file.filename}]
        )
        logger.info(f"Generated {len(docs)} text chunks from '{file.filename}'.")

        self.repo.add_documents(docs)
        return len(docs)
