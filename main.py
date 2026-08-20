from fastapi import FastAPI
from src.routers import ingest_router, query_router
from src.config.config_parser import settings
from src.logging.logger import logger
from src.core.factories import ModelFactory
from src.vectors.database import VectorDatabase


async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name} v{settings.app_version}...")
    ModelFactory.get_embedding()
    ModelFactory.get_llm()

    logger.info("Application started...")
    try:
        repo = VectorDatabase()
        repo.load_index()
        logger.info("FAISS vector database loaded into memory...")
    except Exception as e:
        logger.error(f"Failed to load FAISS vector database: {e}")

    yield  # sever runs here until shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    lifespan=lifespan,
)

app.include_router(ingest_router.router)
app.include_router(query_router.router)

@app.get("/", tags=["health check"], status_code=200)
def health_check():
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "docs_url": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
