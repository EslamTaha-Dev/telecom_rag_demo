from fastapi import FastAPI
from src.config.config_parser import settings
from src.logging.logger import logger
from src.core.factories import ModelFactory


async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name} v{settings.app_version}...")
    ModelFactory.get_embedding()
    ModelFactory.get_llm()

    logger.info("Application started...")

    yield
    logger.info("Shutting down...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    lifespan=lifespan,
)


@app.get("/health", status_code=200)
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
