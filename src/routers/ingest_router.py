from fastapi import APIRouter, File, HTTPException, UploadFile, status
from src.services.ingest_service import IngestService
from src.logging.logger import logger
from src.config.config_parser import settings
from src.models.schemas import IngestResponse

router = APIRouter(
    prefix="/api/v1",
    tags=["ingest & index"],
    responses={404: {"description": "Not found"}},
)

ingest_service = IngestService()


@router.post(
    "/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED
)
async def upload_and_ingest(
    file: UploadFile = File(
        ..., description="Select text file to upload and index into FAISS"
    )
):
    """
    Controller Endpoint: Accepts document uploads via multipart form data, splits text, and indexes in FAISS.
    """
    try:
        logger.info(f"Received file upload endpoint hit for filename: {file.filename}")

        if not file.filename.endswith((".txt", ".md", ".csv")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only text files (.txt, .md, .csv) are supported.",
            )

        chunks_count = await ingest_service.process_uploaded_file(
            file, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )

        return IngestResponse(
            message=f"File '{file.filename}' uploaded and indexed into FAISS successfully.",
            chunks_indexed=chunks_count,
            index_path=settings.vector_index_path,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during file upload ingestion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process file upload: {str(e)}",
        )
