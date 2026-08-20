from fastapi import APIRouter, HTTPException, status
from src.logging.logger import logger
from src.services.rag_service import RAGService
from src.models.schemas import QueryRequest, QueryResponse

router = APIRouter(
    prefix="/api/v1",
    tags=["RAG Query"],
)
rag_service = RAGService()


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
def process_query(request: QueryRequest):
    """
    Controller Endpoint: Accepts customer tickets, runs RAG retrieval + LLM generation, and returns response.
    """
    try:
        if not request.ticket.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Ticket is empty."
            )

        result = rag_service.answer_ticket(request.ticket)
        return QueryResponse(**result)

    except HTTPException:
        raise

    except FileNotFoundError as e:
        logger.error(f"FAISS Index missing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vector database index not found. Please upload a file via /api/v1/ingest first.",
        )

    except Exception as e:
        logger.error(f"Failed to process query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}",
        )
